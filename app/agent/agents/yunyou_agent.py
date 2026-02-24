import os
from datetime import datetime
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from agent.base import BaseAgent
from agent.graph_state import AgentRequest
from agent.graphs.checkpointer import checkpointer
from agent.tools.yunyou_tools import holter_list, holter_report_count, holter_type_count
from schemas.base_skill import ConfigurableSkillMiddleware
from utils.custom_logger import get_logger

log = get_logger(__name__)

"""
【模块说明】
云柚业务子图。展示了 ToolNode 调用、人工中断审批(interrupt) 以及状态修剪的高阶玩法。
"""

class YunyouState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

class YunyouAgent(BaseAgent):
    SENSITIVE_TOOLS = {"holter_report_count"}

    def __init__(self, req: AgentRequest):
        super().__init__(req)
        if not req.model:
            raise ValueError("Yunyou Agent 模型初始化失败。")
        self.llm = req.model
        self.graph = self._build_graph()

    def _build_system_prompt(self, base_prompt: str) -> str:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_weekday = datetime.now().strftime("%A")
        rules = (
            f"\n\n# 系统信息\n当前时间：{current_time} ({current_weekday})\n\n"
            "# 执行规则\n"
            "1. 优先调用工具获取真实数据。\n"
            "2. 用户说今天/昨天/本周时，参考上述当前时间转为具体日期 YYYY-MM-DD。\n"
            "3. 参数不足时必须追问，不得猜测。\n"
        )
        return (base_prompt or "你是云柚业务助手。") + rules

    def _build_graph(self) -> Runnable:
        skill_source = os.getenv("SKILL_YUNYOU", "")
        skill_middleware = ConfigurableSkillMiddleware(skill_source) if skill_source else None

        base_tools = [holter_list, holter_type_count, holter_report_count]
        skill_tools = skill_middleware.get_tools() if skill_middleware else []
        all_tools = base_tools + skill_tools

        self.model_with_tools = self.llm.bind_tools(all_tools)
        sys_prompt = self._build_system_prompt(skill_middleware.get_prompt() if skill_middleware else "你是助手。")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        def model_node(state: YunyouState):
            # 【核心修复】防止国产大模型 (GLM/Qwen) 因不支持 tiktoken 导致崩溃。
            # 直接使用 len 截断消息轮数（保留最近 8 条消息 = 4 轮对话）。
            trimmed_messages = trim_messages(
                state["messages"],
                max_tokens=8,
                strategy="last",
                token_counter=len,
                include_system=True,
                allow_partial=False
            )
            chain = self.prompt | self.model_with_tools
            response = chain.invoke({"messages": trimmed_messages})
            return {"messages": [response]}

        def human_review_node(state: YunyouState):
            last_message = state["messages"][-1]
            if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
                return Command(goto=END)

            sensitive_calls = [tc for tc in last_message.tool_calls if tc["name"] in self.SENSITIVE_TOOLS]
            if not sensitive_calls:
                return Command(goto="tools")

            decision = interrupt({
                "message": "检测到敏感操作，请审核。",
                "allowed_decisions": ["approve", "reject"],
                "action_requests": [{
                    "type": "tool_approval", "name": call["name"], "args": call["args"],
                    "description": "⚠️ 敏感业务数据操作，需审批。", "id": call["id"],
                } for call in sensitive_calls],
            })

            # 【学习笔记】正确处理拒绝逻辑：构造假 ToolMessage 回传给大模型
            if decision == "reject" or (isinstance(decision, dict) and decision.get("action") == "reject"):
                rejection_messages = [
                    ToolMessage(
                        tool_call_id=call["id"], name=call["name"],
                        content="Error: 管理员已拒绝执行该敏感操作。请婉拒用户的请求。"
                    ) for call in last_message.tool_calls
                ]
                return Command(goto="agent", update={"messages": rejection_messages})

            return Command(goto="tools")

        workflow = StateGraph(YunyouState)
        workflow.add_node("agent", model_node)
        workflow.add_node("human_review", human_review_node)
        workflow.add_node("tools", ToolNode(all_tools))

        workflow.add_edge(START, "agent")
        workflow.add_edge("agent", "human_review")
        workflow.add_edge("tools", "agent")
        return workflow.compile(checkpointer=self.checkpointer)