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
云柚业务相关的子图(Agent)实现。

展示了如何将工具(ToolNode)调用、人工中断与审批(Interrupt)
以及通过状态修剪防止上下文溢出的高阶功能结合。
"""

class YunyouState(TypedDict):
    """
    云柚业务图的状态对象。

    Attributes:
        messages: 对话消息列表，支持通过 add_messages 自动累加。
    """
    messages: Annotated[List[BaseMessage], add_messages]

class YunyouAgent(BaseAgent):
    """
    云柚业务领域智能体，负责处理与云柚数据分析和处理相关的问题。

    通过绑定包含企业敏感操作约束的工具集合来解答问题。
    """
    SENSITIVE_TOOLS = {"holter_report_count"}

    def __init__(self, req: AgentRequest):
        """
        初始化云柚智能体并创建相关工作流子图。

        Args:
            req (AgentRequest): 用户请求上下文，如包含绑定的语言模型。
        """
        super().__init__(req)
        if not req.model:
            raise ValueError("Yunyou Agent 模型初始化失败。")
        self.llm = req.model
        self.graph = self._build_graph()

    def _build_system_prompt(self, base_prompt: str) -> str:
        """
        生成智能体所需的系统提示词 (System Prompt)。

        补充当前系统时间等实时上下文维度，并强化遵守工具调用的指令。

        Args:
            base_prompt (str): 从中间件或默认项获取的基础提示词。

        Returns:
            str: 拼接增强规则后的最终系统提示词。
        """
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
        """
        构建并返回云柚业务的状态图 (StateGraph)。

        整合基础分析工具与自定义技能工具，并配置人工审批的人机回环 (HITL)。

        Returns:
            Runnable: 编译后的 LangGraph 运行图。
        """
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
            """
            模型处理节点。接收并响应最新对话。
            """
            # 防止部分国产大模型因内部不支持 tiktoken 而崩溃。
            # 这里统一使用内置 len 方法截断消息轮数（保留最近 8 条消息即 4 轮对话）。
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
            """
            人工审批节点。检查生成的消息是否需要调用受保护的敏感工具。
            如果需要调用，向调用方抛出 interrupt 命令暂停执行，等待确认。
            """
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

            # 正确处理拒绝逻辑：构造假 ToolMessage 回传给大模型
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