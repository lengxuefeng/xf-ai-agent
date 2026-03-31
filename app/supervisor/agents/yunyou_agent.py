import os

from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from models.schemas.base_skill import ConfigurableSkillMiddleware
from prompts.agent_prompts.yunyou_prompt import YunyouPrompt
from supervisor.base import BaseAgent
from supervisor.graph_state import AgentRequest, AgentState
from tools.agent_tools.yunyou_tools import (
    holter_list,
    holter_log_info,
    holter_recent_db,
    holter_report_count,
    holter_type_count,
)


class YunyouAgent(BaseAgent):
    def __init__(self, req: AgentRequest):
        super().__init__(req)
        if not req.model:
            raise ValueError("Yunyou Agent 模型初始化失败。")
        self.llm = req.model
        self.subgraph_id = "yunyou_agent"
        self.system_prompt = (
            f"{YunyouPrompt.DEFAULT_SYSTEM_ROLE}"
            "优先调用真实业务工具获取数据；只有拿到工具结果后再给用户结论。"
        )

        skill_source = os.getenv("SKILL_YUNYOU", "")
        skill_middleware = ConfigurableSkillMiddleware(skill_source) if skill_source else None
        base_tools = [holter_list, holter_recent_db, holter_type_count, holter_report_count, holter_log_info]
        skill_tools = skill_middleware.get_tools() if skill_middleware else []
        self.tools = base_tools + skill_tools
        self.graph = self._build_graph()

    async def _model_node(self, state: AgentState, config: RunnableConfig):
        messages = list(state.get("messages", []) or [])
        llm_with_tools = self.llm.bind_tools(self.tools)
        response = await llm_with_tools.ainvoke(messages, config=config)
        return {"messages": [response]}

    def _build_graph(self) -> Runnable:
        workflow = StateGraph(AgentState)
        workflow.add_node("model_node", self._model_node, retry_policy=self.RETRY_POLICY)
        workflow.add_node("tools", ToolNode(self.tools), retry_policy=self.RETRY_POLICY)
        workflow.add_edge(START, "model_node")
        workflow.add_conditional_edges("model_node", tools_condition)
        workflow.add_edge("tools", "model_node")
        return workflow.compile(checkpointer=self.checkpointer)
