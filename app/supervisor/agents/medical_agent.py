from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prompts.agent_prompts.medical_prompt import MedicalPrompt
from supervisor.base import BaseAgent
from supervisor.graph_state import AgentRequest, AgentState


class MedicalAgent(BaseAgent):
    def __init__(self, req: AgentRequest):
        super().__init__(req)
        if not req.model:
            raise ValueError("医疗模型初始化失败，请检查配置。")
        self.llm = req.model
        self.subgraph_id = "medical_agent"
        self.system_prompt = MedicalPrompt.SYSTEM
        self.tools = []
        self.graph = self._build_graph()

    async def _model_node(self, state: AgentState, config: RunnableConfig):
        messages = list(state.get("messages", []) or [])
        llm_with_tools = self.llm.bind_tools(self.tools) if self.tools else self.llm
        response = await llm_with_tools.ainvoke(messages, config=config)
        response.content = f"{self._message_text(response)}{MedicalPrompt.DISCLAIMER}"
        return {"messages": [response]}

    def _build_graph(self) -> Runnable:
        workflow = StateGraph(AgentState)
        workflow.add_node("model_node", self._model_node, retry_policy=self.RETRY_POLICY)
        workflow.add_node("tools", ToolNode(self.tools), retry_policy=self.RETRY_POLICY)
        workflow.add_edge(START, "model_node")
        workflow.add_conditional_edges("model_node", tools_condition)
        workflow.add_edge("tools", "model_node")
        return workflow.compile(checkpointer=self.checkpointer)
