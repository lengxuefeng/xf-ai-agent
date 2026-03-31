from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prompts.agent_prompts.tools_prompt import ToolsPrompt
from supervisor.base import BaseAgent
from app.supervisor.graph_state import AgentRequest, AgentState
from tools.agent_tools.weather_tools import get_weathers


class WeatherAgent(BaseAgent):
    def __init__(self, req: AgentRequest):
        super().__init__(req)
        if not req.model:
            raise ValueError("天气模型初始化失败，请检查配置。")
        self.llm = req.model
        self.subgraph_id = "weather_agent"
        self.system_prompt = (
            f"{ToolsPrompt.WEATHER_SYSTEM}"
            "如果需要调用天气工具，请传入 `city_names` 数组。"
        )
        self.tools = [get_weathers]
        self.graph = self._build_graph()

    async def _model_node(self, state: AgentState, config: RunnableConfig):
        messages = state.get("messages", [])
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
