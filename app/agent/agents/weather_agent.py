from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode

from agent.base import BaseAgent
from agent.graph_state import AgentRequest
from agent.tools.weather_tools import get_weathers
from utils.custom_logger import get_logger

log = get_logger(__name__)

class WeatherAgentState(TypedDict):
    """
    天气子图状态
    """
    messages: Annotated[List[BaseMessage], add_messages]


class WeatherAgent(BaseAgent):
    """
    天气查询 Agent (LangGraph 1.0 Refactored)
    """

    def __init__(self, req: AgentRequest):
        super().__init__(req)
        if not req.model:
            raise ValueError("天气模型初始化失败，请检查配置。")
        self.llm = req.model
        self.subgraph_id = "weather_agent"
        
        # 工具
        self.tools = [get_weathers]
        self.model_with_tools = self.llm.bind_tools(self.tools)
        
        # 提示词
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个天气查询助手。请使用工具查询天气信息。"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> Runnable:
        workflow = StateGraph(WeatherAgentState)

        def model_node(state: WeatherAgentState):
            chain = self.prompt | self.model_with_tools
            return {"messages": [chain.invoke(state)]}

        tool_node = ToolNode(self.tools)

        workflow.add_node("agent", model_node)
        workflow.add_node("tools", tool_node)

        workflow.add_edge(START, "agent")
        
        from langgraph.prebuilt import tools_condition
        workflow.add_conditional_edges("agent", tools_condition)
        
        workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=self.checkpointer)
