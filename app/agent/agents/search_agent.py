from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode

from agent.base import BaseAgent
from agent.graph_state import AgentRequest
from agent.tools.search_tools import tavily_search_tool
from utils.custom_logger import get_logger

log = get_logger(__name__)

class SearchAgentState(TypedDict):
    """
    搜索子图状态
    """
    messages: Annotated[List[BaseMessage], add_messages]


class SearchAgent(BaseAgent):
    """
    网络搜索 Agent (LangGraph 1.0 Refactored)
    """

    def __init__(self, req: AgentRequest):
        super().__init__(req)
        if not req.model:
            raise ValueError("搜索模型初始化失败，请检查配置。")
        self.llm = req.model
        self.subgraph_id = "search_agent"
        
        # 工具
        self.tools = [tavily_search_tool]
        self.model_with_tools = self.llm.bind_tools(self.tools)
        
        # 提示词
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个有用的网络搜索助手，能够使用搜索工具查找信息并回答用户的问题。"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> Runnable:
        workflow = StateGraph(SearchAgentState)

        def model_node(state: SearchAgentState):
            chain = self.prompt | self.model_with_tools
            return {"messages": [chain.invoke(state)]}

        tool_node = ToolNode(self.tools)

        workflow.add_node("agent", model_node)
        workflow.add_node("tools", tool_node)

        workflow.add_edge(START, "agent")
        
        # 条件边：如果模型决定调用工具，去 tools；否则结束
        from langgraph.pregel import tools_condition
        workflow.add_conditional_edges("agent", tools_condition)
        
        workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=self.checkpointer)
