import io
from typing import TypedDict, Annotated

from IPython.core.display_functions import display

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode

from agent.graph_state import AgentRequest
from agent.llm.loader_llm_multi import load_open_router
from agent.llm.ollama_model import load_ollama_model
from agent.tools.search_tools import tavily_search_tool
from IPython.display import Image as IPyImage  # 重命名为 IPyImage
from PIL import Image as PILImage  # 重命名为 PILImage，避免冲突

from utils import redis_manager, concurrent_executor

load_dotenv()

"""
定义和创建网络搜索子图（Web Search Agent）。

功能：
    1.当主图（Supervisor）判断需要从互联网获取最新信息时，接收任务。
    2.利用语言模型（LLM）的函数调用（Function Calling）或工具使用（Tool Using）能力，
    3.执行搜索工具。
    4.将搜索结果整合后，生成一个自然的语言回答，并返回给主图。
"""


class SearchAgentStat(TypedDict):
    """
    搜索子图状态
    """
    messages: Annotated[list, add_messages]


class SearchAgent:
    def __init__(self, req: AgentRequest):
        if not req.model:
            raise ValueError("搜索模型初始化失败，请检查配置。")
        self.llm = req.model
        self.redis_manager = redis_manager.RedisManager()
        self.session_id = req.session_id
        self.subgraph_id = "search_agent"
        self.graph = self._build_graph()

    def _agent_node(self, state: SearchAgentStat):
        """
          定义智能体节点（Agent Node）。

          这个节点是决策点。它调用模型，模型会根据对话历史决定：
          1.  直接生成一个回答（如果信息足够）。
          2.  生成一个工具调用请求（如果需要搜索）。
          3.  在工具执行后，根据工具返回的结果生成最终回答。

          Args:
              state (SearchAgentState): 当前图的状态。

          Returns:
              dict: 包含模型生成的新消息（可能是回答，也可能是工具调用）的字典。
          """

        last_messages = state["messages"][-1]

        # 如果最后一条消息是工具执行的结果 (ToolMessage)，
        if isinstance(last_messages, ToolMessage):
            response = self.llm.invoke(state["messages"])
        else:
            model_with_tools = self.llm.bind_tools([tavily_search_tool])
            response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # 定义路由逻辑
    def _router(self, state: SearchAgentStat):
        """
        定义路由函数，在 agent 节点执行后决定下一步走向。

        Args:
            state (SearchAgentState): 当前图的状态。

        Returns:
            str: 下一个节点的名称。'tools' 表示需要执行工具，'__end__' 表示结束。
        """

        last_messages = state["messages"][-1]

        # # 检查这条消息中是否包含工具调用请求
        if hasattr(last_messages, "tool_calls") and last_messages.tool_calls:
            # 如果有，流程应该走向 "tools" 节点去执行工具
            return "tools"
        else:
            # 如果没有，说明 AI 已经生成了最终答案，流程结束
            return "__end__"

    def _build_graph(self):
        """
        构建搜索子图
        """
        graph = StateGraph(SearchAgentStat)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode([tavily_search_tool]))
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", self._router)
        graph.add_edge("tools", "agent")
        return graph.compile()

    def run(self, req: AgentRequest):
        state = self.redis_manager.load_graph_state(req.session_id, subgraph_id=self.subgraph_id)
        if not state:
            state = {"messages": []}

        last_msg = state["messages"][-1] if state["messages"] else None
        if not last_msg or getattr(last_msg, "content", None) != req.user_input:
            state["messages"].append(HumanMessage(content=req.user_input))
        result = self.graph.invoke(state)
        self.redis_manager.save_graph_state(result, req.session_id, req.subgraph_id)
        return result


if __name__ == '__main__':
    # llm = load_open_router("deepseek/deepseek-chat-v3-0324:free")
    # llm = load_ollama_model("gemma3:270m")
    llm = load_ollama_model("qwen3:8b")
    req = AgentRequest(
        user_input="最新的AI模型是什么",
        model=llm,
        session_id="123232132",
        subgraph_id="search_agent",
    )
    search_agent = SearchAgent(req)
    result = search_agent.run(req)
    print(result["messages"][-1].content)
