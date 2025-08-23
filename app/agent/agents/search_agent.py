# -*- coding: utf-8 -*-
from typing import TypedDict, Annotated, Generator

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages

from agent.agent_builder import create_tool_agent_executor
from agent.graph_state import AgentRequest
from agent.llm.loader_llm_multi import load_open_router
from agent.llm.ollama_model import load_ollama_model
from agent.tools.search_tools import tavily_search_tool
from utils import redis_manager

load_dotenv()

"""
定义和创建网络搜索子图（Web Search Agent）。

功能：
    1. 当主图（Supervisor）判断需要从互联网获取最新信息时，接收任务。
    2. 利用语言模型（LLM）的函数调用（Function Calling）或工具使用（Tool Using）能力，
    3. 执行搜索工具。
    4. 将搜索结果整合后，生成一个自然的语言回答，并返回给主图。
"""


class SearchAgentStat(TypedDict):
    """
    搜索子图状态
    """
    messages: Annotated[list, add_messages]


class SearchAgent:
    """
    网络搜索 Agent。

    本 Agent 的核心职责是处理需要从互联网获取信息的任务。
    它利用一个通用的 `create_tool_agent_executor` 构建器来创建一个具备工具使用能力的图。
    """

    def __init__(self, req: AgentRequest):
        """
        初始化搜索 Agent。

        Args:
            req (AgentRequest): 包含模型、会话 ID 等信息的请求对象。
        """
        if not req.model:
            raise ValueError("搜索模型初始化失败，请检查配置。")

        # 定义 Agent 可用的工具
        tools = [tavily_search_tool]

        # 使用通用构建器创建图执行器
        self.graph = create_tool_agent_executor(
            model=req.model,
            tools=tools,
            state_class=SearchAgentStat
        )

        # 初始化状态管理器
        self.redis_manager = redis_manager.RedisManager()
        self.session_id = req.session_id
        self.subgraph_id = "search_agent"

    def run(self, req: AgentRequest) -> Generator:
        """
        执行 Agent 的主流程。

        该方法负责加载历史状态、追加新消息、调用图执行器，并保存最新状态。

        Args:
            req (AgentRequest): 包含用户输入的请求对象。

        Yields:
            dict: 执行过程中的事件流。
        """
        # 从 Redis 加载当前会话的历史状态
        state = self.redis_manager.load_graph_state(req.session_id, subgraph_id=self.subgraph_id)
        if not state:
            state = {"messages": []}

        # 避免重复追加相同的用户输入
        last_msg = state["messages"][-1] if state["messages"] else None
        if not last_msg or getattr(last_msg, "content", None) != req.user_input:
            state["messages"].append(HumanMessage(content=req.user_input))

        # 以流式方式调用图执行器
        final_state = None
        for event in self.graph.stream(state):
            final_state = event
            yield event

        # 将执行后的新状态保存回 Redis
        if final_state:
            self.redis_manager.save_graph_state(final_state, req.session_id, req.subgraph_id)


if __name__ == '__main__':
    # llm = load_open_router("deepseek/deepseek-chat-v3-0324:free")
    llm = load_ollama_model("qwen3:8b")
    agent_req = AgentRequest(
        user_input="最新的AI模型是什么",
        model=llm,
        session_id="123232132",
        subgraph_id="search_agent",
    )
    search_agent = SearchAgent(agent_req)
    main_result = search_agent.run(agent_req)
    print(main_result["messages"][-1].content)
