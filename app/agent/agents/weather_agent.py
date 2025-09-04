# -*- coding: utf-8 -*-
from typing import TypedDict, Annotated, Generator

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages

from agent.agent_builder import create_tool_agent_executor
from agent.graph_state import AgentRequest
from agent.llm.loader_llm_multi import load_open_router
from agent.llm.model_config import ModelConfig
from agent.tools.weather_tools import get_weathers
from utils import redis_manager

load_dotenv()

"""
天气查询子图（Weather Agent）

功能：
1. 接收用户的自然语言查询（例如，“北京今天天气怎么样？”）。
2. 利用 `create_tool_agent_executor` 构建一个具备工具调用能力的图。
3. LLM 识别出需要使用 `get_weathers` 工具，并提取出城市参数。
4. 将工具的输出结果整合后，生成一个自然的语言回答，并返回给主图。
"""


class WeatherAgentState(TypedDict):
    """
    天气查询子图状态定义
    """
    messages: Annotated[list, add_messages]


class WeatherAgent:
    """
    天气查询 Agent
    """

    def __init__(self, req: AgentRequest):
        """
        初始化天气 Agent
        """
        if not req.model:
            raise ValueError("天气 Agent 模型加载失败，请检查配置。")

        tools = [get_weathers]
        
        # 检查是否启用RAG功能
        llm_config = req.llm_config or {}
        rag_enabled = llm_config.get('rag_enabled', False)
        
        if rag_enabled:
            # 如果启用RAG，可以在这里添加相关功能
            # 例如：加载向量数据库、检索工具等
            print(f"ℹ️ 天气代理已启用RAG功能，相似度阈值: {llm_config.get('similarity_threshold', 0.7)}")
            print(f"ℹ️ 使用嵌入模型: {llm_config.get('embedding_model', 'bge-m3:latest')}")        
        self.graph = create_tool_agent_executor(
            model=req.model,
            tools=tools,
            state_class=WeatherAgentState
        )
        self.redis_manager = redis_manager.RedisManager()
        self.session_id = req.session_id
        self.subgraph_id = "weather_agent"

    def run(self, req: AgentRequest) -> Generator:
        """
        以流式方式执行 Agent。

        该方法是一个生成器，会实时产出图执行过程中的每一步事件。
        它负责处理状态加载、保存，并将图的流式输出传递出去。

        Args:
            req (AgentRequest): 包含用户输入的请求对象。

        Yields:
            dict: LangGraph 执行器产出的事件流。
        """
        state = self.redis_manager.load_graph_state(req.session_id, self.subgraph_id)
        if not state:
            state = {"messages": []}

        last_msg = state["messages"][-1] if state["messages"] else None
        if not last_msg or getattr(last_msg, "content", "") != req.user_input:
            state["messages"].append(HumanMessage(content=req.user_input))

        # 以流式方式调用图，并向上产出事件
        final_state = None
        for event in self.graph.stream(state):
            # 保存最终状态以供后续使用
            final_state = event
            yield event
        
        # 如果有最终状态，保存它
        if final_state:
            self.redis_manager.save_graph_state(final_state, req.session_id, self.subgraph_id)


# if __name__ == '__main__':
#     config = ModelConfig(
#         model="deepseek/deepseek-chat-v3-0324:free",
#         model_key="sk-xxxx",
#         model_url="https://openrouter.ai/api/v1",
#     )
#     agent_llm = load_open_router(config)
#     agent_req = AgentRequest(
#         user_input="上海和北京今天天气怎么样？",
#         model=agent_llm,
#         session_id="session_weather_12345",
#         subgraph_id="weather_agent",
#     )
#     weather_agent = WeatherAgent(req=agent_req)
#     # 流式获取结果
#     final_state = None
#     for chunk in weather_agent.run(agent_req):
#         final_state = chunk
#         print("---CHUNK START---")
#         print(final_state)
#         print("---CHUNK END---\\n")
#
#     print("\n\n===== FINAL RESPONSE =====")
#     print(final_state["messages"][-1].content)
