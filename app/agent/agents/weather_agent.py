import io
import json
from typing import TypedDict, Annotated

from IPython.core.display_functions import display
from IPython.display import Image as IPyImage  # 重命名为 IPyImage
from PIL import Image as PILImage  # 重命名为 PILImage，避免冲突
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode

from agent.graph_state import AgentRequest
from agent.llm.loader_llm_multi import load_open_router
from agent.llm.ollama_model import load_ollama_model
from agent.tools.weather_tools import get_weathers
from utils import redis_manager, concurrent_executor

"""
天气查询子图（Weather Agent）

功能：
1.接收用户的自然语言查询（例如，“北京今天天气怎么样？”）。
2.利用语言模型（LLM）的函数调用（Function Calling）或工具使用（Tool Using）能力，
    识别出需要使用 `city_search_tool` 工具，并提取出城市参数。
    识别出需要使用 `get_weather` 执行工具。
4.将工具的输出结果整合后，生成一个自然的语言回答，并返回给主管。
"""

load_dotenv()


class WeatherAgentState(TypedDict):
    """
    天气查询子图状态定义

    Attributes:
        messages (list): 消息列表，用于存储交互过程中的消息。
    """
    messages: Annotated[list, add_messages]


class WeatherAgent:
    def __init__(self, req: AgentRequest, max_threads: int = 2):
        if not req.model:
            raise ValueError("代码编写智能体模型加载失败，请检查配置。")
        self.model = req.model
        self.state_manager = redis_manager.RedisManager()
        self.session_id = req.session_id
        self.subgraph_id = req.subgraph_id
        self.concurrent_executor = concurrent_executor.ConcurrentExecutor(max_threads=max_threads)
        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         (
        #             "system",
        #             "你是一个专业的软件工程师。你的任务是根据用户的需求，编写高质量、可读性强且功能正确的代码。"
        #             "请在代码块中提供完整的代码实现。如果用户提供了反馈，请根据反馈修改你的代码。"
        #         ),
        #         MessagesPlaceholder(variable_name="messages")
        #     ]
        # )

        # self.chain = prompt | self.model
        self.graph = self._build_graph()

    def _agent_node(self, state: WeatherAgentState):
        """
        定义智能体节点（Agent Node）。
        """
        last_messages = state["messages"][-1]
        if isinstance(last_messages, ToolMessage):
            response = self.model.invoke(state["messages"])
        else:
            model_with_tools = self.model.bind_tools([get_weathers])
            response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def _router(self, state: WeatherAgentState):
        """
        定义路由函数，在 agent 节点执行后决定下一步走向。

        Args:
            state (WeatherAgentState): 当前图的状态。

        Returns:
            str: 下一个节点的名称。'tools' 表示需要执行工具，'__end__' 表示结束。
        """

        last_messages = state["messages"][-1]
        if hasattr(last_messages, "tool_calls") and last_messages.tool_calls:
            return "tools"
        else:
            return "__end__"

    def _build_graph(self):
        """
        构建天气查询子图
        """
        graph = StateGraph(WeatherAgentState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode([get_weathers]))
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", self._router)
        graph.add_edge("tools", "agent")
        return graph.compile()

    def run(self, req: AgentRequest):
        """
        执行子图一次，支持暂停/中断功能
        自动加载历史状态，避免重复追加 user_input
        """
        state = self.state_manager.load_graph_state(req.session_id, req.subgraph_id)
        if not state:
            state = {"messages": [], "interrupt": ""}
        else:
            # 清理上一次中断，保证新一轮中断独立
            state["interrupt"] = ""

        last_msg = state["messages"][-1] if state["messages"] else None
        if not last_msg or getattr(last_msg, "content", "") != req.user_input:
            state["messages"].append(HumanMessage(content=req.user_input))

        result = self.graph.invoke(state)

        # 4. 保存执行结果
        self.redis_manager.save_graph_state(result, req.session_id, req.subgraph_id)

        return result


if __name__ == '__main__':
    llm = load_open_router("deepseek/deepseek-chat-v3-0324:free")
    # llm = load_ollama_model("qwen3:8b")
    # llm = load_ollama_model("gemma3:270m")
    req = AgentRequest(
        user_input="上海,北京,今天天气怎么样？",
        model=llm,
        session_id="11111",
        subgraph_id="weather_agent",
    )
    weather_agent = WeatherAgent(req=req)
    result = weather_agent.run(req)
    print(result["messages"][-1].content)
