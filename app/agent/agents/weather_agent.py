import re
from typing import Annotated, TypedDict, List
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agent.base import BaseAgent
from agent.graph_state import AgentRequest
from config.runtime_settings import AGENT_LOOP_CONFIG
from constants.agent_messages import WEATHER_TOOL_LOOP_EXCEEDED_MESSAGE
from constants.weather_tool_constants import WEATHER_CITY_REQUIRED_MESSAGE, WEATHER_QUERY_KEYWORDS
from agent.tools.weather_tools import get_weathers
from utils.custom_logger import get_logger
from utils.location_parser import extract_valid_city_candidate
from agent.prompts.tools_prompt import ToolsPrompt

log = get_logger(__name__)

class WeatherAgentState(TypedDict):
    """天气子图状态"""
    messages: Annotated[List[BaseMessage], add_messages]
    tool_loop_count: int  # 工具调用次数计数器


class WeatherAgent(BaseAgent):
    """天气查询Agent：处理天气相关查询"""

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
                ("system", ToolsPrompt.WEATHER_SYSTEM),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> Runnable:
        """构建天气子图，限制工具循环次数防止死循环"""
        workflow = StateGraph(WeatherAgentState)

        def _latest_human_text(messages: List[BaseMessage]) -> str:
            """提取最近一条用户输入文本。"""
            for message_item in reversed(messages):
                if isinstance(message_item, HumanMessage):
                    content = getattr(message_item, "content", "")
                    if isinstance(content, str):
                        return content.strip()
                    return str(content or "").strip()
            return ""

        def _extract_city_from_system_context(messages: List[BaseMessage]) -> str:
            """从系统上下文中提取已知城市（来自会话槽位摘要）。"""
            for message_item in reversed(messages[-12:]):
                if not isinstance(message_item, SystemMessage):
                    continue
                content = getattr(message_item, "content", "")
                text = content if isinstance(content, str) else str(content or "")
                matched = re.search(r"当前城市:\s*([^\n，。；;]+)", text)
                if matched:
                    city_value = (matched.group(1) or "").strip()
                    if city_value:
                        return city_value
            return ""

        def _looks_like_weather_query(text: str) -> bool:
            """判断输入是否为天气相关问题。"""
            lower_text = (text or "").strip().lower()
            if not lower_text:
                return False
            return any(keyword in lower_text for keyword in WEATHER_QUERY_KEYWORDS)

        def model_node(state: WeatherAgentState):
            loop_count = int(state.get("tool_loop_count", 0) or 0)
            if loop_count >= AGENT_LOOP_CONFIG.weather_max_tool_loops:
                return {
                    "tool_loop_count": loop_count,
                    "messages": [AIMessage(content=WEATHER_TOOL_LOOP_EXCEEDED_MESSAGE)],
                }

            # 生产门控：城市缺失时先追问，不调用工具，防止“玩吗”被误识别成城市。
            messages = state.get("messages", [])
            latest_user_text = _latest_human_text(messages)
            context_city = _extract_city_from_system_context(messages)
            explicit_city = extract_valid_city_candidate(latest_user_text)
            if (
                loop_count == 0
                and _looks_like_weather_query(latest_user_text)
                and not context_city
                and not explicit_city
            ):
                return {
                    "tool_loop_count": loop_count + 1,
                    "messages": [AIMessage(content=WEATHER_CITY_REQUIRED_MESSAGE)],
                }

            chain = self.prompt | self.model_with_tools
            return {"tool_loop_count": loop_count + 1, "messages": [chain.invoke(state)]}

        def route_after_agent(state: WeatherAgentState):
            """判断是否继续调用工具"""
            if int(state.get("tool_loop_count", 0) or 0) > AGENT_LOOP_CONFIG.weather_max_tool_loops:
                return END
            return tools_condition(state)

        tool_node = ToolNode(self.tools)

        workflow.add_node("agent", model_node)
        workflow.add_node("tools", tool_node)

        workflow.add_edge(START, "agent")

        workflow.add_conditional_edges("agent", route_after_agent)

        workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=self.checkpointer)
