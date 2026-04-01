import re
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from common.utils.custom_logger import get_logger
from common.utils.location_parser import extract_valid_city_candidate
from config.constants.agent_messages import WEATHER_TOOL_LOOP_EXCEEDED_MESSAGE
from config.constants.weather_agent_keywords import WEATHER_QUERY_KEYWORDS
from config.runtime_settings import AGENT_LOOP_CONFIG
from prompts.agent_prompts.tools_prompt import ToolsPrompt
from supervisor.base import BaseAgent
from supervisor.graph_state import AgentRequest
from tools.agent_tools.weather_tools import get_weathers

log = get_logger(__name__)


class WeatherAgentState(TypedDict):
    """天气子图状态"""

    messages: Annotated[List[BaseMessage], add_messages]
    tool_loop_count: int
    current_task: str


class WeatherAgent(BaseAgent):
    """天气查询 Agent。"""

    def __init__(self, req: AgentRequest):
        super().__init__(req)
        if not req.model:
            raise ValueError("天气模型初始化失败，请检查配置。")
        self.llm = req.model
        self.subgraph_id = "weather_agent"
        self.tools = self._resolve_runtime_tools([get_weathers], agent_name="weather_agent")
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        f"{ToolsPrompt.WEATHER_SYSTEM}"
                        "如果需要调用天气工具，请传入 `city_names` 数组；"
                        "拿到工具结果后，直接用中文给出简洁结论。"
                    ),
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> Runnable:
        workflow = StateGraph(WeatherAgentState)
        workflow.add_node("weather_model_node", self._model_node, retry_policy=self.RETRY_POLICY)
        workflow.add_node("tools", ToolNode(self.tools), retry_policy=self.RETRY_POLICY)
        workflow.add_edge(START, "weather_model_node")
        workflow.add_conditional_edges("weather_model_node", tools_condition)
        workflow.add_edge("tools", "weather_model_node")
        return workflow.compile(checkpointer=self.checkpointer)

    @classmethod
    def _looks_like_weather_query(cls, text: str) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        return any(keyword in normalized for keyword in WEATHER_QUERY_KEYWORDS)

    @staticmethod
    def _latest_human_text(messages: List[BaseMessage]) -> str:
        for message_item in reversed(messages):
            if isinstance(message_item, HumanMessage):
                content = getattr(message_item, "content", "")
                if isinstance(content, str):
                    return content.strip()
                return str(content or "").strip()
        return ""

    @staticmethod
    def _extract_city_from_system_context(messages: List[BaseMessage]) -> str:
        for message_item in reversed(messages[-12:]):
            if not isinstance(message_item, SystemMessage):
                continue
            content = getattr(message_item, "content", "")
            text = content if isinstance(content, str) else str(content or "")
            matched = re.search(r"当前城市:\s*([^\n，。；]+)", text)
            if matched:
                city_value = (matched.group(1) or "").strip()
                if city_value:
                    return city_value
        return ""

    @staticmethod
    def _extract_city_list(text: str) -> List[str]:
        raw = (text or "").strip()
        if not raw:
            return []

        parts = [part.strip() for part in re.split(r"[，、；;\s]+", raw) if part.strip()]
        candidates: List[str] = []
        for part in parts:
            city_val = extract_valid_city_candidate(part)
            if city_val and city_val not in candidates:
                candidates.append(city_val)

        whole_city = extract_valid_city_candidate(raw)
        if whole_city and whole_city not in candidates:
            candidates.append(whole_city)

        return candidates

    @staticmethod
    def _extract_city_from_recent_history(messages: List[BaseMessage]) -> str:
        for message_item in reversed(messages[-16:]):
            if not isinstance(message_item, HumanMessage):
                continue
            content = getattr(message_item, "content", "")
            text = content if isinstance(content, str) else str(content or "")
            city_val = extract_valid_city_candidate(text)
            if city_val:
                return city_val
        return ""

    async def _model_node(self, state: WeatherAgentState, config: RunnableConfig):
        loop_count = int(state.get("tool_loop_count", 0) or 0)
        if loop_count >= AGENT_LOOP_CONFIG.weather_max_tool_loops:
            return {
                "tool_loop_count": loop_count,
                "messages": [AIMessage(content=WEATHER_TOOL_LOOP_EXCEEDED_MESSAGE)],
            }

        source_messages = list(state.get("messages", []) or [])
        focused_messages = list(source_messages)
        current_task = str(state.get("current_task") or "").strip()
        latest_user_text = current_task or self._latest_human_text(source_messages)

        if current_task and current_task != "END_TASK":
            focused_messages.append(HumanMessage(content=f"系统指令：请专注执行当前子任务 -> {current_task}"))

        explicit_cities = self._extract_city_list(latest_user_text)
        context_city = self._extract_city_from_system_context(source_messages)
        history_city = self._extract_city_from_recent_history(source_messages)
        resolved_city = (explicit_cities[0] if explicit_cities else "") or context_city or history_city

        if resolved_city and self._looks_like_weather_query(latest_user_text):
            focused_messages.append(
                HumanMessage(
                    content=(
                        f"系统补充：当前天气任务默认城市为 {resolved_city}。"
                        "若用户未显式更换城市，请直接基于该城市调用天气工具。"
                    )
                )
            )

        llm_with_tools = self.llm.bind_tools(self.tools)
        response = await (self.prompt | llm_with_tools).ainvoke({"messages": focused_messages}, config=config)
        return {"tool_loop_count": loop_count + 1, "messages": [response]}
