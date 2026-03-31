import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, Dict, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, START, StateGraph, add_messages

from supervisor.base import BaseAgent
from supervisor.graph_state import AgentRequest
from tools.agent_tools.weather_tools import location_search
from config.constants.weather_agent_keywords import WEATHER_FOLLOWUP_CONFIRM_TOKENS, WEATHER_QUERY_KEYWORDS
from config.constants.weather_tool_constants import WEATHER_CITY_REQUIRED_MESSAGE
from common.utils.custom_logger import get_logger
from common.utils.location_parser import extract_valid_city_candidate

log = get_logger(__name__)


class WeatherAgentState(TypedDict):
    """天气子图状态"""
    messages: Annotated[List[BaseMessage], add_messages]


class WeatherAgent(BaseAgent):
    """
    天气查询Agent：处理天气相关查询。
    """

    def __init__(self, req: AgentRequest):
        super().__init__(req)
        self.subgraph_id = "weather_agent"
        # 初始化图结构
        self.graph = self._build_graph()

    # ==========================================
    # 🆕 补全的方法 1: 构建 LangGraph 的执行图
    # ==========================================
    def _build_graph(self) -> Runnable:
        """构建天气Agent的执行流图"""
        workflow = StateGraph(WeatherAgentState)
        # 添加核心推理节点
        workflow.add_node("weather_model_node", self._model_node, retry_policy=self.RETRY_POLICY)
        # 定义流转边
        workflow.add_edge(START, "weather_model_node")
        workflow.add_edge("weather_model_node", END)

        return workflow.compile(checkpointer=self.checkpointer)

    # ==========================================
    # 🆕 补全的方法 2: 判断是否是天气查询
    # ==========================================
    @classmethod
    def _looks_like_weather_query(cls, text: str) -> bool:
        """判断文本中是否包含天气查询的关键词"""
        if not text:
            return False
        text_lower = text.lower()
        # WEATHER_QUERY_KEYWORDS 从常量文件导入的列表，如 ["天气", "温度", "下雨", "预报"] 等
        for keyword in WEATHER_QUERY_KEYWORDS:
            if keyword in text_lower:
                return True
        return False

    @staticmethod
    def _latest_human_text(messages: List[BaseMessage]) -> str:
        for message_item in reversed(messages):
            if isinstance(message_item, HumanMessage):
                content = getattr(message_item, "content", "")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, str):
                    return str(content or "").strip()
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
        task_match = re.search(r"用户子任务[:：]\s*(.+)$", raw, flags=re.S)
        if task_match:
            raw = (task_match.group(1) or "").strip() or raw

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
        found_human_count = 0
        for message_item in reversed(messages[-16:]):
            if not isinstance(message_item, HumanMessage):
                continue
            found_human_count += 1
            if found_human_count == 1:
                content = getattr(message_item, "content", "")
                text = content if isinstance(content, str) else str(content or "")
                city_val = extract_valid_city_candidate(text)
                if city_val:
                    return city_val
        return ""

    @classmethod
    def _has_recent_weather_intent(cls, messages: List[BaseMessage]) -> bool:
        found_human_count = 0
        for message_item in reversed(messages[-16:]):
            if not isinstance(message_item, HumanMessage):
                continue
            found_human_count += 1
            if found_human_count == 1:
                continue

            content = getattr(message_item, "content", "")
            text = content if isinstance(content, str) else str(content or "")
            if cls._looks_like_weather_query(text):
                return True
        return False

    @staticmethod
    def _is_followup_confirmation(text: str) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        return normalized in WEATHER_FOLLOWUP_CONFIRM_TOKENS

    @staticmethod
    def _query_weather_for_cities(cities: List[str]) -> str:
        safe_cities = [city for city in cities][:5]
        if not safe_cities:
            return WEATHER_CITY_REQUIRED_MESSAGE

        results: Dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=min(4, len(safe_cities))) as executor:
            future_map = {
                executor.submit(location_search, city): city
                for city in safe_cities
            }
            for future in as_completed(future_map):
                city = future_map[future]
                try:
                    results[city] = str(future.result() or "")
                except Exception as exc:
                    results[city] = f"{city}：天气服务暂时不可用（{exc}）"

        ordered = [results.get(city, "") for city in safe_cities]
        visible = [item for item in ordered if item]
        return "\n\n".join(visible) if visible else "天气服务暂时不可用，请稍后重试。"

    def _model_node(self, state: WeatherAgentState, config: RunnableConfig) -> Dict[str, List[AIMessage]]:
        messages = state.get("messages", [])
        latest_user_text = self._latest_human_text(messages)
        context_city = self._extract_city_from_system_context(messages)
        explicit_cities = self._extract_city_list(latest_user_text)
        explicit_city = explicit_cities[0] if explicit_cities else ""
        history_city = self._extract_city_from_recent_history(messages)
        resolved_city = explicit_city or context_city or history_city

        has_recent_weather_intent = self._has_recent_weather_intent(messages)
        location_only_followup = bool(explicit_cities) and (not self._looks_like_weather_query(latest_user_text))

        should_force_query = (
                bool(resolved_city)
                and (
                        self._looks_like_weather_query(latest_user_text)
                        or (location_only_followup and has_recent_weather_intent)
                        or has_recent_weather_intent
                )
        )

        if should_force_query:
            weather_answer = self._query_weather_for_cities([resolved_city])
            return {"messages": [AIMessage(content=weather_answer)]}

        if self._looks_like_weather_query(latest_user_text):
            return {"messages": [AIMessage(content=WEATHER_CITY_REQUIRED_MESSAGE)]}

        return {"messages": [AIMessage(content="请提供城市名称，我将为您查询天气信息。")]}
