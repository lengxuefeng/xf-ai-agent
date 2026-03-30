import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, Dict, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph, add_messages

from agent.base import BaseAgent
from agent.graph_state import AgentRequest
from agent.tools.weather_tools import location_search
from constants.weather_agent_keywords import WEATHER_FOLLOWUP_CONFIRM_TOKENS
from constants.weather_tool_constants import WEATHER_CITY_REQUIRED_MESSAGE, WEATHER_QUERY_KEYWORDS
from utils.custom_logger import get_logger
from utils.location_parser import extract_valid_city_candidate

log = get_logger(__name__)


class WeatherAgentState(TypedDict):
    """天气子图状态"""
    messages: Annotated[List[BaseMessage], add_messages]


class WeatherAgent(BaseAgent):
    """天气查询Agent：处理天气相关查询"""

    def __init__(self, req: AgentRequest):
        super().__init__(req)
        self.subgraph_id = "weather_agent"
        self.graph = self._build_graph()

    @staticmethod
    def _latest_human_text(messages: List[BaseMessage]) -> str:
        """提取最近一条用户输入文本。"""
        for message_item in reversed(messages):
            if isinstance(message_item, HumanMessage):
                content = getattr(message_item, "content", "")
                if isinstance(content, str):
                    return content.strip()
                return str(content or "").strip()
        return ""

    @staticmethod
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

    @staticmethod
    def _looks_like_weather_query(text: str) -> bool:
        """判断输入是否为天气相关问题。"""
        lower_text = (text or "").strip().lower()
        if not lower_text:
            return False
        return any(keyword in lower_text for keyword in WEATHER_QUERY_KEYWORDS)

    @staticmethod
    def _extract_city_list(text: str) -> List[str]:
        """
        从用户输入中抽取城市列表（支持逗号分隔）。
        例如：“南京，北京，上海” -> ["南京", "北京", "上海"]。
        """
        raw = (text or "").strip()
        if not raw:
            return []
        task_match = re.search(r"用户子任务[:：]\s*(.+)$", raw, flags=re.S)
        if task_match:
            raw = (task_match.group(1) or "").strip() or raw
        parts = [part.strip() for part in re.split(r"[，,、；;\s]+", raw) if part.strip()]
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
        """
        从最近多轮用户消息中回捞城市名。

        典型场景：
        - 用户先说“南京”，下一轮只回复“是的/继续”，仍应继续查天气而不是重复追问城市。
        """
        found_human_count = 0
        for message_item in reversed(messages[-16:]):
            if not isinstance(message_item, HumanMessage):
                continue
            found_human_count += 1
            if found_human_count == 1:
                continue
            content = getattr(message_item, "content", "")
            text = content if isinstance(content, str) else str(content or "")
            city_val = extract_valid_city_candidate(text)
            if city_val:
                return city_val
        return ""

    @classmethod
    def _has_recent_weather_intent(cls, messages: List[BaseMessage]) -> bool:
        """判断历史中是否存在明确天气诉求。"""
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
        """
        并发查询多城市天气，避免多城市输入被串行请求拖慢。
        """
        safe_cities = [city for city in cities if city][:5]
        if not safe_cities:
            return WEATHER_CITY_REQUIRED_MESSAGE
        if len(safe_cities) == 1:
            return location_search(safe_cities[0])

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

    def _model_node(self, state: WeatherAgentState) -> Dict[str, List[AIMessage]]:
        messages = state.get("messages", [])
        latest_user_text = self._latest_human_text(messages)
        context_city = self._extract_city_from_system_context(messages)
        explicit_cities = self._extract_city_list(latest_user_text)
        explicit_city = explicit_cities[0] if explicit_cities else ""
        history_city = self._extract_city_from_recent_history(messages)
        resolved_city = explicit_city or context_city or history_city

        has_recent_weather_intent = self._has_recent_weather_intent(messages)
        location_only_followup = bool(explicit_cities) and (not self._looks_like_weather_query(latest_user_text))
        followup_confirmation = self._is_followup_confirmation(latest_user_text)

        if explicit_cities:
            weather_answer = self._query_weather_for_cities(explicit_cities)
            return {"messages": [AIMessage(content=weather_answer)]}

        should_force_query = (
            bool(resolved_city)
            and (
                self._looks_like_weather_query(latest_user_text)
                or (location_only_followup and has_recent_weather_intent)
                or (followup_confirmation and has_recent_weather_intent)
                or has_recent_weather_intent
            )
        )
        if should_force_query:
            query_cities = [resolved_city] if resolved_city else []
            weather_answer = self._query_weather_for_cities(query_cities)
            return {"messages": [AIMessage(content=weather_answer)]}

        return {"messages": [AIMessage(content=WEATHER_CITY_REQUIRED_MESSAGE)]}

    def _build_graph(self) -> Runnable:
        """
        天气子图（确定性执行版）。

        根因修复：
        - 不再走 LLM「先口头承诺、再尝试工具」路径，避免出现“我来帮你查…”后长时间无最终正文；
        - 路由到 weather_agent 后，有城市就直接查天气，无城市才追问城市。
        """
        workflow = StateGraph(WeatherAgentState)
        workflow.add_node("agent", self._model_node)
        workflow.add_edge(START, "agent")
        workflow.add_edge("agent", END)
        return workflow.compile(checkpointer=self.checkpointer)
