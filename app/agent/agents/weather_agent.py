import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, Dict, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
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
    """
    天气查询Agent：处理天气相关查询。

    核心职责：
    1. 查询指定城市的天气情况和气象预报
    2. 支持多城市并发查询
    3. 提供天气结论和出行建议
    4. 避免扩展到旅游路线、景点攻略等与天气无关的内容

    设计要点：
    1. 优先使用系统上下文中的城市信息（来自会话状态）
    2. 如果没有城市，从用户输入中提取
    3. 如果还是没有城市，但历史中有天气查询，使用历史城市
    4. 支持多城市并发查询，但限制为最多5个城市
    5. 查询失败时要有友好的错误提示

    使用场景：
    - 用户问"郑州今天天气如何？"
    - 用户问"郑州和北京今天天气怎么样？"
    - 用户说"我在郑州，查一下天气"
    - 用户继续对话："北京呢？"

    重要说明：
    - 这个Agent只输出天气结论和出行建议
    - 不要输出旅游路线、景点攻略或与天气无关的内容
    - 避免在用户问"郑州的房价怎么样？"这种问题中误路由
    - 需要与search_agent配合，避免重复信息
    """

    def __init__(self, req: AgentRequest):
        super().__init__(req)
        self.subgraph_id = "weather_agent"
        self.graph = self._build_graph()

    @staticmethod
    def _latest_human_text(messages: List[BaseMessage]) -> str:
        """
        提取最近一条用户输入文本。

        设计要点：
        1. 从历史消息中倒序查找
        2. 只关注HumanMessage，忽略其他类型的消息
        3. 提取content字段，如果content不是字符串则转为字符串

        Args:
            messages: 历史消息列表

        Returns:
            str: 最近一条用户输入文本，空字符串则返回空

        使用场景：
        - 从会话历史中提取最近的用户输入
        - 判断当前话题是否是天气
        - 结合城市提取逻辑使用
        """
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
        """
        从系统上下文中提取已知城市（来自会话槽位摘要）。

        设计要点：
        1. 从最近的12条消息中查找
        2. 查找包含"当前城市："格式的SystemMessage
        3. 提取冒号后面的城市名
        4. 如果有多个，取最后一个（最新的）

        设计理由：
        1. 会话状态管理器会将城市槽位写入上下文摘要
        2. 通过这个方法，Agent可以复用用户之前输入的城市信息
        3. 避免重复询问城市

        Args:
            messages: 历史消息列表

        Returns:
            str: 提取的城市名，没找到则返回空字符串

        使用场景：
        - 用户说"天气怎么样？"，从上下文中获取城市
        - 用户说"查一下天气"，从上下文中获取城市
        - 避免每次都问城市
        """
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
        """
        从用户输入中抽取城市列表（支持逗号分隔）。

        设计要点：
        1. 支持逗号、顿号、分号等常见分隔符
        2. 提取"用户子任务：[\s*(.+)$"前缀之后的内容
        3. 对每个片段调用城市提取函数
        4. 去重并保持原始顺序

        设计理由：
        1. 支持复合查询："郑州，北京，上海"
        2. 从完整输入中提取上下文
        3. 保留原始顺序，按优先级排序

        示例：
        - "郑州，北京，上海" -> ["郑州", "北京", "上海"]
        - "用户子任务：查郑州和北京的天气" -> ["郑州", "北京"]

        Args:
            text: 用户输入的完整文本

        Returns:
            List[str]: 提取的城市列表，空列表则返回空列表

        使用场景：
        - 多城市天气查询
        - 复合问题中的多城市提取
        - 城于任务构建函数
        """
        raw = (text or "").strip()
        if not raw:
            return []

        # 提取"用户子任务："前缀之后的完整文本
        task_match = re.search(r"用户子任务[:]]\s*(.+)$", raw, flags=re.S)
        if task_match:
            raw = (task_match.group(1) or "").strip() or raw

        # 按分隔符分割
        parts = [part.strip() for part in re.split(r"[，、；;\s]+", raw) if part.strip()]
        candidates: List[str] = []

        for part in parts:
            # 调用城市提取函数
            city_val = extract_valid_city_candidate(part)
            if city_val and city_val not in candidates:
                candidates.append(city_val)

        # 提取完整文本中的城市（兜底）
        whole_city = extract_valid_city_candidate(raw)
        if whole_city and whole_city not in candidates:
            candidates.append(whole_city)

        return candidates

    @staticmethod
    def _extract_city_from_recent_history(messages: List[BaseMessage]) -> str:
        """
        从最近多轮用户消息中回捞城市名。

        设计要点：
        1. 从最近16条消息中查找
        2. 只查找HumanMessage，忽略其他类型的消息
        3. 从最近的用户消息开始查找
        4. 找到第一个匹配的城市就返回

        设计理由：
        1. 支持场景：用户先说"南京"，下一轮只回复"是的/继续"，仍应查南京的天气
        2. 优先使用最近的用户消息
        3. 避免使用AI消息中的城市，因为可能是AI生成的
        4. 只查找最后一条用户消息，而不是所有用户消息

        Args:
            messages: 历史消息列表

        Returns:
            str: 提取的城市名，没找到则返回空字符串

        使用场景：
        - 上下文城市提取
        - 兜底城市来源
        - 支持多轮对话中的城市复用
        """
        found_human_count = 0
        for message_item in reversed(messages[-16:]):
            if not isinstance(message_item, HumanMessage):
                continue
            found_human_count += 1
            # 只检查最近一条用户消息
            if found_human_count == 1:
                content = getattr(message_item, "content", "")
                text = content if isinstance(content, str) else str(content or "")
                city_val = extract_valid_city_candidate(text)
                if city_val:
                    return city_val
        return ""

    @classmethod
    def _has_recent_weather_intent(cls, messages: List[BaseMessage]) -> bool:
        """
        判断历史中是否存在明确天气诉求。

        设计要点：
        1. 检查最近的用户消息是否有天气相关内容
        2. 使用天气关键词列表进行匹配
        3. 限制检查范围到最近16条消息
        4. 只检查HumanMessage，忽略其他类型的消息

        设计理由：
        1. 识别历史话题，避免主题漂移
        2. 确定当前对话是否在天气话题
        3. 只关心用户输入，忽略AI生成的消息

        Args:
            messages: 历史消息列表

        Returns:
            bool: True表示最近有天气诉求，False表示没有

        使用场景：
        - 判断是否应该使用历史城市
        - 确定是否需要追问城市
        - 识别历史话题
        """
        found_human_count = 0
        for message_item in reversed(messages[-16:]):
            if not isinstance(message_item, HumanMessage):
                continue
            found_human_count += 1
            # 只检查最近一条用户消息
            if found_human_count == 1:
                continue

            content = getattr(message_item, "content", "")
            text = content if isinstance(content, str) else str(content or "")
            if cls._looks_like_weather_query(text):
                return True
        return False

    @staticmethod
    def _is_followup_confirmation(text: str) -> bool:
        """
        判断是否是"天气怎么样？好的，继续"这类确认回复。

        设计要点：
        1. 检查预定义的确认提示词列表
        2. 匹配这些提示词，表示用户确认继续之前的天气查询
        3. 用于避免重复查询，提高响应效率

        设计理由：
        1. 用户可能说"天气怎么样？好的，继续"表示确认
        2. 这时应该直接查询天气，而不是重新分析
        3. 提高对话的连贯性

        Args:
            text: 用户输入文本

        Returns:
            bool: True表示是确认回复，False表示不是

        使用场景：
        - 快速判断是否需要重新分析
        - 优化对话连贯性
        """
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        return normalized in WEATHER_FOLLOWUP_CONFIRM_TOKENS

    @staticmethod
    def _query_weather_for_cities(cities: List[str]) -> str:
        """
        并发查询多城市天气，避免多城市输入被串行请求拖慢。

        设计要点：
        1. 限制最多查询5个城市，避免请求过多
        2. 使用线程池并发查询
        3. 查询失败时返回友好的错误提示
        4. 保留查询结果的顺序，与输入顺序一致

        设计理由：
        1. 并发查询提高响应速度
        2. 限制城市数量，避免请求过多被限流
        3. 友好错误提示，提升用户体验
        4. 保留顺序，便于用户对比不同城市的天气

        Args:
            cities: 城市列表，如["郑州", "北京", "上海"]

        Returns:
            str: 天气信息或错误提示

        使用场景：
        - 多城市查询："郑州、北京、上海的天气怎么样？"
        - 多城市确认："北京和天津天气怎么样？"
        - 快速多城市天气对比
        """
        # 限制最多5个城市，避免请求过多
        safe_cities = [city for city in cities][:5]
        if not safe_cities:
            return WEATHER_CITY_REQUIRED_MESSAGE

        results: Dict[str, str] = {}

        # 使用线程池并发查询
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

        # 按输入顺序拼接结果
        ordered = [results.get(city, "") for city in safe_cities]
        visible = [item for item in ordered if item]
        return "\n\n".join(visible) if visible else "天气服务暂时不可用，请稍后重试。"

    def _model_node(self, state: WeatherAgentState) -> Dict[str, List[AIMessage]]:
        """
        模型节点：调用LLM生成天气回复。

        设计要点：
        1. 检查是否有明确的天气确认
        2. 从系统上下文中提取城市
        3. 从用户输入中提取城市列表
        4. 检查最近历史是否有天气诉求
        5. 判断是否应该查询天气

        设计理由：
        1. 明确确认查询，避免重复查询
        2. 多层城市提取，提高准确性
        3. 历史诉求检查，识别话题连续性
        4. 只在必要时才查询天气，节省API调用

        查询策略：
        1. 如果有城市 -> 查询天气
        2. 如果无城市但有天气诉求 -> 使用默认城市（如果配置）或提醒用户输入
        3. 如果无城市无天气诉求 -> 提示用户输入城市

        Args:
            state: 天气Agent当前状态

        Returns:
            Dict[str, List[AIMessage]]: 包含AI回复的字典

        使用场景：
        - 生成天气查询结果
        - 生成天气查询问题
        - 生成城市输入提示
        """
        messages = state.get("messages", [])
        latest_user_text = self._latest_human_text(messages)
        context_city = self._extract_city_from_system_context(messages)
        explicit_cities = self._extract_city_list(latest_user_text)
        explicit_city = explicit_cities[0] if explicit_cities else ""
        history_city = self._extract_city_from_recent_history(messages)
        resolved_city = explicit_city or context_city or history_city

        has_recent_weather_intent = self._has_recent_weather_intent(messages)
        location_only_followup = bool(explicit_cities) and (not self._looks_like_weather_query(latest_user_text))

        # 优先检查是否需要查询天气
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

        # 如果没有城市但有天气诉求，提示用户输入城市
        if self._looks_like_weather_query(latest_user_text):
            return {"messages": [AIMessage(content=WEATHER_CITY_REQUIRED_MESSAGE)]}

        # 其他情况，默认回复
        return {"messages": [AIMessage(content="请提供城市名称，我将为您查询天气信息。")]}
