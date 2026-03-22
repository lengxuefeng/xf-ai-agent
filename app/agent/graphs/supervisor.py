import functools
import re
import json
import time
import queue
import threading
from typing import Optional, List, Dict, Any, TypedDict, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START
from langgraph.constants import Send
from pydantic import BaseModel, Field

from agent.graph_state import AgentRequest
from agent.graphs.checkpointer import get_checkpointer
from agent.graphs.state import GraphState, SubTask, WorkerResult
from agent.llm.unified_loader import create_model_from_config
from agent.registry import agent_classes, MEMBERS
from agent.prompts.supervisor_prompt import IntentRouterPrompt, ChatFallbackPrompt, PlannerPrompt, AggregatorPrompt
from constants.agent_registry_keywords import AGENT_KEYWORDS, AgentKeywordGroup
from constants.approval_constants import DEFAULT_ALLOWED_DECISIONS, DEFAULT_INTERRUPT_MESSAGE
from constants.search_keywords import SEARCH_REAL_ESTATE_KEYWORDS, SEARCH_WEATHER_KEYWORDS
from constants.supervisor_keywords import SUPERVISOR_KEYWORDS, SupervisorKeywordGroup
from constants.workflow_constants import (
    AGENT_DOMAIN_MAP,
    MULTI_DOMAIN_AGENT_PRIORITY,
    PENDING_TASK_STATUSES,
    RouteStrategy,
    TaskStatus,
    WORKER_CANCELLED_RESULT,
    WORKER_PENDING_APPROVAL_RESULT,
    WORKFLOW_INTERRUPT_RESULT_TYPE,
)
from config.runtime_settings import (
    AGENT_LOOP_CONFIG,
    AGENT_LIVE_STREAM_ENABLED,
    AGGREGATOR_CONFIG,
    CHAT_NODE_FIRST_TOKEN_TIMEOUT_SEC,
    CHAT_NODE_STREAM_ENABLED,
    CHAT_NODE_TOTAL_TIMEOUT_SEC,
    MODEL_TIERING_CONFIG,
    ROUTER_POLICY_CONFIG,
)
from services.route_metrics_service import route_metrics_service
from services.agent_stream_bus import agent_stream_bus
from services.request_cancellation_service import request_cancellation_service
from utils.custom_logger import get_logger
from utils.date_utils import get_agent_date_context

log = get_logger(__name__)
INTERRUPT_RESULT_TYPE = WORKFLOW_INTERRUPT_RESULT_TYPE

# --- Models ---
class IntentDecision(BaseModel):
    """意图路由决策模型"""
    intent: str = "CHAT"  # 目标Agent名称
    confidence: float = 0.5  # 置信度
    is_complex: bool = False  # 是否复杂任务
    direct_answer: str = ""  # 简单问题的直接答案


class DomainDecision(BaseModel):
    """数据域路由决策模型"""
    data_domain: str = "GENERAL"  # 数据域名称
    confidence: float = 0.5  # 置信度
    source: str = "llm"  # 决策来源


class PlannerTaskDecision(BaseModel):
    """规划器输出的单个任务模型。"""
    id: str = "t1"  # 子任务 ID
    agent: str = "CHAT"  # 目标执行 Agent
    input: str = ""  # 子任务输入
    depends_on: List[str] = Field(default_factory=list)  # 依赖任务 ID 列表


class PlannerDecision(BaseModel):
    """规划器结构化输出模型。"""
    tasks: List[PlannerTaskDecision] = Field(default_factory=list)  # 子任务列表


class RequestAnalysisDecision(BaseModel):
    """统一请求分析结果，供 Domain/Intent/Planner 共享。"""
    candidate_agents: List[str] = Field(default_factory=list)  # 命中的 Agent 候选（按优先级）
    candidate_domains: List[str] = Field(default_factory=list)  # 命中的数据域候选（按优先级）
    is_multi_intent: bool = False  # 是否命中多意图
    is_multi_domain: bool = False  # 是否命中多数据域
    has_dependency_hint: bool = False  # 是否存在先后依赖提示
    route_strategy: str = RouteStrategy.SINGLE_DOMAIN.value  # 路由策略
    reason: str = "single_domain"  # 策略命中原因


class WorkerState(TypedDict):
    """Worker节点状态"""
    task: SubTask  # 待执行任务
    context_slots: Dict[str, Any]  # 会话结构化槽位
    context_summary: str  # 会话摘要
    messages: List[BaseMessage]  # 最近对话窗口


class _ChatNodeStreamFailure(Exception):
    """chat_node 首包/流式阶段失败。"""

    def __init__(self, detail: str, *, partial_output_emitted: bool = False) -> None:
        super().__init__(detail)
        self.partial_output_emitted = partial_output_emitted


def _classify_agent_failure(exc: Exception) -> tuple[str, str]:
    """统一归类 Agent 执行失败，避免将内部异常直接暴露为正文。"""
    detail = str(exc or "").strip()
    lower = detail.lower()
    timeout_markers = (
        "timeout",
        "timed out",
        "readtimeout",
        "first_token_timeout",
        "total_timeout",
        "超时",
    )
    connection_markers = (
        "connection",
        "connecterror",
        "connectionerror",
        "apiconnectionerror",
        "连接失败",
        "连接断开",
    )
    if isinstance(exc, TimeoutError) or any(marker in lower for marker in timeout_markers):
        return "处理超时，请稍后重试。", detail
    if isinstance(exc, ConnectionError) or any(marker in lower for marker in connection_markers):
        return "模型服务连接异常，请稍后重试。", detail
    return "底层服务执行异常，请稍后重试。", detail


def _looks_like_sql_request(text: str) -> bool:
    """快速识别可直接路由到 SQL Agent 的请求模式。"""
    t = (text or "").strip().lower()
    if not t:
        return False

    # Holter 语境下优先走 yunyou_agent，只有出现“显式本地 SQL 锚点”才判定为 sql_agent。
    if _looks_like_holter_request(t):
        explicit_sql_anchors = (
            r"\bselect\b",
            r"\border\s+by\b",
            r"\blimit\b",
            r"\bwhere\b",
            r"\bgroup\s+by\b",
            r"\bjoin\b",
            r"\bsql\b",
            r"数据库",
            r"数据表",
            r"本地库",
            r"库里",
            r"\bt_[a-z0-9_]+\b",
        )
        return any(re.search(anchor, t) for anchor in explicit_sql_anchors)

    patterns = SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.SQL_REGEX_PATTERNS]
    return any(re.search(p, t) for p in patterns)


def _looks_like_holter_request(text: str) -> bool:
    """识别 Holter/云柚业务域请求，用于优先路由到 yunyou_agent。"""
    t = (text or "").strip().lower()
    if not t:
        return False

    keywords = SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.HOLTER_DOMAIN]
    return any(k in t for k in keywords)


def _looks_like_weather_request(text: str) -> bool:
    """识别天气类问题。"""
    t = (text or "").strip().lower()
    if not t:
        return False
    keywords = SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.WEATHER_DOMAIN]
    return any(k in t for k in keywords)


def _looks_like_search_request(text: str) -> bool:
    """识别互联网检索类问题。"""
    t = (text or "").strip().lower()
    if not t:
        return False
    keywords = SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.SEARCH_DOMAIN]
    if any(k in t for k in keywords):
        return True
    # 对“查一下/搜一下”这类泛词做降噪：若同时命中其他明确业务域，不视为 search。
    generic_query_hints = ("查一下", "搜一下", "查一查", "搜一搜")
    if any(h in t for h in generic_query_hints):
        if _looks_like_holter_request(t) or _looks_like_sql_request(t) or _looks_like_weather_request(t):
            return False
        return True
    return False


def _is_weather_actionable_clause(text: str) -> bool:
    """
    判断“天气关键词”是否属于明确执行请求，而不是背景情绪描述。

    典型背景描述：
    - “天气也这么不好，老板让我查 holter”
    这类句子不应拆出 weather_agent 子任务。
    """
    t = (text or "").strip().lower()
    if not t or (not _looks_like_weather_request(t)):
        return False

    if _looks_like_weather_reuse_query(t) or _wants_weather_refresh(t):
        return True
    if any(mark in t for mark in ("?", "？")):
        return True
    if t.endswith(("吗", "么", "呢")):
        return True

    action_patterns = (
        r"(查|查询|看看|看下|看一下|告诉我|帮我|搜|获取).{0,8}(天气|气温|温度|下雨|降雨|风力|湿度|空气质量|体感|能见度)",
        r"(天气|气温|温度|下雨|降雨|风力|湿度|空气质量|体感|能见度).{0,8}(如何|怎么样|多少|几度|吗|呢|建议|适合)",
        r"(适合出门|适合外出|可以出门|会不会下雨|冷不冷|热不热|空气怎么样)",
    )
    return any(re.search(pattern, t) for pattern in action_patterns)


def _is_search_actionable_clause(text: str) -> bool:
    """
    判断“搜索关键词”是否属于明确检索请求。
    避免把“附近/推荐”这类非请求性描述误拆成 search_agent。
    """
    t = (text or "").strip().lower()
    if not t or (not _looks_like_search_request(t)):
        return False
    if any(mark in t for mark in ("?", "？")):
        return True
    query_hints = ("查", "查询", "搜", "搜索", "帮我", "给我", "推荐", "看看", "列出", "有哪些", "哪里", "去哪")
    return any(hint in t for hint in query_hints)


def _looks_like_medical_request(text: str) -> bool:
    """识别医疗健康类问题。"""
    t = (text or "").strip().lower()
    if not t:
        return False
    keywords = tuple(k.lower() for k in AGENT_KEYWORDS[AgentKeywordGroup.MEDICAL])
    return any(k in t for k in keywords)


def _looks_like_code_request(text: str) -> bool:
    """识别代码开发类问题。"""
    t = (text or "").strip().lower()
    if not t:
        return False
    keywords = tuple(k.lower() for k in AGENT_KEYWORDS[AgentKeywordGroup.CODE])
    if not any(k in t for k in keywords):
        return False

    learning_hints = (
        "能学吗",
        "值得学吗",
        "好学吗",
        "难学吗",
        "怎么学",
        "学习路线",
        "学习路径",
        "入门",
        "前景",
        "就业",
        "转行",
        "零基础",
    )
    code_action_hints = (
        "写",
        "实现",
        "生成",
        "代码",
        "函数",
        "脚本",
        "程序",
        "接口",
        "类",
        "编译",
        "运行",
        "执行",
        "报错",
        "bug",
        "异常",
        "修复",
        "调试",
        "优化",
        "重构",
        "算法",
    )

    if any(hint in t for hint in code_action_hints):
        return True

    if any(hint in t for hint in learning_hints):
        return False

    if re.search(r"(class\s+\w+|def\s+\w+|function\s+\w+|if\s*\(|for\s*\(|while\s*\(|print\s*\()", t):
        return True

    # 仅提到语言名但没有执行/排障动作时，视为泛问答而不是代码代理。
    language_only_markers = ("python", "java", "javascript", "typescript", "go", "rust", "c++", "c#", "php", "ruby")
    if any(marker in t for marker in language_only_markers):
        return False

    return any(k in t for k in keywords if k not in {"python", "java", "编程"})


def _looks_like_general_chat_request(text: str) -> bool:
    """识别更接近闲聊/泛问答的输入，避免无必要触发慢路由。"""
    t = (text or "").strip().lower()
    if not t:
        return True

    # 命中任一专业域关键词，则不视为纯闲聊。
    specialized_hits = (
        _looks_like_holter_request(t)
        or _looks_like_sql_request(t)
        or _looks_like_weather_request(t)
        or _looks_like_search_request(t)
        or _looks_like_medical_request(t)
        or _looks_like_code_request(t)
    )
    return not specialized_hits

# --- Helpers ---
def _latest_human_message(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        text = item.get("text") or item.get("content") or ""
                        if isinstance(text, str):
                            parts.append(text)
                return " ".join(parts).strip()
            return str(content or "").strip()
    return ""


def _content_to_text(content: Any) -> str:
    """将 LLM content（str / block list / other）统一转为可展示文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str) and text:
                    parts.append(text)
        if parts:
            return "\n".join(parts)
        return str(content)
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        if isinstance(text, str):
            return text
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _history_requests_location(messages: Optional[List[BaseMessage]]) -> bool:
    """判断最近一轮 AI 是否明确向用户追问过城市。"""
    if not messages:
        return False
    city_request_hints = (
        "哪个城市",
        "所在城市",
        "请告诉我城市",
        "请提供城市",
        "城市名",
        "告诉我你在哪个城市",
    )
    for msg in reversed(messages[-6:]):
        if not isinstance(msg, AIMessage):
            continue
        text = _content_to_text(getattr(msg, "content", "")).lower()
        if any(hint in text for hint in city_request_hints):
            return True
    return False


def _is_followup_supplement(text: str, messages: Optional[List[BaseMessage]] = None) -> bool:
    """识别用户是否在补充上轮追问信息（日期、确认词、简短参数）。"""
    t = (text or "").strip().lower()
    if not t:
        return False

    if t in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.FOLLOWUP_CONFIRM]:
        return True

    if any(k in t for k in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.FOLLOWUP_REFERENCE]):
        return True

    # 仅包含日期/分隔符/空白的短文本，通常是对“请补充时间范围”的回答
    if len(t) <= 60 and re.fullmatch(r"[0-9\-\s:~至到/,，.]+", t):
        return True

    if len(t) <= 80 and re.search(r"\b20\d{2}-\d{2}-\d{2}\b", t):
        return True

    # 排序/条数/字段补充，常见于“你先查最近数据，再补一句按 id 倒序前 5 条”
    if len(t) <= 120 and any(k in t for k in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.FOLLOWUP_ORDER_HINT]):
        return True

    # 仅当“上一轮明确追问过城市”时，才把短词当作地点补充，避免把“开心/难受”等误判为城市。
    if _history_requests_location(messages) and _looks_like_location_fragment(t):
        return True

    return False


def _looks_like_location_fragment(text: str) -> bool:
    """识别“仅城市名”这类补充输入。"""
    t = (text or "").strip()
    if not t or len(t) > 40:
        return False
    # 支持多城市补充：如“南京，北京，上海”
    if any(sep in t for sep in ("，", ",", "、", "；", ";", "\n")):
        parts = [part.strip() for part in re.split(r"[，,、；;\n]+", t) if part.strip()]
        if not (1 <= len(parts) <= 5):
            return False
        # 递归复用“单城市片段”判定逻辑
        return all(_looks_like_location_fragment(part) for part in parts)
    if any(sep in t for sep in ("。", "！", "!", "？", "?")):
        return False
    if any(k in t for k in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.NON_LOCATION_EMOTION_WORDS]):
        return False
    patterns = SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.FOLLOWUP_LOCATION_REGEX]
    if not any(re.fullmatch(p, t) for p in patterns):
        return False

    # 无行政后缀时，只接受短城市名（如“郑州”“南京”），过滤普通短语误判。
    has_admin_suffix = t.endswith(("市", "县", "区", "州", "盟", "旗"))
    if has_admin_suffix:
        return True
    return 1 < len(t) <= 4


def _extract_recent_city_from_history(messages: List[BaseMessage]) -> Optional[str]:
    """从最近人类消息中提取城市名，用于后续问题的上下文补全。"""
    if not messages:
        return None

    ignore_keywords = ("附近", "哪里", "什么", "怎么", "为什么", "好玩", "天气", "活动", "房价", "小区", "推荐")
    suffix_particles_pattern = re.compile(r"[的了呀啊呢吧吗嘛～~\s,，。！!？?]+")

    for msg in reversed(messages[-12:]):
        if not isinstance(msg, HumanMessage):
            continue
        text = _content_to_text(getattr(msg, "content", "")).strip()
        if not text:
            continue

        # 先尝试抽取“在郑州/去上海/位于杭州”这类明确表达
        m = re.search(r"(?:在|到|去|位于)([\u4e00-\u9fa5]{2,8}(?:市|县|区|州|盟|旗)?)", text)
        if m:
            city = m.group(1).strip()
            if city and not any(k in city for k in ignore_keywords):
                return city

        # 再尝试“郑州的呀”这种口语补充
        normalized = suffix_particles_pattern.sub("", text)
        if 1 < len(normalized) <= 6 and _looks_like_location_fragment(normalized):
            if not any(k in normalized for k in ignore_keywords):
                return normalized

    # 兜底：从最近 AI 天气回复中抽取城市（如“郑州的实时天气详情”）
    for msg in reversed(messages[-12:]):
        if not isinstance(msg, AIMessage):
            continue
        text = _content_to_text(getattr(msg, "content", "")).strip()
        if not text:
            continue
        m = re.search(r"([\u4e00-\u9fa5]{2,8})(?:市)?的(?:实时)?天气", text)
        if m:
            city = m.group(1).strip()
            if city and not any(k in city for k in ignore_keywords):
                return city

    return None


def _extract_city_from_context_slots(context_slots: Optional[Dict[str, Any]]) -> Optional[str]:
    """从结构化槽位里读取城市。"""
    if not isinstance(context_slots, dict):
        return None
    city_value = context_slots.get("city")
    if isinstance(city_value, str):
        normalized_city = city_value.strip()
        if normalized_city:
            return normalized_city
    return None


def _input_has_location_anchor(text: str) -> bool:
    """判断输入中是否已显式包含位置锚点，避免重复注入上下文城市。"""
    t = (text or "").strip()
    if not t:
        return False
    if _looks_like_location_fragment(t):
        return True
    if re.search(r"(?:在|到|去|位于)\s*[\u4e00-\u9fa5]{2,8}(?:市|县|区|州|盟|旗)?", t):
        return True
    if re.search(r"[\u4e00-\u9fa5]{2,8}(?:市|县|区|州|盟|旗)", t):
        return True
    return False


def _history_hint_intent(messages: List[BaseMessage], latest_user_text: str = "") -> Optional[str]:
    """从最近上下文中推断业务域，避免补充信息被错误路由到 CHAT。"""
    recent = " ".join(
        _content_to_text(getattr(msg, "content", "")) for msg in messages[-8:]
        if isinstance(msg, (HumanMessage, AIMessage))
    ).lower()
    latest = (latest_user_text or "").strip().lower()

    # 当前问题优先：房产/小区问题优先继承 search，避免被历史天气上下文污染。
    if latest:
        if any(k in latest for k in SEARCH_REAL_ESTATE_KEYWORDS) and not any(k in latest for k in SEARCH_WEATHER_KEYWORDS):
            return "search_agent"
        if any(k in latest for k in SEARCH_WEATHER_KEYWORDS):
            return "weather_agent"

    if any(k in recent for k in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.HISTORY_HOLTER]):
        return "yunyou_agent"
    if any(k in recent for k in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.HISTORY_SQL]):
        return "sql_agent"
    if any(k in recent for k in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.HISTORY_WEATHER]):
        return "weather_agent"
    if any(k in recent for k in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.HISTORY_SEARCH]):
        return "search_agent"
    return None


def _has_recent_weather_fact(messages: List[BaseMessage]) -> bool:
    """判断最近上下文中是否已经包含可复用的天气事实数据。"""
    recent_ai_texts = [
        _content_to_text(getattr(msg, "content", ""))
        for msg in messages[-10:]
        if isinstance(msg, AIMessage)
    ]
    if not recent_ai_texts:
        return False
    joined = "\n".join(recent_ai_texts).lower()
    return any(k in joined for k in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.HISTORY_WEATHER_FACT])


def _looks_like_weather_reuse_query(text: str) -> bool:
    """
    识别“基于已有天气结果做建议”的追问。
    例如：这个天气适合出去吗、要不要带伞、穿什么等。
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(k in t for k in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.WEATHER_REUSE_QUERY])


def _wants_weather_refresh(text: str) -> bool:
    """识别用户是否明确要求重新查询实时天气。"""
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(k in t for k in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.WEATHER_REFRESH_HINT])


def _can_reuse_weather_context(messages: List[BaseMessage], latest_user_text: str) -> bool:
    """
    判断天气追问是否可直接复用历史上下文，避免重复调用 weather_agent。
    仅当：
    1) 用户问的是“建议类追问”
    2) 用户没有明确要求“重新拉取实时天气”
    3) 最近对话里已有天气事实数据
    """
    if not _looks_like_weather_reuse_query(latest_user_text):
        return False
    if _wants_weather_refresh(latest_user_text):
        return False
    return _has_recent_weather_fact(messages)


def _collect_intent_signals(text: str) -> List[str]:
    """
    收集一条输入中命中的意图信号集合（去重后按优先级返回）。

    用途：
    1. 判断是否是“多域混合问题”；
    2. 给规则规划器提供稳定的任务拆分候选。
    """
    t = (text or "").strip().lower()
    if not t:
        return []

    signals: List[str] = []
    holter_hit = _looks_like_holter_request(t)
    sql_hit = _looks_like_sql_request(t)
    weather_hit = _looks_like_weather_request(t)
    search_hit = _looks_like_search_request(t)
    medical_hit = _looks_like_medical_request(t)
    code_hit = _looks_like_code_request(t)

    if holter_hit:
        signals.append("yunyou_agent")
    if sql_hit:
        signals.append("sql_agent")
    # 天气意图：若同句已命中更强业务域，且天气不是显式天气请求，按“背景描述”处理，不单独派发任务。
    weather_is_explicit = _is_weather_actionable_clause(t)
    strong_business_present = holter_hit or sql_hit or medical_hit or code_hit
    if weather_hit and (weather_is_explicit or (not strong_business_present)):
        signals.append("weather_agent")

    # 搜索意图：若仅由泛词触发且同时命中业务域，不单独拆 search 任务。
    search_is_explicit = _is_search_actionable_clause(t)
    if search_hit and (search_is_explicit or (not strong_business_present)):
        signals.append("search_agent")
    if medical_hit:
        signals.append("medical_agent")
    if code_hit:
        signals.append("code_agent")

    # 保持顺序去重，且使用统一优先级，避免节点之间顺序不一致。
    ordered_unique: List[str] = []
    for candidate_name in MULTI_DOMAIN_AGENT_PRIORITY:
        if candidate_name in signals and candidate_name not in ordered_unique:
            ordered_unique.append(candidate_name)
    return ordered_unique


def _dedupe_keep_order(values: List[str]) -> List[str]:
    """按出现顺序去重。"""
    deduped: List[str] = []
    for value_item in values:
        if value_item not in deduped:
            deduped.append(value_item)
    return deduped


def _has_dependency_hint(text: str) -> bool:
    """判断是否包含明显的先后依赖语义。"""
    normalized_text = (text or "").strip().lower()
    if not normalized_text:
        return False
    sequential_patterns = (
        ("先", "再"),
        ("先", "然后"),
        ("第一步", "第二步"),
    )
    for first_hint, second_hint in sequential_patterns:
        if first_hint in normalized_text and second_hint in normalized_text:
            return True
    return False


def _is_explicit_request_clause(text: str) -> bool:
    """判断一个子句是否是“明确要求系统执行动作”的请求子句。"""
    t = (text or "").strip().lower()
    if not t:
        return False
    if any(mark in t for mark in ("?", "？")):
        return True
    if t.endswith(("吗", "么", "呢")):
        return True
    # 兜底：中文疑问起手式通常代表独立诉求，即使未出现“请/帮我/查询”等动作词也应保留为任务子句。
    interrogative_patterns = (
        r"^(怎么|如何|怎样|为何|为什么|能否|可否|是否|要不要|有没有|需不需要)",
        r"(怎么|如何|怎样|为何|为什么|能否|可否|是否|要不要|有没有|需不需要).{0,12}(做|办|处理|操作|安排|推进|落地|实现|解决)",
    )
    if any(re.search(pattern, t) for pattern in interrogative_patterns):
        return True
    action_hints = SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.REQUEST_ACTION_HINT]
    return any(hint in t for hint in action_hints)


def _analyze_request(user_text: str) -> RequestAnalysisDecision:
    """
    统一请求分析器（根因修复核心）。

    设计原则：
    1. 先做“候选域/候选意图”分析，再进入 Domain/Intent/Planner；
    2. 一次分析结果在全链路复用，避免每个节点各自猜测导致冲突。
    """
    normalized_text = (user_text or "").strip().lower()
    if not normalized_text:
        return RequestAnalysisDecision()

    clauses = _split_query_clauses(normalized_text) or [normalized_text]
    candidate_agents = _collect_intent_signals(normalized_text)
    # “显式请求”优先：多子句里只把真正请求子句计入多意图判定，避免背景描述触发误拆分。
    explicit_clause_agents: List[str] = []
    for clause in clauses:
        if not _is_explicit_request_clause(clause):
            continue
        clause_agents = _collect_intent_signals(clause)
        if clause_agents:
            explicit_clause_agents.extend(clause_agents)
        else:
            # 显式请求但未命中垂直域关键词：保留为通用 CHAT 子任务，避免被整句吞掉。
            explicit_clause_agents.append("CHAT")
    explicit_clause_agents = _dedupe_keep_order(explicit_clause_agents)
    if explicit_clause_agents:
        candidate_agents = explicit_clause_agents

    # 根因治理：若存在强业务域（云柚/本地SQL/医疗/代码），
    # 且天气/搜索在本轮未形成显式请求，则移除弱域候选，避免“背景句触发误拆分”。
    strong_agents = {"yunyou_agent", "sql_agent", "medical_agent", "code_agent"}
    if any(agent_name in strong_agents for agent_name in candidate_agents):
        filtered_candidates = list(candidate_agents)
        if "weather_agent" in filtered_candidates:
            weather_explicit = any(_is_weather_actionable_clause(clause) for clause in clauses)
            if not weather_explicit:
                filtered_candidates = [agent_name for agent_name in filtered_candidates if agent_name != "weather_agent"]
        if "search_agent" in filtered_candidates:
            search_explicit = any(_is_search_actionable_clause(clause) for clause in clauses)
            if not search_explicit:
                filtered_candidates = [agent_name for agent_name in filtered_candidates if agent_name != "search_agent"]
        candidate_agents = filtered_candidates

    candidate_domains = _dedupe_keep_order(
        [AGENT_DOMAIN_MAP.get(agent_name, "GENERAL") for agent_name in candidate_agents]
    )
    multi_intent = len(candidate_agents) >= 2
    multi_domain = len([domain for domain in candidate_domains if domain != "GENERAL"]) >= 2
    dependency_hint = _has_dependency_hint(normalized_text)

    if multi_intent:
        return RequestAnalysisDecision(
            candidate_agents=candidate_agents,
            candidate_domains=candidate_domains,
            is_multi_intent=True,
            is_multi_domain=multi_domain,
            has_dependency_hint=dependency_hint,
            route_strategy=RouteStrategy.MULTI_DOMAIN_SPLIT.value,
            reason="multiple_intents_detected",
        )

    if dependency_hint:
        return RequestAnalysisDecision(
            candidate_agents=candidate_agents,
            candidate_domains=candidate_domains or ["GENERAL"],
            is_multi_intent=False,
            is_multi_domain=False,
            has_dependency_hint=True,
            route_strategy=RouteStrategy.COMPLEX_SINGLE_DOMAIN.value,
            reason="dependency_hint_detected",
        )

    return RequestAnalysisDecision(
        candidate_agents=candidate_agents,
        candidate_domains=candidate_domains or ["GENERAL"],
        is_multi_intent=False,
        is_multi_domain=False,
        has_dependency_hint=False,
        route_strategy=RouteStrategy.SINGLE_DOMAIN.value,
        reason="single_domain",
    )


def _split_query_clauses(user_text: str) -> List[str]:
    """将复合问题拆成子句，供规则规划器生成任务。"""
    normalized_text = (user_text or "").strip()
    if not normalized_text:
        return []
    # 逗号也作为软分句符，避免“天气..., 再查 holter ...”被误当成单子句。
    normalized_text = re.sub(r"[，,；;。！？?!]+", "|", normalized_text)
    normalized_text = re.sub(
        r"(并且|而且|同时|然后|再帮我|顺便|另外|以及)",
        "|",
        normalized_text,
    )
    clauses = [item.strip(" ,，\n\t") for item in normalized_text.split("|")]
    return [item for item in clauses if item]


def _select_primary_agent_for_clause(clause_text: str, fallback_candidates: List[str]) -> Optional[str]:
    """为单个子句挑选最合适的 Agent。"""
    clause_candidates = _collect_intent_signals(clause_text)
    # 子句内命中优先，避免被全句 fallback 候选“覆盖”导致错配。
    if clause_candidates:
        return clause_candidates[0]
    for candidate_name in fallback_candidates:
        if candidate_name in MULTI_DOMAIN_AGENT_PRIORITY:
            return candidate_name
    return None


def _extract_agent_focus_text(agent_name: str, clause_text: str, full_text: str) -> str:
    """
    从复合子句中提取与目标 Agent 相关的片段。

    例子：
    - 子句“查天气和holter最近数据”
      - weather_agent -> “查天气”
      - yunyou_agent  -> “holter最近数据”
    """
    normalized_clause = clause_text.strip() or full_text.strip()
    if not normalized_clause:
        return full_text.strip()

    segment_parts = [
        item.strip(" ,，;；\n\t")
        for item in re.split(r"(?:和|并且|以及|然后|再|,|，|;|；)", normalized_clause)
    ]
    segment_parts = [item for item in segment_parts if item]
    if not segment_parts:
        return normalized_clause

    focused_segments: List[str] = []
    for part in segment_parts:
        if agent_name in _collect_intent_signals(part):
            focused_segments.append(part)
    focused_segments = _dedupe_keep_order(focused_segments)
    if focused_segments:
        return "；".join(focused_segments)
    return normalized_clause


def _build_agent_specific_task_input(agent_name: str, clause_text: str, full_text: str) -> str:
    """根据 Agent 类型构建明确且自包含的子任务输入。"""
    normalized_clause = _extract_agent_focus_text(agent_name, clause_text, full_text)
    if agent_name == "weather_agent":
        return (
            "你是天气执行子任务，仅允许输出天气结论与出行建议。"
            "不要输出旅游路线、景点攻略或与天气无关内容。"
            f"\n用户子任务：{normalized_clause}"
        )
    if agent_name == "search_agent":
        return (
            "你是互联网检索子任务，仅允许输出用户明确要求的活动/信息检索结果。"
            "不要扩展到不相关主题。"
            f"\n用户子任务：{normalized_clause}"
        )
    if agent_name == "yunyou_agent":
        return (
            "你是云柚 Holter 数据执行子任务，仅允许输出 Holter 数据查询结果或可执行失败原因。"
            "不要输出本地 SQL 解释，除非用户明确要求 SQL 示例。"
            f"\n用户子任务：{normalized_clause}"
        )
    if agent_name == "sql_agent":
        return f"请仅处理本地数据库 SQL 查询相关诉求：{normalized_clause}"
    if agent_name == "medical_agent":
        return f"请仅处理医疗健康分析相关诉求：{normalized_clause}"
    if agent_name == "code_agent":
        return f"请仅处理代码开发相关诉求：{normalized_clause}"
    return normalized_clause


def _build_rule_based_multidomain_tasks(
    user_text: str,
    *,
    candidate_agents: Optional[List[str]] = None,
    route_strategy: str = RouteStrategy.SINGLE_DOMAIN.value,
) -> Optional[List[SubTask]]:
    """
    规则任务规划器：将复合输入拆成可执行任务列表。

    说明：
    - 多域/多意图问题优先走规则规划，避免慢模型反复试探；
    - 仅在规则无法拆分时回退到 LLM Planner。
    """
    normalized_text = (user_text or "").strip()
    if not normalized_text:
        return None

    fallback_candidates = _dedupe_keep_order(candidate_agents or _collect_intent_signals(normalized_text))

    # 单域复杂任务：至少拆成“专业检索 + CHAT 汇总”两步，确保“先查再说”语义可执行。
    if route_strategy == RouteStrategy.COMPLEX_SINGLE_DOMAIN.value and len(fallback_candidates) == 1:
        primary_agent = fallback_candidates[0]
        first_task: SubTask = {
            "id": "t1",
            "agent": primary_agent,
            "input": _build_agent_specific_task_input(primary_agent, normalized_text, normalized_text),
            "depends_on": [],
            "status": TaskStatus.PENDING.value,
            "result": None,
        }
        # 仅当用户明确需要“总结/分析/建议”等二次加工时，再追加 CHAT 汇总任务。
        summary_hints = ("总结", "分析", "解释", "建议", "报告", "结论", "对比", "方案", "归纳")
        needs_chat_summary = any(hint in normalized_text for hint in summary_hints)
        if not needs_chat_summary:
            return [first_task]

        second_task: SubTask = {
            "id": "t2",
            "agent": "CHAT",
            "input": f"基于 t1 的执行结果，给出用户需要的最终建议与结论：{normalized_text}",
            "depends_on": ["t1"],
            "status": TaskStatus.PENDING.value,
            "result": None,
        }
        return [first_task, second_task]

    if route_strategy != RouteStrategy.MULTI_DOMAIN_SPLIT.value and len(fallback_candidates) < 2:
        return None

    clauses = _split_query_clauses(normalized_text)
    if not clauses:
        clauses = [normalized_text]
    clause_request_flags = [_is_explicit_request_clause(clause) for clause in clauses]
    has_explicit_clause = any(clause_request_flags)

    # 先按子句提取 Agent，若子句无法识别则回退到候选列表。
    clause_pairs: List[tuple[str, str]] = []
    for idx, clause in enumerate(clauses):
        # 存在显式请求子句时，忽略纯背景描述子句，防止“天气不好”被拆成独立任务。
        if has_explicit_clause and (not clause_request_flags[idx]):
            continue
        # 一个显式子句里可能同时包含多个动作（如“查天气并查holter”），需要保留多候选而不是只取第一个。
        clause_candidates = _collect_intent_signals(clause)
        if clause_request_flags[idx] and len(clause_candidates) >= 2:
            for candidate_name in clause_candidates:
                clause_pairs.append((candidate_name, clause))
            continue
        if clause_request_flags[idx]:
            if clause_candidates:
                clause_pairs.append((clause_candidates[0], clause))
            else:
                # 显式请求但无法命中垂直域时，交给通用 CHAT 承接该子句。
                clause_pairs.append(("CHAT", clause))
            continue
        selected_agent = _select_primary_agent_for_clause(clause, fallback_candidates)
        if selected_agent:
            clause_pairs.append((selected_agent, clause))

    if not clause_pairs:
        # 子句级未命中时，先尝试全句主任务；再兜底到候选并行。
        full_agent = _select_primary_agent_for_clause(normalized_text, fallback_candidates)
        if full_agent:
            clause_pairs = [(full_agent, normalized_text)]
        else:
            clause_pairs = [(agent_name, normalized_text) for agent_name in fallback_candidates]

    # 合并连续相同 Agent 的子句，避免重复派发。
    merged_pairs: List[tuple[str, str]] = []
    for agent_name, clause in clause_pairs:
        if merged_pairs and merged_pairs[-1][0] == agent_name:
            merged_agent, merged_clause = merged_pairs[-1]
            merged_pairs[-1] = (merged_agent, f"{merged_clause}；{clause}")
        else:
            merged_pairs.append((agent_name, clause))

    # 根因治理：没有显式子句时采用“保守单任务”策略，避免误拆成并行多任务。
    # 只有明确拆分信号时才进行多任务编排。
    if len(merged_pairs) < 2 and len(fallback_candidates) >= 2 and (not has_explicit_clause):
        merged_pairs = [(fallback_candidates[0], normalized_text)]

    # 单任务也可直接执行，避免被强制回退到 LLM Planner 造成漂移。
    if len(merged_pairs) < 1:
        return None

    sequential_mode = _has_dependency_hint(normalized_text)
    task_list: List[SubTask] = []
    previous_task_id: Optional[str] = None
    for index, (agent_name, clause) in enumerate(merged_pairs, start=1):
        task_id = f"t{index}"
        depends_on = [previous_task_id] if (sequential_mode and previous_task_id) else []
        task_list.append(
            {
                "id": task_id,
                "agent": agent_name,
                "input": _build_agent_specific_task_input(agent_name, clause, normalized_text),
                "depends_on": depends_on,
                "status": TaskStatus.PENDING.value,
                "result": None,
            }
        )
        previous_task_id = task_id
    return task_list


def _build_planner_fallback_tasks(
    *,
    user_text: str,
    intent_candidates: List[str],
    route_strategy: str,
    fallback_intent: str,
) -> List[SubTask]:
    """
    构建确定性 Planner 兜底任务。

    目标：
    1. 在 LLM Planner 不可用/禁用时仍可稳定收敛；
    2. 保证任务拆分结果可解释，不出现“乱拆/跑偏”。
    """
    normalized_text = (user_text or "").strip()
    if not normalized_text:
        normalized_text = "请基于当前会话给出可执行结论。"

    deduped_candidates: List[str] = []
    for candidate_name in intent_candidates:
        if (
            candidate_name in MULTI_DOMAIN_AGENT_PRIORITY
            or candidate_name == "CHAT"
        ) and candidate_name not in deduped_candidates:
            deduped_candidates.append(candidate_name)

    # 单域优先：如果已有明确单一候选，直接生成一个单任务，避免多余 DAG。
    if route_strategy == RouteStrategy.SINGLE_DOMAIN.value and len(deduped_candidates) == 1:
        only_agent = deduped_candidates[0]
        return [
            {
                "id": "t1",
                "agent": only_agent,
                "input": _build_agent_specific_task_input(only_agent, normalized_text, normalized_text),
                "depends_on": [],
                "status": TaskStatus.PENDING.value,
                "result": None,
            }
        ]

    # 多域兜底：按候选优先级构建并行/串行任务，绝不回退为自由生成。
    if route_strategy == RouteStrategy.MULTI_DOMAIN_SPLIT.value and len(deduped_candidates) >= 2:
        sequential_mode = _has_dependency_hint(normalized_text)
        fallback_tasks: List[SubTask] = []
        previous_task_id: Optional[str] = None
        for index, agent_name in enumerate(deduped_candidates, start=1):
            task_id = f"t{index}"
            depends_on = [previous_task_id] if (sequential_mode and previous_task_id) else []
            fallback_tasks.append(
                {
                    "id": task_id,
                    "agent": agent_name,
                    "input": _build_agent_specific_task_input(agent_name, normalized_text, normalized_text),
                    "depends_on": depends_on,
                    "status": TaskStatus.PENDING.value,
                    "result": None,
                }
            )
            previous_task_id = task_id
        return fallback_tasks

    # 复杂单域：优先使用候选，其次使用 intent 字段，最后回落 CHAT。
    if deduped_candidates:
        primary_agent = deduped_candidates[0]
    elif fallback_intent in MEMBERS:
        primary_agent = fallback_intent
    else:
        primary_agent = "CHAT"

    return [
        {
            "id": "t1",
            "agent": primary_agent,
            "input": _build_agent_specific_task_input(primary_agent, normalized_text, normalized_text),
            "depends_on": [],
            "status": TaskStatus.PENDING.value,
            "result": None,
        }
    ]


def _looks_like_compound_request(text: str) -> bool:
    """判断用户输入是否属于复合任务（多意图或有先后依赖）。"""
    analysis = _analyze_request(text)
    if analysis.route_strategy == RouteStrategy.MULTI_DOMAIN_SPLIT.value:
        return True
    if analysis.route_strategy == RouteStrategy.COMPLEX_SINGLE_DOMAIN.value:
        return True

    normalized_text = (text or "").strip().lower()
    if not normalized_text:
        return False
    if len(re.findall(r"[?？]", normalized_text)) >= 2:
        return True
    if any(token in normalized_text for token in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.COMPLEX_CONNECTOR_HINT]):
        return len(normalized_text) >= 30
    return False

def _parse_json_from_text(text: str) -> dict:
    """从 LLM 返回文本中提取 JSON 对象（支持嵌套花括号）。"""
    # 找到第一个 '{' 后进行括号配对，避免非贪婪匹配截断嵌套结构
    start = text.find('{')
    if start == -1:
        # 日志只打印预览，避免把长文本整段刷入日志导致排障困难。
        preview_limit = 800
        preview_text = (text or "")[:preview_limit]
        if len(text or "") > preview_limit:
            preview_text += "...(truncated)"
        log.error(f"Failed to find JSON object in text. Raw text preview:\n{preview_text}")
        raise ValueError("No JSON object found in text")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                json_str = text[start:i+1]
                # 清除由于大模型吐字不规范带来的尾部逗号 (Trailing Commas)
                json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
                return json.loads(json_str)
    raise ValueError("Unbalanced braces in JSON")


def _build_non_streaming_config(config: RunnableConfig) -> RunnableConfig:
    """构建去回调版配置，避免路由/规划阶段把中间 token 噪声写进前端流。"""
    runtime_config = dict(config or {})
    runtime_config.pop("callbacks", None)
    return runtime_config


def _build_worker_history_messages_for_agent(
    *,
    agent_name: str,
    history_messages: List[BaseMessage],
    limit: int,
) -> List[BaseMessage]:
    """
    构建按 Agent 相关性裁剪后的 Worker 历史消息窗口。

    设计目标：
    1. 保留最近用户上下文，减少跨域信息污染；
    2. 让专业 Agent 只看到与自身领域相关的历史用户语句。
    """
    if not history_messages:
        return []

    safe_limit = max(1, int(limit or 1))
    human_messages = [msg for msg in history_messages if isinstance(msg, HumanMessage)]
    if not human_messages:
        return []

    def _message_text(msg: BaseMessage) -> str:
        try:
            return _content_to_text(msg.content).strip().lower()
        except Exception:
            content_val = getattr(msg, "content", "")
            return str(content_val or "").strip().lower()

    def _is_relevant(text: str) -> bool:
        if not text:
            return False
        if agent_name == "yunyou_agent":
            return _looks_like_holter_request(text)
        if agent_name == "sql_agent":
            return _looks_like_sql_request(text)
        if agent_name == "weather_agent":
            return _is_weather_actionable_clause(text) or _looks_like_weather_reuse_query(text)
        if agent_name == "search_agent":
            return _is_search_actionable_clause(text)
        if agent_name == "medical_agent":
            return _looks_like_medical_request(text)
        if agent_name == "code_agent":
            return _looks_like_code_request(text)
        return True  # CHAT 或未知 Agent 保留通用历史

    selected: List[BaseMessage] = []
    latest_msg = human_messages[-1]
    latest_text = _message_text(latest_msg)
    if agent_name in {"CHAT", "chat_node", ""} or _is_relevant(latest_text):
        selected.append(latest_msg)

    for msg in reversed(human_messages[:-1]):
        if len(selected) >= safe_limit:
            break
        if _is_relevant(_message_text(msg)):
            selected.append(msg)

    # 若一个相关历史都没命中，至少保留最新输入，避免子图收到空历史窗口。
    if not selected:
        selected.append(latest_msg)

    # 仅通用聊天补齐非相关历史；垂直 Agent 保持“领域纯净上下文”。
    allow_generic_backfill = agent_name in {"CHAT", "chat_node", ""}
    if allow_generic_backfill and len(selected) < safe_limit:
        for msg in reversed(human_messages[:-1]):
            if len(selected) >= safe_limit:
                break
            if msg not in selected:
                selected.append(msg)

    selected.reverse()
    return selected


def _invoke_with_timeout(callable_fn, *, timeout_sec: float, timeout_label: str):
    """
    对同步模型调用增加超时保护。

    说明：
    - 第三方模型 SDK 在网络抖动时可能长时间阻塞；
    - 这里统一加“可配置超时预算”，超时后快速回退到规则路径。
    """
    safe_timeout = max(1.0, float(timeout_sec or 1.0))
    result_queue: queue.Queue = queue.Queue(maxsize=1)

    def _runner():
        try:
            result_queue.put(("ok", callable_fn()))
        except Exception as exc:
            result_queue.put(("error", exc))

    worker = threading.Thread(target=_runner, daemon=True, name=f"invoke-timeout-{timeout_label}")
    worker.start()
    try:
        status, payload = result_queue.get(timeout=safe_timeout)
    except queue.Empty as exc:
        # 关键：这里不等待 worker 结束，直接快速失败，避免“超时后仍阻塞”。
        raise TimeoutError(f"{timeout_label} 超时: {safe_timeout:.1f}s") from exc

    if status == "error":
        if isinstance(payload, Exception):
            raise payload
        raise RuntimeError(str(payload))
    return payload


def _invoke_structured_output_with_fallback(
    *,
    prompt: ChatPromptTemplate,
    model: BaseChatModel,
    schema: Type[BaseModel],
    inputs: Dict[str, Any],
    config: RunnableConfig,
    max_tokens: int,
    log_name: str,
) -> BaseModel:
    """
    优先使用 LangChain 新版 with_structured_output 做强结构化输出。

    回退顺序：
    1) with_structured_output(schema, method='json_mode')
    2) with_structured_output(schema)
    3) 普通 invoke + 本地 JSON 提取
    """
    runtime_config = _build_non_streaming_config(config)
    invoke_timeout_sec = float(ROUTER_POLICY_CONFIG.router_llm_timeout_sec)
    llm = model

    # 优先绑定低温 + 上限，降低路由节点耗时和漂移。
    try:
        llm = model.bind(temperature=0, max_tokens=max_tokens, response_format={"type": "json_object"})
    except Exception:
        try:
            llm = model.bind(temperature=0, max_tokens=max_tokens)
        except Exception:
            llm = model

    last_structured_exc: Optional[Exception] = None

    if hasattr(llm, "with_structured_output"):
        for kwargs in ({"method": "json_mode"}, {}):
            try:
                structured_model = llm.with_structured_output(schema, **kwargs)
                result = _invoke_with_timeout(
                    lambda: (prompt | structured_model).invoke(inputs, config=runtime_config),
                    timeout_sec=invoke_timeout_sec,
                    timeout_label=f"{log_name}.structured",
                )
                if isinstance(result, schema):
                    return result
                if isinstance(result, dict):
                    return schema(**result)
                if hasattr(result, "model_dump"):
                    return schema(**result.model_dump())
            except Exception as exc:
                last_structured_exc = exc
                continue

    # 回退到传统 JSON 解析
    response = _invoke_with_timeout(
        lambda: (prompt | llm).invoke(inputs, config=runtime_config),
        timeout_sec=invoke_timeout_sec,
        timeout_label=f"{log_name}.json_fallback",
    )
    data = _parse_json_from_text(_content_to_text(getattr(response, "content", response)))
    try:
        return schema(**data)
    except Exception as exc:
        if last_structured_exc is not None:
            log.warning(f"{log_name}: 结构化输出失败，JSON 回退仍失败。structured_error={last_structured_exc}, json_error={exc}")
        raise


def _normalize_interrupt_payload(val: Any) -> dict:
    """将不同形态的 Interrupt 值转成统一 payload。"""
    if hasattr(val, "value"):
        return _normalize_interrupt_payload(getattr(val, "value"))

    if isinstance(val, dict):
        payload = dict(val)
    elif hasattr(val, "__dict__"):
        payload = dict(getattr(val, "__dict__", {}))
    else:
        payload = {"message": str(val)}

    payload.setdefault("message", DEFAULT_INTERRUPT_MESSAGE)
    payload.setdefault("allowed_decisions", list(DEFAULT_ALLOWED_DECISIONS))
    payload.setdefault("action_requests", [])
    return payload


def _extract_interrupt_from_snapshot(snapshot: Any) -> Optional[dict]:
    tasks = getattr(snapshot, "tasks", None) or []
    for task in tasks:
        interrupts = getattr(task, "interrupts", None) or []
        if not interrupts:
            continue
        first_interrupt = interrupts[0]
        payload = getattr(first_interrupt, "value", first_interrupt)
        return _normalize_interrupt_payload(payload)
    return None


def _run_agent_to_completion(
    agent_name: str,
    user_input: str,
    model: BaseChatModel,
    config: RunnableConfig,
    session_id: str = "",
    history_messages: Optional[List[BaseMessage]] = None,
    context_slots: Optional[Dict[str, Any]] = None,
    context_summary: str = "",
) -> Any:
    """共享的 Agent 执行逻辑，供 worker_node 和 single_agent_node 复用。"""
    started_at = time.perf_counter()
    request_id = str(
        config.get("configurable", {}).get("run_id")
        or session_id
        or config.get("configurable", {}).get("thread_id", "")
    ).strip()

    if request_id and request_cancellation_service.is_cancelled(request_id):
        log.info(f"Agent[{agent_name}] 检测到请求已取消，跳过执行。request_id={request_id}")
        return WORKER_CANCELLED_RESULT

    effective_user_input = user_input
    if agent_name in {"search_agent", "weather_agent"}:
        # 优先使用结构化槽位中的城市，其次才回退到历史文本推断
        slot_city = _extract_city_from_context_slots(context_slots)
        inferred_city = slot_city or (
            _extract_recent_city_from_history(history_messages or []) if history_messages else None
        )
        if inferred_city and not _input_has_location_anchor(user_input):
            effective_user_input = (
                f"{user_input}\n\n"
                f"【上下文补全】用户当前所在城市：{inferred_city}。"
                "若用户未明确更换城市，请基于该城市继续回答。"
            )
            source = "context_slots" if slot_city else "history"
            log.info(f"上下文城市继承: {agent_name} 自动沿用城市 [{inferred_city}] (source={source})")

    if agent_name not in MEMBERS:
        # 通用兜底分支也接入会话摘要，避免丢失城市/画像上下文
        fallback_messages: List[Any] = [
            ("system", ChatFallbackPrompt.get_system_prompt()),
            ("system", get_agent_date_context()),
        ]
        if context_summary:
            fallback_messages.append(("system", context_summary))
        fallback_messages.append(HumanMessage(content=effective_user_input))
        # 降级为通用 CHAT
        with request_cancellation_service.bind_request(request_id):
            response = model.invoke(fallback_messages, config=config)
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        log.info(f"⏱️ Agent[CHAT_FALLBACK] 耗时: {elapsed_ms}ms")
        return response.content

    req = AgentRequest(
        user_input=effective_user_input, model=model,
        session_id=session_id or config.get("configurable", {}).get("thread_id", ""),
        subgraph_id=agent_name, llm_config={},
        state={
            "messages": history_messages or [],
            "context_slots": context_slots or {},
            "context_summary": context_summary or "",
        },
    )
    agent_instance = agent_classes[agent_name].cls(req)

    final_response = None
    agent_error = None
    tool_started_at: Dict[str, float] = {}
    tool_name_by_id: Dict[str, str] = {}
    with request_cancellation_service.bind_request(request_id):
        for event in agent_instance.run(req, config=config):
            if request_id and request_cancellation_service.is_cancelled(request_id):
                log.info(f"Agent[{agent_name}] 执行中收到取消信号，提前结束。request_id={request_id}")
                return WORKER_CANCELLED_RESULT

            if not isinstance(event, dict):
                continue
            if "error" in event:
                agent_error = event["error"]
            if "interrupt" in event:
                payload = _normalize_interrupt_payload(event.get("interrupt"))
                payload["agent_name"] = payload.get("agent_name") or agent_name
                elapsed_ms = int((time.perf_counter() - started_at) * 1000)
                log.info(f"⏱️ Agent[{agent_name}] 中断挂起，耗时: {elapsed_ms}ms")
                return {"type": INTERRUPT_RESULT_TYPE, "payload": payload}
            for node_val in event.values():
                if not isinstance(node_val, dict) or "messages" not in node_val:
                    continue
                for msg in node_val.get("messages", []):
                    # 实时代理: 将内部 Agent 的 Tool Calls 推送到日志系统 (这会被 graph_runner 的 interceptor 捕获并推至前端)
                    if getattr(msg, "tool_calls", None):
                        for idx, tc in enumerate(msg.tool_calls):
                            tool_name = tc.get("name", "...")
                            tool_id = str(tc.get("id") or f"{tool_name}_{idx}_{int(time.time() * 1000)}")
                            tool_started_at[tool_id] = time.perf_counter()
                            tool_name_by_id[tool_id] = tool_name
                            log.info(f"被动调度: 正在调用工具 {tool_name} ...")
                    # 寻找最终的纯文本响应
                    if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                        final_response = msg
                    if isinstance(msg, ToolMessage):
                        tool_id = str(getattr(msg, "tool_call_id", "") or getattr(msg, "name", "") or "")
                        started = tool_started_at.pop(tool_id, None) if tool_id else None
                        tool_name = tool_name_by_id.pop(tool_id, None) if tool_id else None
                        if started is not None:
                            elapsed_ms = int((time.perf_counter() - started) * 1000)
                            display_name = tool_name or getattr(msg, "name", None) or tool_id or "unknown_tool"
                            log.info(f"⏱️ 工具[{display_name}] 耗时: {elapsed_ms}ms")

    if final_response:
        normalized = _content_to_text(final_response.content).strip()
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        log.info(f"⏱️ Agent[{agent_name}] 完成，耗时: {elapsed_ms}ms")
        return normalized or "已完成处理，但未生成可展示文本。请重试或更换提问方式。"

    if request_id and request_cancellation_service.is_cancelled(request_id):
        log.info(f"Agent[{agent_name}] 结束前检测到取消信号，返回取消结果。request_id={request_id}")
        return WORKER_CANCELLED_RESULT
        
    # 如果没有最终 response，检查是否因为中断挂起
    if agent_error is None:
        root_thread_id = session_id or config.get("configurable", {}).get("thread_id", "")
        if root_thread_id:
            subgraph_config = {"configurable": {"thread_id": f"{root_thread_id}_{agent_name}"}}
            snapshot = agent_instance.graph.get_state(subgraph_config)
            snapshot_interrupt = _extract_interrupt_from_snapshot(snapshot)
            if snapshot_interrupt:
                snapshot_interrupt["agent_name"] = snapshot_interrupt.get("agent_name") or agent_name
                elapsed_ms = int((time.perf_counter() - started_at) * 1000)
                log.info(f"⏱️ Agent[{agent_name}] 快照中断挂起，耗时: {elapsed_ms}ms")
                return {"type": INTERRUPT_RESULT_TYPE, "payload": snapshot_interrupt}

    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    log.warning(f"⏱️ Agent[{agent_name}] 失败，耗时: {elapsed_ms}ms")
    raise RuntimeError(f"{agent_name} 执行失败: {agent_error}")


# ==================== Tier-0.5: 数据域路由器 (Domain Router) ====================
def domain_router_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
    """
    先识别“去哪类数据域”，再做意图路由。

    数据域定义：
    1. YUNYOU_DB: 云柚/holter 业务域
    2. LOCAL_DB: 本地业务库 SQL 域
    3. WEB_SEARCH: 互联网检索域
    4. GENERAL: 通用对话域
    """
    messages = state.get("messages", [])
    # 分类窗口：避免把过长历史塞给路由器，降低延迟和漂移概率。
    classify_window = max(3, int(ROUTER_POLICY_CONFIG.classifier_history_messages))
    classify_messages = messages[-classify_window:]
    session_id = state.get("session_id") or config.get("configurable", {}).get("thread_id", "")
    latest_user_text = _latest_human_message(classify_messages)
    request_analysis = _analyze_request(latest_user_text)
    started_at = time.perf_counter()

    def _finalize_domain(
        decision: DomainDecision,
        analysis: Optional[RequestAnalysisDecision] = None,
    ) -> dict:
        route_metrics_service.record_domain_decision(
            session_id=session_id,
            user_text=latest_user_text,
            domain=decision.data_domain,
            confidence=decision.confidence,
            source=decision.source,
        )
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        effective_analysis = analysis or request_analysis
        domain_candidates = effective_analysis.candidate_domains or [decision.data_domain]
        intent_candidates = effective_analysis.candidate_agents or []
        log.info(
            f"⏱️ Domain Router [{decision.source}] -> {decision.data_domain} "
            f"(conf={decision.confidence:.2f})，耗时: {elapsed_ms}ms"
        )
        return {
            "data_domain": decision.data_domain,
            "domain_confidence": decision.confidence,
            "domain_route_source": decision.source,
            "domain_candidates": domain_candidates,
            "intent_candidates": intent_candidates,
            "route_strategy": effective_analysis.route_strategy,
            "route_reason": effective_analysis.reason,
            "domain_elapsed_ms": elapsed_ms,
        }

    # follow-up 补充句优先继承上轮域
    if _is_followup_supplement(latest_user_text, classify_messages):
        hinted_intent = _history_hint_intent(classify_messages, latest_user_text)
        if hinted_intent == "yunyou_agent":
            decision = DomainDecision(data_domain="YUNYOU_DB", confidence=0.94, source="history")
        elif hinted_intent == "sql_agent":
            decision = DomainDecision(data_domain="LOCAL_DB", confidence=0.93, source="history")
        elif hinted_intent in {"weather_agent", "search_agent"}:
            decision = DomainDecision(data_domain="WEB_SEARCH", confidence=0.92, source="history")
        else:
            decision = None
        if decision:
            hinted_domain = AGENT_DOMAIN_MAP.get(hinted_intent or "", decision.data_domain)
            followup_analysis = RequestAnalysisDecision(
                candidate_agents=[hinted_intent] if hinted_intent else [],
                candidate_domains=[hinted_domain],
                is_multi_intent=False,
                is_multi_domain=False,
                has_dependency_hint=False,
                route_strategy=RouteStrategy.SINGLE_DOMAIN.value,
                reason="followup_history",
            )
            return _finalize_domain(decision, followup_analysis)

    # 根因修复：多意图请求在 Domain 层即标记为“需要拆分”，防止被单域强路由吞掉。
    if (
        request_analysis.route_strategy == RouteStrategy.MULTI_DOMAIN_SPLIT.value
        and len(request_analysis.candidate_agents) >= 2
    ):
        primary_domain = (request_analysis.candidate_domains or ["GENERAL"])[0]
        decision = DomainDecision(data_domain=primary_domain, confidence=0.97, source="rule_multi_domain")
        return _finalize_domain(decision, request_analysis)

    # 规则优先
    if _looks_like_holter_request(latest_user_text):
        decision = DomainDecision(data_domain="YUNYOU_DB", confidence=0.98, source="rule")
    elif _looks_like_sql_request(latest_user_text):
        decision = DomainDecision(data_domain="LOCAL_DB", confidence=0.95, source="rule")
    elif _looks_like_weather_request(latest_user_text) or _looks_like_search_request(latest_user_text):
        decision = DomainDecision(data_domain="WEB_SEARCH", confidence=0.9, source="rule")
    else:
        # 默认策略：未知场景直接归入 GENERAL，避免每轮都走慢速 LLM 分类。
        # 只有显式打开开关才启用 LLM 兜底。
        if not ROUTER_POLICY_CONFIG.domain_llm_fallback_enabled:
            decision = DomainDecision(data_domain="GENERAL", confidence=0.88, source="rule_default_general")
        else:
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "你是数据域分类器。"
                    "只返回一个 JSON 对象，不要 Markdown、不要解释。"
                    "字段必须包含 data_domain 与 confidence。"
                    "data_domain 只能是 YUNYOU_DB、LOCAL_DB、WEB_SEARCH、GENERAL。",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ])
            try:
                structured = _invoke_structured_output_with_fallback(
                    prompt=prompt,
                    model=model,
                    schema=DomainDecision,
                    inputs={"messages": classify_messages},
                    config=config,
                    max_tokens=96,
                    log_name="domain_router",
                )
                domain = str(structured.data_domain or "GENERAL").upper()
                confidence = float(structured.confidence or 0.5)
                if domain not in {"YUNYOU_DB", "LOCAL_DB", "WEB_SEARCH", "GENERAL"}:
                    domain = "GENERAL"
                decision = DomainDecision(data_domain=domain, confidence=confidence, source="llm")
            except Exception as exc:
                log.warning(f"Domain router fallback: {exc}")
                decision = DomainDecision(data_domain="GENERAL", confidence=0.4, source="fallback")

    single_domain_analysis = RequestAnalysisDecision(
        candidate_agents=request_analysis.candidate_agents,
        candidate_domains=[decision.data_domain],
        is_multi_intent=False,
        is_multi_domain=False,
        has_dependency_hint=request_analysis.has_dependency_hint,
        route_strategy=request_analysis.route_strategy
        if request_analysis.route_strategy == RouteStrategy.COMPLEX_SINGLE_DOMAIN.value
        else RouteStrategy.SINGLE_DOMAIN.value,
        reason=request_analysis.reason,
    )
    return _finalize_domain(decision, single_domain_analysis)


# ==================== Tier-1: 意图路由器 (Intent Router) ====================
def intent_router_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
    """第二级：意图识别节点 (Intent_Router_Node)"""
    messages = state.get("messages", [])
    session_id = state.get("session_id") or config.get("configurable", {}).get("thread_id", "")
    data_domain = (state.get("data_domain") or "GENERAL").upper()
    route_strategy = (state.get("route_strategy") or RouteStrategy.SINGLE_DOMAIN.value).strip()
    route_reason = (state.get("route_reason") or "").strip()
    intent_candidates = [str(item) for item in (state.get("intent_candidates") or []) if isinstance(item, str)]
    domain_conf = float(state.get("domain_confidence") or 0.0)
    domain_source = state.get("domain_route_source") or "unknown"
    # 保留最近多轮上下文，避免“用户补充参数”被误判为新问题导致重复追问。
    classify_window = max(3, int(ROUTER_POLICY_CONFIG.classifier_history_messages))
    trimmed_messages = messages[-classify_window:]

    latest_user_text = _latest_human_message(trimmed_messages)
    started_at = time.perf_counter()

    def _finalize_intent(
        intent: str,
        confidence: float,
        is_complex: bool,
        source: str,
        direct_answer: str = "",
    ) -> dict:
        log.info(f"Tier-1 Router: intent=[{intent}], conf=[{confidence}], complex=[{is_complex}]")
        route_metrics_service.record_intent_decision(
            session_id=session_id,
            user_text=latest_user_text,
            intent=intent,
            confidence=confidence,
            source=source,
        )
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        log.info(f"⏱️ Intent Router [{source}] -> {intent}，耗时: {elapsed_ms}ms")
        return {
            "intent": intent,
            "intent_confidence": confidence,
            "is_complex": is_complex,
            "direct_answer": direct_answer,
            "intent_elapsed_ms": elapsed_ms,
        }

    # 根因修复：多意图输入在 Intent 层直接标记复杂任务，强制走 Parent Planner。
    if route_strategy == RouteStrategy.MULTI_DOMAIN_SPLIT.value and len(intent_candidates) >= 2:
        log.info(
            "Intent multi-domain split: 命中多意图候选，转入 Parent Planner "
            f"(candidates={intent_candidates}, reason={route_reason})"
        )
        return _finalize_intent("CHAT", max(domain_conf, 0.96), True, "analysis_multi_domain", "")

    # 单域但有先后依赖，也强制走规划节点，避免“先查再总结”被误判为单兵。
    if route_strategy == RouteStrategy.COMPLEX_SINGLE_DOMAIN.value and len(intent_candidates) == 1:
        planned_intent = intent_candidates[0]
        return _finalize_intent(planned_intent, max(domain_conf, 0.93), True, "analysis_complex_single", "")

    # 天气追问优化：若上轮已给出天气事实，优先复用上下文做建议，不再重复调 weather_agent。
    if _can_reuse_weather_context(trimmed_messages, latest_user_text):
        log.info("Intent weather reuse: 复用最近天气上下文，直接路由 CHAT")
        return _finalize_intent("CHAT", 0.94, False, "history_weather_reuse", "")

    # 先处理“补充参数”场景：继承上轮领域，避免补充句被 SQL fast-path 截走。
    if _is_followup_supplement(latest_user_text, trimmed_messages):
        hinted_intent = _history_hint_intent(trimmed_messages, latest_user_text)
        if hinted_intent:
            log.info(f"Intent follow-up carry-over: route to [{hinted_intent}]")
            return _finalize_intent(hinted_intent, 0.93, False, "followup_history", "")

    # 先按数据域强约束路由，避免跨域误查
    if data_domain == "YUNYOU_DB":
        return _finalize_intent("yunyou_agent", max(domain_conf, 0.95), False, f"domain_{domain_source}", "")
    if data_domain == "LOCAL_DB" and _looks_like_sql_request(latest_user_text):
        return _finalize_intent("sql_agent", max(domain_conf, 0.94), False, f"domain_{domain_source}", "")
    if data_domain == "WEB_SEARCH":
        hinted_intent = _history_hint_intent(trimmed_messages, latest_user_text)
        if _looks_like_location_fragment(latest_user_text) and hinted_intent in {"weather_agent", "search_agent"}:
            fallback_agent = hinted_intent
        else:
            fallback_agent = "weather_agent" if _looks_like_weather_request(latest_user_text) else "search_agent"
        return _finalize_intent(fallback_agent, max(domain_conf, 0.9), False, f"domain_{domain_source}", "")

    # WEB/GENERAL 的短问题兜底规则：优先走单兵，避免误判成复杂 DAG。
    if _looks_like_weather_request(latest_user_text):
        return _finalize_intent("weather_agent", max(domain_conf, 0.9), False, "rule_weather_fastpath", "")
    if data_domain == "GENERAL" and _looks_like_search_request(latest_user_text):
        return _finalize_intent("search_agent", max(domain_conf, 0.88), False, "rule_search_fastpath", "")

    # 规则快路由：医疗/代码问题优先落到垂直 Agent（若已注册）。
    if _looks_like_medical_request(latest_user_text) and "medical_agent" in MEMBERS:
        return _finalize_intent("medical_agent", max(domain_conf, 0.9), False, "rule_medical_fastpath", "")
    if _looks_like_code_request(latest_user_text) and "code_agent" in MEMBERS:
        return _finalize_intent("code_agent", max(domain_conf, 0.9), False, "rule_code_fastpath", "")

    # 业务域优先路由：Holter/云柚相关查询，优先进入 yunyou_agent。
    # 注意：必须放在 SQL fast-path 之前，否则会被“order by/limit/数据库”误路由到 sql_agent。
    if _looks_like_holter_request(latest_user_text):
        log.info("Intent fast-path: 命中 Holter/云柚业务域，直接路由 yunyou_agent")
        return _finalize_intent("yunyou_agent", 0.96, False, "rule_holter", "")

    # SQL 快速路由：用户明确表达了 SQL/排序/TopN 查询诉求时，优先进入 sql_agent。
    # 业务上“按 id 倒序/前 N 条/order by/limit”这类语句通常是直接查库意图。
    if _looks_like_sql_request(latest_user_text):
        log.info("Intent fast-path: 命中 SQL 语义特征，直接路由 sql_agent")
        return _finalize_intent("sql_agent", 0.95, False, "rule_sql", "")

    # GENERAL 快速通道：纯闲聊默认直达 CHAT，避免慢速 LLM 分类。
    if (
        data_domain == "GENERAL"
        and ROUTER_POLICY_CONFIG.general_chat_fastpath_enabled
        and _looks_like_general_chat_request(latest_user_text)
    ):
        direct_complex = _looks_like_compound_request(latest_user_text)
        return _finalize_intent("CHAT", 0.92, direct_complex, "rule_general_chat_fastpath", "")

    # 默认不走 LLM 意图分类，直接按规则兜底。
    if not ROUTER_POLICY_CONFIG.intent_llm_fallback_enabled:
        direct_complex = _looks_like_compound_request(latest_user_text)
        return _finalize_intent("CHAT", 0.85, direct_complex, "rule_default_chat", "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", IntentRouterPrompt.get_system_prompt()),
        MessagesPlaceholder(variable_name="messages"),
    ])

    decision_source = "llm"

    try:
        structured = _invoke_structured_output_with_fallback(
            prompt=prompt,
            model=model,
            schema=IntentDecision,
            inputs={"messages": trimmed_messages},
            config=config,
            max_tokens=128,
            log_name="intent_router",
        )
        decision = IntentDecision(
            intent=structured.intent or "CHAT",
            confidence=float(structured.confidence or 0.5),
            is_complex=bool(structured.is_complex),
            direct_answer=structured.direct_answer or "",
        )
    except Exception as exc:
        log.warning(f"Intent parsing fallback: {exc}")
        decision_source = "fallback"
        # 如果解析失败但用户输入比较长，宁可错杀交给 Planner 当做复杂任务，避免直通 CHAT 白屏
        user_input_length = len(trimmed_messages[-1].content) if trimmed_messages else 0
        if user_input_length > 50:
            log.info("Fallack 捕获长句 (>50 chars)，强制路由至 Parent_Planner_Node")
            decision = IntentDecision(intent="CHAT", confidence=0.3, is_complex=True, direct_answer="")
        else:
            decision = IntentDecision(intent="CHAT", confidence=0.3, is_complex=False, direct_answer="")

    # 防止简单问句被路由器误判为复杂任务，导致进入 DAG 造成“循环等待”体验。
    if decision.intent == "CHAT" and decision.is_complex and not _looks_like_compound_request(latest_user_text):
        log.info("Intent anti-overplanning: simple CHAT question downgraded to non-complex")
        decision.is_complex = False
        
    return _finalize_intent(
        decision.intent,
        decision.confidence,
        decision.is_complex,
        decision_source,
        decision.direct_answer,
    )

# ==================== Tier-2: DAG Planner & Dispatcher ====================
def parent_planner_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
    """第三级：父规划器 (Parent_Planner_Node)"""
    started_at = time.perf_counter()
    messages = state.get("messages", [])
    latest_user_text = _latest_human_message(messages)
    route_strategy = (state.get("route_strategy") or RouteStrategy.SINGLE_DOMAIN.value).strip()
    intent_candidates = [str(item) for item in (state.get("intent_candidates") or []) if isinstance(item, str)]
    intent_name = str(state.get("intent") or "").strip()

    # 根因修复：优先使用统一规则规划器，避免多域问题被 LLM 规划器抖动影响。
    rule_tasks = _build_rule_based_multidomain_tasks(
        latest_user_text,
        candidate_agents=intent_candidates,
        route_strategy=route_strategy,
    )
    if rule_tasks:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        log.info(f"⏱️ Parent Planner [rule_split] 耗时: {elapsed_ms}ms")
        log.info(f"Tier-2 Planner: Generated {len(rule_tasks)} tasks -> {[t['id'] for t in rule_tasks]}")
        return {
            "task_list": rule_tasks,
            "task_results": {},
            "current_wave": 0,
            "max_waves": len(rule_tasks) * 2 + 2,
            "planner_source": "rule_split",
            "planner_elapsed_ms": elapsed_ms,
        }

    deterministic_tasks = _build_planner_fallback_tasks(
        user_text=latest_user_text,
        intent_candidates=intent_candidates,
        route_strategy=route_strategy,
        fallback_intent=intent_name,
    )

    # 单域任务一律走确定性编排，避免把“明确单任务”交给慢模型规划器导致乱拆和时延抖动。
    if route_strategy == RouteStrategy.SINGLE_DOMAIN.value and deterministic_tasks:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        log.info(f"⏱️ Parent Planner [single_domain_guard] 耗时: {elapsed_ms}ms")
        log.info(f"Tier-2 Planner: Generated {len(deterministic_tasks)} tasks -> {[t['id'] for t in deterministic_tasks]}")
        return {
            "task_list": deterministic_tasks,
            "task_results": {},
            "current_wave": 0,
            "max_waves": len(deterministic_tasks) * 2 + 2,
            "planner_source": "single_domain_guard",
            "planner_elapsed_ms": elapsed_ms,
        }

    # 默认关闭 LLM Planner：生产场景优先稳定性与可解释性，避免模型规划漂移。
    if not ROUTER_POLICY_CONFIG.planner_llm_fallback_enabled:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        log.info(f"⏱️ Parent Planner [deterministic_fallback] 耗时: {elapsed_ms}ms")
        log.info(f"Tier-2 Planner: Generated {len(deterministic_tasks)} tasks -> {[t['id'] for t in deterministic_tasks]}")
        return {
            "task_list": deterministic_tasks,
            "task_results": {},
            "current_wave": 0,
            "max_waves": len(deterministic_tasks) * 2 + 2,
            "planner_source": "deterministic_fallback",
            "planner_elapsed_ms": elapsed_ms,
        }

    prompt = ChatPromptTemplate.from_messages([
        ("system", PlannerPrompt.get_system_prompt()),
        MessagesPlaceholder(variable_name="messages"),
    ])

    planner_source = "llm"
    try:
        structured = _invoke_structured_output_with_fallback(
            prompt=prompt,
            model=model,
            schema=PlannerDecision,
            inputs={"messages": messages},
            config=config,
            max_tokens=384,
            log_name="parent_planner",
        )
        tasks_data = list(structured.tasks or [])

        task_list: List[SubTask] = []
        for idx, t in enumerate(tasks_data):
            task_list.append({
                "id": str(getattr(t, "id", "") or f"t{idx}"),
                "agent": str(getattr(t, "agent", "") or "CHAT"),
                "input": str(getattr(t, "input", "") or ""),
                "depends_on": [str(x) for x in (getattr(t, "depends_on", []) or [])],
                "status": TaskStatus.PENDING.value,
                "result": None
            })

        # LLM 规划器返回空任务时，兜底为单任务，保证链路可收敛。
        if not task_list:
            task_list = [{
                "id": "t1",
                "agent": "CHAT",
                "input": latest_user_text,
                "depends_on": [],
                "status": TaskStatus.PENDING.value,
                "result": None,
            }]
            planner_source = "llm_empty_fallback"

    except Exception as exc:
        log.warning(f"Planner parsing fallback: {exc}")
        planner_source = "fallback_deterministic"
        task_list = deterministic_tasks
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    log.info(f"⏱️ Parent Planner [{planner_source}] 耗时: {elapsed_ms}ms")
    log.info(f"Tier-2 Planner: Generated {len(task_list)} tasks -> {[t['id'] for t in task_list]}")
    return {
        "task_list": task_list,
        "task_results": {},
        "current_wave": 0,
        "max_waves": len(task_list) * 2 + 2,
        "planner_source": planner_source,
        "planner_elapsed_ms": elapsed_ms,
    }


def dispatcher_node(state: GraphState) -> dict:
    """Dispatcher：提取可并发执行的无依赖任务，并标记为 dispatched。"""
    tasks = state.get("task_list", [])
    current_wave = state.get("current_wave", 0)
    task_results = dict(state.get("task_results", {}) or {})
    done_ids = {t["id"] for t in tasks if t.get("status") == TaskStatus.DONE.value}
    status_by_id = {
        str(t.get("id")): str(t.get("status") or "")
        for t in tasks
        if t.get("id")
    }

    active_tasks = []
    new_task_list = []

    for task in tasks:
        new_task = dict(task)
        if new_task.get("status") == TaskStatus.PENDING.value:
            deps = [str(dep) for dep in (new_task.get("depends_on") or []) if dep]
            blocked_deps = [
                dep_id
                for dep_id in deps
                if status_by_id.get(dep_id) in {TaskStatus.ERROR.value, TaskStatus.CANCELLED.value}
            ]
            if blocked_deps:
                blocked_reason = f"依赖任务失败或取消，已跳过执行: {', '.join(blocked_deps)}"
                new_task["status"] = TaskStatus.ERROR.value
                new_task["result"] = blocked_reason
                task_results[str(new_task.get("id"))] = blocked_reason
                new_task_list.append(new_task)
                continue
            # 检查所有依赖是否已完成
            deps_met = all(dep_id in done_ids for dep_id in deps)
            if deps_met:
                new_task["status"] = TaskStatus.DISPATCHED.value
                active_tasks.append(new_task)
        new_task_list.append(new_task)

    log.info(f"Dispatcher [Wave {current_wave}]: {len(active_tasks)} tasks ready to dispatch.")
    return {
        "task_list": new_task_list,
        "task_results": task_results,
        "active_tasks": active_tasks,
        "current_wave": current_wave + 1,
        "worker_results": [],  # 重置本轮 worker 结果缓冲
    }

def dispatch_router(state: GraphState):
    """Conditional Edge：根据 active_tasks 发起扇出，若全部完成则聚合。"""
    active = state.get("active_tasks", [])
    if active:
        # 把会话上下文和最近消息一并透传给 worker，避免并行子任务丢上下文
        worker_context_slots = state.get("context_slots") or {}
        worker_context_summary = state.get("context_summary") or ""
        history_messages = state.get("messages", []) or []
        fanout_payloads: List[Send] = []
        for task_item in active:
            task_agent = str(task_item.get("agent") or "")
            worker_messages = _build_worker_history_messages_for_agent(
                agent_name=task_agent,
                history_messages=history_messages,
                limit=AGENT_LOOP_CONFIG.context_history_messages,
            )
            fanout_payloads.append(
                Send(
                    "worker_node",
                    {
                        "task": task_item,
                        "context_slots": worker_context_slots,
                        "context_summary": worker_context_summary,
                        "messages": worker_messages,
                    },
                )
            )
        # Fan-out 到 worker_node 进行并行执行
        return fanout_payloads
        
    # 如果没要执行的任务，查验是等待别人完成，还是全剧终，还是死锁
    tasks = state.get("task_list", [])
    status_values = [str(t.get("status") or "") for t in tasks]
    if any(status in PENDING_TASK_STATUSES for status in status_values):
        # 异常：死锁或者波次超限
        max_waves = state.get("max_waves", 10)
        current = state.get("current_wave", 0)
        
        # 波次超限：直接退出
        if current >= max_waves:
            log.warning("Dispatcher: DAG execution reached max waves, force quitting.")
            return "aggregator_node"
            
        # 死锁检测：图里有 pending，且没有 dispatched 在跑（全军覆没卡死在等待依赖上）
        _has_dispatched = any(status == TaskStatus.DISPATCHED.value for status in status_values)
        _has_pending = any(status == TaskStatus.PENDING.value for status in status_values)
        _has_approval = any(status == TaskStatus.PENDING_APPROVAL.value for status in status_values)
        
        if _has_pending and not _has_dispatched and not _has_approval:
            log.warning("Dispatcher: 💥 Dependency deadlock detected (pending 任务无法满足依赖)，强制进入聚合收敛。")
            return "aggregator_node"
            
        # 否则如果是还在等待 dispatched 或审批任务完成，直接由于没有 active 返回，这里 LangGraph 会暂停 (因为没有出边可走)
        if _has_dispatched or _has_approval:
            log.info("Dispatcher: 当前仅剩 dispatched/pending_approval 任务，进入聚合节点做无正文收敛。")
            return "aggregator_node"
            
    return "aggregator_node"


# ==================== Worker & Reducer ====================
def worker_node(state: WorkerState, config: RunnableConfig, model: BaseChatModel) -> dict:
    """Worker节点：执行单个子任务并返回结果"""
    started_at = time.perf_counter()
    task = state["task"]
    log.info(f"Worker start: task=[{task['id']}], agent=[{task['agent']}]")
    request_id = str(
        config.get("configurable", {}).get("run_id")
        or config.get("configurable", {}).get("thread_id")
        or ""
    ).strip()

    if request_id and request_cancellation_service.is_cancelled(request_id):
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        log.info(f"Worker [{task['id']}] 命中取消信号，跳过执行。request_id={request_id}")
        return {
            "worker_results": [
                WorkerResult(
                    task_id=task["id"],
                    result=WORKER_CANCELLED_RESULT,
                    error=None,
                    agent=task.get("agent"),
                    elapsed_ms=elapsed_ms,
                )
            ]
        }

    try:
        res_text = _run_agent_to_completion(
            agent_name=task["agent"],
            user_input=task["input"],
            model=model,
            config=config,
            history_messages=state.get("messages", []),
            context_slots=state.get("context_slots", {}),
            context_summary=state.get("context_summary", ""),
        )
        if res_text == WORKER_CANCELLED_RESULT:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            log.info(f"⏱️ Worker[{task['id']}] 已取消，耗时: {elapsed_ms}ms")
            return {
                "worker_results": [
                    WorkerResult(
                        task_id=task["id"],
                        result=WORKER_CANCELLED_RESULT,
                        error=None,
                        agent=task.get("agent"),
                        elapsed_ms=elapsed_ms,
                    )
                ]
            }
        if isinstance(res_text, dict) and res_text.get("type") == INTERRUPT_RESULT_TYPE:
            log.info(f"Worker [{task['id']}] 中断挂起，等待人工审批")
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            log.info(f"⏱️ Worker[{task['id']}] 挂起前耗时: {elapsed_ms}ms")
            payload = _normalize_interrupt_payload(res_text.get("payload"))
            payload["agent_name"] = payload.get("agent_name") or task["agent"]
            return {
                "worker_results": [
                    WorkerResult(
                        task_id=task["id"],
                        result=WORKER_PENDING_APPROVAL_RESULT,
                        error=None,
                        agent=task.get("agent"),
                        elapsed_ms=elapsed_ms,
                    )
                ],
                "interrupt_payload": payload
            }
    except Exception as exc:
        err_msg = str(exc)
        if "Interrupt(" in err_msg or exc.__class__.__name__ == "GraphInterrupt":
            log.info(f"Worker [{task['id']}] 中断挂起 (通过异常捕获)，等待人工审批")
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            log.info(f"⏱️ Worker[{task['id']}] 异常挂起前耗时: {elapsed_ms}ms")
            return {
                "worker_results": [
                    WorkerResult(
                        task_id=task["id"],
                        result=WORKER_PENDING_APPROVAL_RESULT,
                        error=None,
                        agent=task.get("agent"),
                        elapsed_ms=elapsed_ms,
                    )
                ],
                "interrupt_payload": {
                    "message": DEFAULT_INTERRUPT_MESSAGE,
                    "allowed_decisions": list(DEFAULT_ALLOWED_DECISIONS),
                    "action_requests": [],
                    "agent_name": task["agent"],
                }
            }
        log.error(f"Worker [{task['id']}] error: {exc}")
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        log.info(f"⏱️ Worker[{task['id']}] 失败耗时: {elapsed_ms}ms")
        return {
            "worker_results": [
                WorkerResult(
                    task_id=task["id"],
                    result="",
                    error=str(exc),
                    agent=task.get("agent"),
                    elapsed_ms=elapsed_ms,
                )
            ]
        }

    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    log.info(f"⏱️ Worker[{task['id']}] 完成耗时: {elapsed_ms}ms")
    return {
        "worker_results": [
            WorkerResult(
                task_id=task["id"],
                result=res_text,
                error=None,
                agent=task.get("agent"),
                elapsed_ms=elapsed_ms,
            )
        ]
    }


def reducer_node(state: GraphState) -> dict:
    """Reducer节点：回收并发结果，更新全图task_list状态"""
    new_results = state.get("worker_results", [])
    if not new_results:
        return {}

    tasks = state.get("task_list", [])
    task_res_map = state.get("task_results", {})

    new_task_list = []

    for task in tasks:
        new_task = dict(task)
        # 寻找匹配的执行结果
        matched = [r for r in new_results if r["task_id"] == task["id"]]
        if matched:
            worker_res = matched[0] # 取出其中一个
            if new_task.get("status") not in {
                TaskStatus.DONE.value,
                TaskStatus.ERROR.value,
                TaskStatus.CANCELLED.value,
            }:  # 防止重复标记
                if worker_res.get("result") == WORKER_PENDING_APPROVAL_RESULT:
                    new_task["status"] = TaskStatus.PENDING_APPROVAL.value  # 更改状态为 pending_approval 阻塞 DAG 但不触发死锁
                    log.info(f"Reducer: task [{task['id']}] 挂起，等待人工审批完成。")
                elif worker_res.get("result") == WORKER_CANCELLED_RESULT:
                    new_task["status"] = TaskStatus.CANCELLED.value
                    new_task["result"] = "任务已因客户端停止生成而取消。"
                    task_res_map[task["id"]] = new_task["result"]
                    log.info(f"Reducer: task [{task['id']}] cancelled by client.")
                else:
                    new_task["status"] = TaskStatus.DONE.value
                    if worker_res.get("error"):
                        new_task["result"] = f"Error: {worker_res['error']}"
                        new_task["status"] = TaskStatus.ERROR.value
                    else:
                        new_task["result"] = worker_res["result"]

                    task_res_map[task["id"]] = new_task["result"]
                    log.info(f"Reducer: task [{task['id']}] finished.")

        new_task_list.append(new_task)

    return {"task_list": new_task_list, "task_results": task_res_map}


# ==================== Aggregator ====================
def _normalize_aggregator_result_text(task_id: str, value: Any, max_chars: int) -> str:
    """将子任务结果统一转成可展示文本，并控制长度。"""
    if isinstance(value, str):
        text = value.strip()
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, indent=2)
        except Exception:
            text = str(value)

    if not text:
        text = "（该子任务未返回可展示内容）"

    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + f"\n...（任务 {task_id} 结果过长，已截断）"
    return text


def _build_deterministic_aggregation(
    user_request: str,
    normalized_results: List[tuple[str, str]],
) -> str:
    """构造稳定、快速、不依赖额外大模型调用的最终聚合结果。"""
    if not normalized_results:
        return "没有任何子任务结果可以聚合。"

    # 单任务场景直接返回正文，避免重复包裹影响阅读。
    if len(normalized_results) == 1:
        return normalized_results[0][1]

    lines: List[str] = []
    if user_request:
        lines.append(f"已完成你的请求：{user_request}")
        lines.append("")
    lines.append(f"共完成 {len(normalized_results)} 个子任务，汇总如下：")
    for task_id, text in normalized_results:
        lines.append(f"\n【{task_id}】\n{text}")
    return "\n".join(lines).strip()


def aggregator_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
    """聚合节点：将多个子任务结果整合成最终回答"""
    started_at = time.perf_counter()
    tasks = state.get("task_list", []) or []
    pending_like = [t for t in tasks if str(t.get("status") or "") in PENDING_TASK_STATUSES]
    if pending_like:
        log.info("Aggregator: 检测到未终态任务（含 pending/pending_approval/dispatched），本轮不输出正文。")
        return {"direct_answer": ""}

    results = state.get("task_results", {}) or {}
    if not results:
        msg = AIMessage(
            content="没有任何子任务结果可以聚合。",
            name="Aggregator",
            response_metadata={"synthetic": True, "force_emit": True},
        )
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        log.info(f"⏱️ Aggregator 空结果耗时: {elapsed_ms}ms")
        return {"messages": [msg], "direct_answer": "没有任何子任务结果可以聚合。"}

    def _task_sort_key(item: tuple[str, Any]) -> tuple[int, str]:
        """按任务编号排序（t1/t2...），保证输出顺序稳定。"""
        task_id = str(item[0])
        match = re.search(r"\d+", task_id)
        seq = int(match.group(0)) if match else 10**9
        return seq, task_id

    max_chars = AGGREGATOR_CONFIG.max_result_chars
    normalized_results: List[tuple[str, str]] = []
    for task_id, value in sorted(results.items(), key=_task_sort_key):
        normalized_results.append(
            (str(task_id), _normalize_aggregator_result_text(str(task_id), value, max_chars))
        )

    user_request = _latest_human_message(state.get("messages", []))
    deterministic_answer = _build_deterministic_aggregation(user_request, normalized_results)

    # 默认走确定性聚合，避免“任务都完成了但又卡在二次 LLM 聚合”的慢阻塞链路。
    if not AGGREGATOR_CONFIG.use_llm_aggregation:
        msg = AIMessage(
            content=deterministic_answer,
            name="Aggregator",
            response_metadata={"synthetic": True, "force_emit": True},
        )
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        log.info(f"⏱️ Aggregator [deterministic] 耗时: {elapsed_ms}ms")
        return {"messages": [msg], "direct_answer": deterministic_answer}

    # 可选：启用 LLM 聚合时，失败自动回退确定性聚合，保证链路可收敛。
    final_content = deterministic_answer
    agg_mode = "deterministic_fallback"
    try:
        res_list = [f"【任务 {task_id} 的反馈】:\n{text}" for task_id, text in normalized_results]
        agg_msg = "\n\n---\n\n".join(res_list)

        prompt = ChatPromptTemplate.from_messages([
            ("system", AggregatorPrompt.SYSTEM),
            ("human", f"用户的原始请求：\n{user_request}\n\n执行结果反馈：\n{agg_msg}")
        ])

        response = (prompt | model).invoke({}, config=config)
        parsed_content = _content_to_text(getattr(response, "content", "")).strip()
        if parsed_content:
            final_content = parsed_content
            agg_mode = "llm"
    except Exception as exc:
        log.warning(f"Aggregator LLM 聚合失败，已回退确定性聚合: {exc}")

    msg = AIMessage(
        content=final_content,
        name="Aggregator",
        response_metadata={"synthetic": True, "force_emit": True},
    )
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    log.info(f"⏱️ Aggregator [{agg_mode}] 耗时: {elapsed_ms}ms")
    return {"messages": [msg], "direct_answer": final_content}


def chat_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
    """聊天节点：处理通用对话"""
    started_at = time.perf_counter()
    direct_ans = state.get("direct_answer", "")
    if direct_ans and len(direct_ans.strip()) > 3:
        msg = AIMessage(content=direct_ans, name="ChatAgent", response_metadata={"synthetic": True})
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        log.info(f"⏱️ chat_node 直出耗时: {elapsed_ms}ms")
    else:
        prompt = ChatFallbackPrompt.get_system_prompt()
        # Chat 兜底只保留最近窗口，避免超长历史拖慢响应。
        recent_messages = state.get("messages", [])[-AGENT_LOOP_CONFIG.context_history_messages:]
        request_messages = [("system", prompt), ("system", get_agent_date_context())] + recent_messages
        runtime_config = _build_non_streaming_config(config)
        run_id = str(
            config.get("configurable", {}).get("run_id")
            or state.get("session_id")
            or config.get("configurable", {}).get("thread_id")
            or ""
        ).strip()

        first_token_timeout_sec = float(CHAT_NODE_FIRST_TOKEN_TIMEOUT_SEC)
        total_timeout_sec = float(CHAT_NODE_TOTAL_TIMEOUT_SEC)

        def _publish_live_chunk(text: str) -> bool:
            if not (AGENT_LIVE_STREAM_ENABLED and run_id and text.strip()):
                return False
            agent_stream_bus.publish(
                run_id=run_id,
                agent_name="ChatAgent",
                content=text,
            )
            return True

        def _build_retry_messages() -> list[Any]:
            compact_window = max(1, min(4, AGENT_LOOP_CONFIG.context_history_messages))
            compact_history = recent_messages[-compact_window:]
            return [("system", prompt), ("system", get_agent_date_context())] + compact_history

        def _retry_compact_invoke() -> str:
            retry_timeout_sec = min(max(total_timeout_sec * 0.4, 1.0), 8.0)
            response = _invoke_with_timeout(
                lambda: model.invoke(_build_retry_messages(), config=runtime_config),
                timeout_sec=retry_timeout_sec,
                timeout_label="chat_node.retry_invoke",
            )
            return _content_to_text(getattr(response, "content", "")).strip()

        def _stream_with_timeout() -> tuple[str, bool]:
            """
            用队列桥接同步节点与流式模型，支持首 token/总时长双超时。

            Returns:
                (content, live_streamed)
            """
            live_streamed = False
            assembled_parts: list[str] = []
            event_queue: queue.Queue = queue.Queue()
            stop_signal = threading.Event()

            def _producer():
                try:
                    for chunk in model.stream(request_messages, config=runtime_config):
                        if stop_signal.is_set():
                            break
                        piece = _content_to_text(getattr(chunk, "content", chunk))
                        if piece:
                            event_queue.put(("chunk", piece))
                    event_queue.put(("done", None))
                except Exception as exc:
                    event_queue.put(("error", exc))

            producer_thread = threading.Thread(target=_producer, daemon=True)
            producer_thread.start()

            started = time.perf_counter()
            first_deadline = started + max(0.1, first_token_timeout_sec)
            total_deadline = started + max(0.2, total_timeout_sec)
            first_chunk_seen = False

            try:
                while True:
                    now = time.perf_counter()
                    if now >= total_deadline:
                        stop_signal.set()
                        raise _ChatNodeStreamFailure(
                            f"chat_node.total_timeout: {total_timeout_sec:.1f}s",
                            partial_output_emitted=bool(assembled_parts),
                        )

                    wait_timeout = min(0.2, max(0.01, total_deadline - now))
                    if not first_chunk_seen:
                        wait_timeout = min(wait_timeout, max(0.01, first_deadline - now))

                    try:
                        event_type, payload = event_queue.get(timeout=wait_timeout)
                    except queue.Empty:
                        if (not first_chunk_seen) and (time.perf_counter() >= first_deadline):
                            stop_signal.set()
                            raise _ChatNodeStreamFailure(
                                f"chat_node.first_token_timeout: {first_token_timeout_sec:.1f}s",
                                partial_output_emitted=bool(assembled_parts),
                            )
                        continue

                    if event_type == "chunk":
                        first_chunk_seen = True
                        chunk_text = str(payload)
                        assembled_parts.append(chunk_text)
                        if _publish_live_chunk(chunk_text):
                            live_streamed = True
                        continue

                    if event_type == "error":
                        raise _ChatNodeStreamFailure(
                            str(payload),
                            partial_output_emitted=bool(assembled_parts),
                        )

                    if event_type == "done":
                        break
            finally:
                stop_signal.set()
                producer_thread.join(timeout=0.2)

            return "".join(assembled_parts).strip(), live_streamed

        msg: Optional[AIMessage] = None
        final_error: Optional[Exception] = None

        if CHAT_NODE_STREAM_ENABLED:
            try:
                streamed_content, live_streamed = _stream_with_timeout()
                if streamed_content:
                    msg = AIMessage(
                        content=streamed_content,
                        name="ChatAgent",
                        response_metadata={"synthetic": True, "live_streamed": live_streamed},
                    )
                else:
                    final_error = RuntimeError("chat_node.empty_response")
            except Exception as exc:
                final_error = exc
                partial_output_emitted = bool(getattr(exc, "partial_output_emitted", False))
                can_retry = (
                    (not partial_output_emitted)
                    and total_timeout_sec >= 1.0
                    and (not request_cancellation_service.is_cancelled(run_id))
                )
                if can_retry:
                    try:
                        retry_content = _retry_compact_invoke()
                        if retry_content:
                            msg = AIMessage(
                                content=retry_content,
                                name="ChatAgent",
                                response_metadata={"synthetic": True, "force_emit": True},
                            )
                            final_error = None
                    except Exception as retry_exc:
                        final_error = retry_exc
        else:
            try:
                response = _invoke_with_timeout(
                    lambda: model.invoke(request_messages, config=runtime_config),
                    timeout_sec=max(1.0, total_timeout_sec),
                    timeout_label="chat_node.invoke",
                )
                content = _content_to_text(getattr(response, "content", "")).strip()
                if content:
                    msg = AIMessage(
                        content=content,
                        name="ChatAgent",
                        response_metadata={"synthetic": True},
                    )
                else:
                    final_error = RuntimeError("chat_node.empty_response")
            except Exception as exc:
                final_error = exc
                if total_timeout_sec >= 1.0 and not request_cancellation_service.is_cancelled(run_id):
                    try:
                        retry_content = _retry_compact_invoke()
                        if retry_content:
                            msg = AIMessage(
                                content=retry_content,
                                name="ChatAgent",
                                response_metadata={"synthetic": True},
                            )
                            final_error = None
                    except Exception as retry_exc:
                        final_error = retry_exc

        if msg is None:
            user_message, error_detail = _classify_agent_failure(final_error or RuntimeError("chat_node.empty_response"))
            log.warning(f"chat_node 调用失败，返回 error 事件。detail={error_detail}")
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            log.info(f"⏱️ chat_node 模型耗时: {elapsed_ms}ms")
            return {"error_message": user_message, "error_detail": error_detail}

        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        log.info(f"⏱️ chat_node 模型耗时: {elapsed_ms}ms")
    return {"messages": [msg]}


def single_agent_node(state: GraphState, agent_name: str, model: BaseChatModel, config: RunnableConfig) -> dict:
    """单一Agent节点：执行单个专业Agent"""
    user_input = _latest_human_message(state.get("messages", []))
    try:
        history_messages = _build_worker_history_messages_for_agent(
            agent_name=agent_name,
            history_messages=state.get("messages", []) or [],
            limit=AGENT_LOOP_CONFIG.context_history_messages,
        )
        content = _run_agent_to_completion(
            agent_name, user_input, model, config,
            session_id=state.get("session_id") or "",
            history_messages=history_messages,
            context_slots=state.get("context_slots") or {},
            context_summary=state.get("context_summary") or "",
        )
        if content == WORKER_CANCELLED_RESULT:
            return {"direct_answer": ""}
        if isinstance(content, dict) and content.get("type") == INTERRUPT_RESULT_TYPE:
            payload = _normalize_interrupt_payload(content.get("payload"))
            payload["agent_name"] = payload.get("agent_name") or agent_name
            return {"interrupt_payload": payload}
        run_id = str(
            config.get("configurable", {}).get("run_id")
            or state.get("session_id")
            or config.get("configurable", {}).get("thread_id")
            or ""
        ).strip()
        live_streamed = AGENT_LIVE_STREAM_ENABLED and agent_stream_bus.has_streamed(run_id, agent_name)
        res_msg = AIMessage(
            content=content,
            name=agent_name,
            response_metadata={"synthetic": True, "live_streamed": live_streamed},
        )
    except Exception as exc:
        err_msg = str(exc)
        if "Interrupt(" in err_msg or exc.__class__.__name__ == "GraphInterrupt":
            log.info(f"Single Agent [{agent_name}] 中断挂起，等待人工审批")
            return {
                "interrupt_payload": {
                    "message": DEFAULT_INTERRUPT_MESSAGE,
                    "allowed_decisions": list(DEFAULT_ALLOWED_DECISIONS),
                    "action_requests": [],
                    "agent_name": agent_name,
                }
            }
        user_message, error_detail = _classify_agent_failure(exc)
        log.warning(f"Single Agent [{agent_name}] 执行失败，返回 error 事件。detail={error_detail}")
        return {"error_message": user_message, "error_detail": error_detail}
    return {"messages": [res_msg]}


# ==================== 路由与编排逻辑 ====================
def _route_after_intent(state: GraphState) -> str:
    """
    条件边：Intent_Router → 单兵/DAG/聊天

    决策逻辑（与架构规范对齐）：
    1. is_complex=True → Parent_Planner_Node（DAG 拆解）
    2. confidence >= 0.7 且 intent 在 MEMBERS 中 → 直接路由到专业 Agent
    3. 其余情况 → chat_node（通用对话兜底）
    """
    conf = state.get("intent_confidence", 0.0)
    is_comp = state.get("is_complex", False)
    intent = state.get("intent", "CHAT")

    # 复杂任务 → 无论置信度高低，都进入 DAG 拆解
    if is_comp:
        log.info(f"路由决策: 复杂任务 → Parent_Planner_Node (intent={intent}, conf={conf:.2f})")
        return "Parent_Planner_Node"

    # 高置信单一意图 → 直接路由到对应 Agent
    if conf >= 0.7 and intent in MEMBERS:
        log.info(f"路由决策: 高置信单兵 → {intent} (conf={conf:.2f})")
        return intent

    # 兜底 → 通用对话
    log.info(f"路由决策: 兜底对话 → chat_node (intent={intent}, conf={conf:.2f})")
    return "chat_node"


def create_graph(model_config: Optional[dict] = None):
    """构建遵循生产级 3 层逻辑拓扑结构的融合 StateGraph"""
    config_dict = model_config or {}
    model, _ = create_model_from_config(**config_dict)

    # 优先读取请求配置，其次使用环境变量默认值
    router_model_name = str(config_dict.get("router_model") or MODEL_TIERING_CONFIG.router_model or "").strip()
    simple_chat_model_name = str(
        config_dict.get("simple_chat_model") or MODEL_TIERING_CONFIG.simple_chat_model or ""
    ).strip()
    # 若未单独指定简单对话模型，则默认复用路由小模型，实现“简单问题走小模型”。
    if (not simple_chat_model_name) and router_model_name:
        simple_chat_model_name = router_model_name

    router_model = model
    chat_model = model

    if router_model_name and router_model_name != config_dict.get("model"):
        try:
            router_config = dict(config_dict)
            router_config["model"] = router_model_name
            # 这里默认共享主模型的 service_type 和 keys
            temp_router_model, _ = create_model_from_config(**router_config)
            router_model = temp_router_model
            log.info(f"Tier-1 极速路由引擎已挂载小模型: {router_model_name}")
        except Exception as e:
            log.warning(f"挂载小模型路由 [{router_model_name}] 失败，回退至主模型: {e}")

    if simple_chat_model_name and simple_chat_model_name != config_dict.get("model"):
        # 若聊天小模型和路由小模型相同，复用已加载实例，避免重复初始化。
        if simple_chat_model_name == router_model_name and router_model is not model:
            chat_model = router_model
            log.info(f"简单对话引擎复用路由小模型: {simple_chat_model_name}")
        else:
            try:
                chat_config = dict(config_dict)
                chat_config["model"] = simple_chat_model_name
                temp_chat_model, _ = create_model_from_config(**chat_config)
                chat_model = temp_chat_model
                log.info(f"简单对话引擎已挂载小模型: {simple_chat_model_name}")
            except Exception as e:
                log.warning(f"挂载简单对话小模型 [{simple_chat_model_name}] 失败，回退至主模型: {e}")

    workflow = StateGraph(GraphState)

    # 上层的决策分析
    workflow.add_node("Domain_Router_Node", functools.partial(domain_router_node, model=router_model))
    workflow.add_node("Intent_Router_Node", functools.partial(intent_router_node, model=router_model))
    workflow.add_node("Parent_Planner_Node", functools.partial(parent_planner_node, model=model))
    
    # 动态 DAG 发牌与执行网络
    workflow.add_node("dispatcher_node", dispatcher_node)
    workflow.add_node("worker_node", functools.partial(worker_node, model=model))
    workflow.add_node("reducer_node", reducer_node)
    workflow.add_node("aggregator_node", functools.partial(aggregator_node, model=model))
    
    # 单兵节点 (非 DAG 路径使用): 简单单意图优先走轻量模型，复杂任务仍走 DAG+主模型
    workflow.add_node("chat_node", functools.partial(chat_node, model=chat_model))
    for name in MEMBERS:
        workflow.add_node(name, functools.partial(single_agent_node, agent_name=name, model=chat_model))

    # ================= 编织拓扑关系 =================
    workflow.add_edge(START, "Domain_Router_Node")
    workflow.add_edge("Domain_Router_Node", "Intent_Router_Node")
    
    router_options = {name: name for name in MEMBERS}
    router_options.update({"chat_node": "chat_node", "Parent_Planner_Node": "Parent_Planner_Node"})
    workflow.add_conditional_edges("Intent_Router_Node", _route_after_intent, router_options)
    
    # DAG 循环执行部分
    workflow.add_edge("Parent_Planner_Node", "dispatcher_node")
    workflow.add_conditional_edges("dispatcher_node", dispatch_router, ["worker_node", "aggregator_node"])
    workflow.add_edge("worker_node", "reducer_node")
    workflow.add_edge("reducer_node", "dispatcher_node") # 闭环：拉取下一波任务
    
    # 单兵出口
    workflow.add_edge("chat_node", END)
    for name in MEMBERS:
        workflow.add_edge(name, END)
        
    workflow.add_edge("aggregator_node", END)

    return workflow.compile(checkpointer=get_checkpointer("supervisor"))
