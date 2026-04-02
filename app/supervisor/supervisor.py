import asyncio
import functools
import json
import re
import time
from typing import Optional, List, Dict, Any, TypedDict, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, RemoveMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Send
from pydantic import BaseModel

from supervisor.graph_state import AgentRequest
from supervisor.checkpointer import get_checkpointer
from supervisor.state import GraphState, SubTask, WorkerResult
from supervisor.supervisor_support import (
    build_non_streaming_config as _support_build_non_streaming_config,
    can_reuse_weather_context as _support_can_reuse_weather_context,
    content_to_text as _support_content_to_text,
    extract_city_from_context_slots as _support_extract_city_from_context_slots,
    extract_recent_city_from_history as _support_extract_recent_city_from_history,
    extract_interrupt_from_snapshot as _support_extract_interrupt_from_snapshot,
    has_recent_weather_fact as _support_has_recent_weather_fact,
    history_hint_intent as _support_history_hint_intent,
    history_requests_location as _support_history_requests_location,
    input_has_location_anchor as _support_input_has_location_anchor,
    is_followup_supplement as _support_is_followup_supplement,
    invoke_with_timeout as _support_invoke_with_timeout,
    latest_human_message as _support_latest_human_message,
    looks_like_location_fragment as _support_looks_like_location_fragment,
    looks_like_weather_reuse_query as _support_looks_like_weather_reuse_query,
    normalize_interrupt_payload as _support_normalize_interrupt_payload,
    parse_json_from_text as _support_parse_json_from_text,
    wants_weather_refresh as _support_wants_weather_refresh,
)
from supervisor.supervisor_rule_support import (
    analyze_request_payload as _support_analyze_request_payload,
    collect_intent_signals as _support_collect_intent_signals,
    dedupe_keep_order as _support_dedupe_keep_order,
    has_dependency_hint as _support_has_dependency_hint,
    is_explicit_request_clause as _support_is_explicit_request_clause,
    looks_like_compound_request as _support_looks_like_compound_request,
    select_primary_agent_for_clause as _support_select_primary_agent_for_clause,
    split_query_clauses as _support_split_query_clauses,
)
from common.llm.unified_loader import create_model_from_config
from prompts.agent_prompts.domain_prompt import DomainPrompt
from supervisor.registry import agent_classes, MEMBERS
from prompts.agent_prompts.supervisor_prompt import (
    IntentRouterPrompt,
    ChatFallbackPrompt,
    PlannerPrompt,
    ReflectionPrompt,
    AggregatorPrompt,
)
from config.constants.agent_registry_keywords import AGENT_KEYWORDS, AgentKeywordGroup
from config.constants.approval_constants import DEFAULT_ALLOWED_DECISIONS, DEFAULT_INTERRUPT_MESSAGE
from config.constants.supervisor_keywords import (
    SUPERVISOR_AGENT_FAILURE_CONNECTION_MARKERS,
    SUPERVISOR_AGENT_FAILURE_TIMEOUT_MARKERS,
    SUPERVISOR_CODE_ACTION_HINTS,
    SUPERVISOR_CODE_LANGUAGE_ONLY_MARKERS,
    SUPERVISOR_CODE_LEARNING_HINTS,
    SUPERVISOR_CODE_SNIPPET_PATTERNS,
    SUPERVISOR_KEYWORDS,
    SUPERVISOR_SEARCH_ACTION_HINTS,
    SUPERVISOR_SEARCH_GENERIC_QUERY_HINTS,
    SUPERVISOR_SQL_EXPLICIT_ANCHORS,
    SUPERVISOR_WEATHER_ACTION_PATTERNS,
    SupervisorKeywordGroup,
)
from config.constants.workflow_constants import (
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
    CHAT_NODE_TOTAL_TIMEOUT_SEC,
    MODEL_TIERING_CONFIG,
    ROUTER_POLICY_CONFIG,
    WORKFLOW_REFLECTION_CONFIG,
    WORKFLOW_MAX_REPLANS,
)
from common.enums.agent_enum import AgentTypeEnum
from supervisor.policy.intent_policy import IntentPolicy
from planner.planner import PlannerNode
from models.schemas.supervisor_schemas import (
    DomainDecision,
    IntentDecision,
    PlannerDecision,
    PlannerTaskDecision,
    ReflectionDecision,
    RequestAnalysisDecision,
)
from prompts.runtime_prompts.system_prompt_registry import system_prompt_registry
from services.route_metrics_service import route_metrics_service
from services.agent_stream_bus import agent_stream_bus
from services.request_cancellation_service import request_cancellation_service
from common.utils.custom_logger import get_logger
from common.utils.date_utils import get_agent_date_context
from common.utils.history_compressor import (
    HISTORY_SUMMARY_BATCH_ROUNDS,
    HISTORY_SUMMARY_KEEP_RECENT_ROUNDS,
    HISTORY_SUMMARY_TRIGGER_ROUNDS,
    build_sliding_summary_messages,
    count_dialog_rounds,
)
from common.utils.retry_utils import GRAPH_RETRY_POLICY
from tools.runtime_tools.tool_registry import runtime_tool_registry

log = get_logger(__name__)
INTERRUPT_RESULT_TYPE = WORKFLOW_INTERRUPT_RESULT_TYPE


def _parse_json_from_text(text: str) -> dict:
    """内部转调独立支持模块。"""
    return _support_parse_json_from_text(text, log=log)


def _build_non_streaming_config(config: RunnableConfig) -> RunnableConfig:
    """内部转调独立支持模块。"""
    return _support_build_non_streaming_config(config)


def _invoke_with_timeout(callable_fn, *, timeout_sec: float, timeout_label: str):
    """内部转调独立支持模块。"""
    return _support_invoke_with_timeout(
        callable_fn,
        timeout_sec=timeout_sec,
        timeout_label=timeout_label,
    )


async def _ainvoke_with_timeout(awaitable, *, timeout_sec: float, timeout_label: str):
    """异步超时包装，供 async 节点统一使用。"""
    try:
        return await asyncio.wait_for(awaitable, timeout=max(0.1, float(timeout_sec)))
    except asyncio.TimeoutError as exc:
        raise TimeoutError(timeout_label) from exc


def _run_async_in_new_loop(awaitable):
    """在隔离事件循环中阻塞等待一个 awaitable 完成。"""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(awaitable)
    finally:
        loop.close()


def _normalize_interrupt_payload(val: Any) -> dict:
    """补齐默认审核文案和允许操作。"""
    return _support_normalize_interrupt_payload(
        val,
        default_message=DEFAULT_INTERRUPT_MESSAGE,
        default_allowed_decisions=list(DEFAULT_ALLOWED_DECISIONS),
    )


def _extract_interrupt_from_snapshot(snapshot: Any) -> Optional[dict]:
    """从快照里提取并规整 interrupt。"""
    return _support_extract_interrupt_from_snapshot(
        snapshot,
        default_message=DEFAULT_INTERRUPT_MESSAGE,
        default_allowed_decisions=list(DEFAULT_ALLOWED_DECISIONS),
    )


class WorkerState(TypedDict):
    """Worker节点状态"""
    task_id: str
    current_task: str
    session_id: str
    llm_config: Dict[str, Any]
    context_slots: Dict[str, Any]  # 会话结构化槽位
    context_summary: str  # 会话摘要
    messages: List[BaseMessage]  # 最近对话窗口


def _sanitize_error_message_text(raw_text: Any) -> str:
    """将底层异常规整为可对外展示的安全描述。"""
    detail = str(raw_text or "").strip()
    lowered = detail.lower()
    if not detail:
        return "服务暂时不可用。"
    if "timeout" in lowered or "超时" in detail:
        return "服务请求超时，请稍后再试。"
    if any(marker in lowered for marker in SUPERVISOR_AGENT_FAILURE_CONNECTION_MARKERS):
        return "外部服务连接异常，请稍后再试。"
    return "服务暂时不可用。"


def _build_tool_failure_message(agent_name: str) -> str:
    normalized_agent = str(agent_name or "").strip() or "unknown_agent"
    return f"工具 [{normalized_agent}] 暂时不可用，请稍后再试。"


def _exc_info_tuple(exc: Optional[BaseException]) -> Any:
    if exc is None:
        return None
    return (type(exc), exc, exc.__traceback__)


def _classify_agent_failure(exc: Exception) -> tuple[str, str]:
    """统一归类 Agent 执行失败，避免将内部异常直接暴露为正文。"""
    detail = str(exc or "").strip()
    lower = detail.lower()
    if isinstance(exc, TimeoutError) or any(marker in lower for marker in SUPERVISOR_AGENT_FAILURE_TIMEOUT_MARKERS):
        return "处理超时，请稍后重试。", detail
    if isinstance(exc, ConnectionError) or any(
            marker in lower for marker in SUPERVISOR_AGENT_FAILURE_CONNECTION_MARKERS
    ):
        return "模型服务连接异常，请稍后重试。", detail
    return "底层服务执行异常，请稍后重试。", detail


def _normalize_usage_payload(usage_payload: Any) -> Dict[str, int]:
    """将各模型返回的 token 用量结构归一成前端统一格式。"""

    def _to_int(value: Any) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    if not isinstance(usage_payload, dict):
        return {}

    input_tokens = _to_int(
        usage_payload.get("input_tokens")
        or usage_payload.get("prompt_tokens")
    )
    output_tokens = _to_int(
        usage_payload.get("output_tokens")
        or usage_payload.get("completion_tokens")
    )
    total_tokens = _to_int(
        usage_payload.get("total_tokens")
        or usage_payload.get("total")
    )
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens

    if input_tokens <= 0 and output_tokens <= 0 and total_tokens <= 0:
        return {}

    return {
        "input": input_tokens,
        "output": output_tokens,
        "total": total_tokens,
    }


def _extract_usage_from_response_metadata(response_metadata: Any) -> Dict[str, int]:
    """从 LLM response_metadata 中提取 token 用量。"""
    if not isinstance(response_metadata, dict):
        return {}

    for key in ("token_usage", "usage", "usage_metadata"):
        normalized = _normalize_usage_payload(response_metadata.get(key))
        if normalized:
            return normalized
    return {}


def _build_agent_error_payload(
        *,
        agent_name: str,
        exc: Exception,
        step_input: str = "",
) -> Dict[str, Any]:
    """构造稳定的结构化 Agent 错误载荷，避免 DB/网络故障拖垮链路。"""
    user_message, error_detail = _classify_agent_failure(exc)
    detail_text = str(error_detail or "").strip()
    lowered = detail_text.lower()
    error_type = "execution_error"
    retryable = False
    if isinstance(exc, TimeoutError) or "timeout" in lowered or "超时" in detail_text:
        error_type = "timeout_error"
        retryable = True
    elif isinstance(exc, ConnectionError) or any(
            marker in lowered for marker in SUPERVISOR_AGENT_FAILURE_CONNECTION_MARKERS):
        error_type = "connection_error"
        retryable = True

    return {
        "type": "agent_error",
        "agent": str(agent_name or "").strip() or "unknown_agent",
        "error_type": error_type,
        "message": user_message,
        "retryable": retryable,
        "safe_detail": _sanitize_error_message_text(detail_text),
    }


def _serialize_agent_error_payload(error_payload: Dict[str, Any]) -> str:
    try:
        return json.dumps(error_payload, ensure_ascii=False)
    except Exception:
        return str(error_payload)


def _compact_step_result_text(result: Any, *, max_chars: int = 360) -> str:
    """压缩写入 memory 的步骤结果，只保留对子任务复盘有用的核心结论。"""
    text = str(result or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""
    normalized = re.sub(r"\n{3,}", "\n\n", text)
    if len(normalized) <= max_chars:
        return normalized
    head = normalized[:max_chars].rstrip()
    cutoff = max(head.rfind("\n"), head.rfind("。"), head.rfind("；"), head.rfind(". "))
    compact = head[:cutoff].rstrip() if cutoff >= max_chars // 2 else head
    return f"{compact}..."


def _should_fail_fast_on_agent_error(
        error_detail: Any,
        *,
        agent_name: str = "",
) -> Optional[Dict[str, Any]]:
    """识别需要立即结束而非继续 replan 的结构化 Agent 错误。"""
    if not isinstance(error_detail, dict):
        return None

    if str(error_detail.get("type") or "").strip() != "agent_error":
        return None

    resolved_agent = str(error_detail.get("agent") or agent_name or "").strip()
    error_type = str(error_detail.get("error_type") or "").strip()
    retryable = bool(error_detail.get("retryable"))

    if (
            resolved_agent in {"sql_agent", "yunyou_agent"}
            or error_type in {"connection_error", "timeout_error"}
            or retryable
    ):
        return dict(error_detail)
    return None


def _looks_like_sql_request(text: str) -> bool:
    """快速识别可直接路由到 SQL Agent 的请求模式。"""
    t = (text or "").strip().lower()
    if not t:
        return False

    # Holter 语境下优先走 yunyou_agent，只有出现“显式本地 SQL 锚点”才判定为 sql_agent。
    if _looks_like_holter_request(t):
        return any(re.search(anchor, t) for anchor in SUPERVISOR_SQL_EXPLICIT_ANCHORS)

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
    company_info_markers = (
        "企业信息",
        "公司信息",
        "工商信息",
        "公开信息",
        "股东",
        "法人",
        "官网",
        "注册资本",
        "成立时间",
        "企业关系",
        "公司关系",
        "关联关系",
    )
    if any(marker in t for marker in company_info_markers):
        return True
    if any(entity in t for entity in ("公司", "企业")) and any(
        hint in t for hint in ("关系", "背景", "信息", "公开", "工商", "法人", "股东", "官网", "注册资本", "成立时间")
    ):
        return True
    keywords = SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.SEARCH_DOMAIN]
    if any(k in t for k in keywords):
        return True
    # 对“查一下/搜一下”这类泛词做降噪：若同时命中其他明确业务域，不视为 search。
    if any(h in t for h in SUPERVISOR_SEARCH_GENERIC_QUERY_HINTS):
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

    return any(re.search(pattern, t) for pattern in SUPERVISOR_WEATHER_ACTION_PATTERNS)


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
    return any(hint in t for hint in SUPERVISOR_SEARCH_ACTION_HINTS)


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

    if any(hint in t for hint in SUPERVISOR_CODE_ACTION_HINTS):
        return True

    if any(hint in t for hint in SUPERVISOR_CODE_LEARNING_HINTS):
        return False

    if any(re.search(pattern, t) for pattern in SUPERVISOR_CODE_SNIPPET_PATTERNS):
        return True

    followup_fix_hints = ("要的是", "改成", "换成", "不对", "错了", "main方法", "main method")
    if any(marker in t for marker in SUPERVISOR_CODE_LANGUAGE_ONLY_MARKERS) and any(
            hint in t for hint in followup_fix_hints):
        return True

    # 仅提到语言名但没有执行/排障动作时，视为泛问答而不是代码代理。
    if any(marker in t for marker in SUPERVISOR_CODE_LANGUAGE_ONLY_MARKERS):
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
            or (_looks_like_weather_request(t) and _is_weather_actionable_clause(t))
            or (_looks_like_search_request(t) and _is_search_actionable_clause(t))
            or _looks_like_medical_request(t)
            or _looks_like_code_request(t)
    )
    return not specialized_hits


def _build_graceful_chat_fallback(user_text: str) -> str:
    """为短句闲聊/情绪表达提供无模型兜底，避免直接抛错误气泡。"""
    normalized = (user_text or "").strip()
    if not normalized:
        return ""

    concern_tokens = ("你怎么了", "你还好吗", "怎么回事", "在吗", "有人吗")
    emotion_tokens = ("唉", "哎", "人生", "迷茫", "累", "烦", "难受", "郁闷", "崩溃", "绝望")

    if _looks_like_weather_request(normalized) and (not _is_weather_actionable_clause(normalized)):
        return "是啊，天气确实会影响心情。你要是想顺手查某个城市的实时气温、降雨或未来预报，直接告诉我城市就行。"
    if any(token in normalized for token in concern_tokens):
        return "我在，刚才响应有点慢，不是故意不回你。你想继续聊什么，直接说就行。"
    if any(token in normalized for token in emotion_tokens):
        return "我在。听起来你这会儿有点感慨或者有点累，如果你愿意，可以继续说说发生了什么，我陪你理一理。"
    return ""


# --- Helpers ---
def _latest_human_message(messages: List[BaseMessage]) -> str:
    """读取最近一条用户消息文本。"""
    return _support_latest_human_message(messages)


def _content_to_text(content: Any) -> str:
    """统一规整消息内容文本。"""
    return _support_content_to_text(content)


def _history_requests_location(messages: Optional[List[BaseMessage]]) -> bool:
    """判断最近是否追问过城市。"""
    return _support_history_requests_location(messages)


def _is_followup_supplement(text: str, messages: Optional[List[BaseMessage]] = None) -> bool:
    """识别是否为补充说明输入。"""
    return _support_is_followup_supplement(text, messages)


def _looks_like_location_fragment(text: str) -> bool:
    """识别短地点片段。"""
    return _support_looks_like_location_fragment(text)


def _extract_recent_city_from_history(messages: List[BaseMessage]) -> Optional[str]:
    """从历史消息推断城市。"""
    return _support_extract_recent_city_from_history(messages)


def _extract_city_from_context_slots(context_slots: Optional[Dict[str, Any]]) -> Optional[str]:
    """从结构化槽位里读取城市。"""
    return _support_extract_city_from_context_slots(context_slots)


def _input_has_location_anchor(text: str) -> bool:
    """判断输入是否已带地点锚点。"""
    return _support_input_has_location_anchor(text)


def _history_hint_intent(messages: List[BaseMessage], latest_user_text: str = "") -> Optional[str]:
    """根据历史上下文提示意图。"""
    return _support_history_hint_intent(messages, latest_user_text)


def _has_recent_weather_fact(messages: List[BaseMessage]) -> bool:
    """判断是否已有可复用天气事实。"""
    return _support_has_recent_weather_fact(messages)


def _looks_like_weather_reuse_query(text: str) -> bool:
    """识别天气建议类追问。"""
    return _support_looks_like_weather_reuse_query(text)


def _wants_weather_refresh(text: str) -> bool:
    """识别重新查询天气的要求。"""
    return _support_wants_weather_refresh(text)


def _can_reuse_weather_context(messages: List[BaseMessage], latest_user_text: str) -> bool:
    """判断天气上下文是否可以直接复用。"""
    return _support_can_reuse_weather_context(messages, latest_user_text)


def _collect_intent_signals(text: str) -> List[str]:
    """统一收集当前输入命中的意图候选。"""
    return _support_collect_intent_signals(
        text,
        looks_like_holter_request=_looks_like_holter_request,
        looks_like_sql_request=_looks_like_sql_request,
        looks_like_weather_request=_looks_like_weather_request,
        looks_like_search_request=_looks_like_search_request,
        looks_like_medical_request=_looks_like_medical_request,
        looks_like_code_request=_looks_like_code_request,
        is_weather_actionable_clause=_is_weather_actionable_clause,
        is_search_actionable_clause=_is_search_actionable_clause,
        multi_domain_agent_priority=MULTI_DOMAIN_AGENT_PRIORITY,
    )


def _dedupe_keep_order(values: List[str]) -> List[str]:
    """按出现顺序去重。"""
    return _support_dedupe_keep_order(values)


def _has_dependency_hint(text: str) -> bool:
    """识别输入里的先后依赖提示。"""
    return _support_has_dependency_hint(text)


def _is_explicit_request_clause(text: str) -> bool:
    """判断子句是否为显式执行请求。"""
    return _support_is_explicit_request_clause(
        text,
        request_action_hints=SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.REQUEST_ACTION_HINT],
    )


def _analyze_request(user_text: str) -> RequestAnalysisDecision:
    """统一分析候选意图与路由策略。"""
    payload = _support_analyze_request_payload(
        user_text,
        split_query_clauses_fn=_split_query_clauses,
        collect_intent_signals_fn=_collect_intent_signals,
        is_explicit_request_clause_fn=_is_explicit_request_clause,
        is_weather_actionable_clause_fn=_is_weather_actionable_clause,
        is_search_actionable_clause_fn=_is_search_actionable_clause,
        has_dependency_hint_fn=_has_dependency_hint,
        dedupe_keep_order_fn=_dedupe_keep_order,
        agent_domain_map=AGENT_DOMAIN_MAP,
        route_strategy_single=RouteStrategy.SINGLE_DOMAIN.value,
        route_strategy_complex_single=RouteStrategy.COMPLEX_SINGLE_DOMAIN.value,
        route_strategy_multi_split=RouteStrategy.MULTI_DOMAIN_SPLIT.value,
    )
    if not payload:
        return RequestAnalysisDecision()
    return RequestAnalysisDecision(**payload)


def _split_query_clauses(user_text: str) -> List[str]:
    """把复合输入切成子句。"""
    return _support_split_query_clauses(user_text)


def _select_primary_agent_for_clause(clause_text: str, fallback_candidates: List[str]) -> Optional[str]:
    """为单子句挑选优先 Agent。"""
    return _support_select_primary_agent_for_clause(
        clause_text,
        fallback_candidates,
        collect_intent_signals_fn=_collect_intent_signals,
        multi_domain_agent_priority=MULTI_DOMAIN_AGENT_PRIORITY,
    )


    return _support_select_primary_agent_for_clause(
        clause_text,
        fallback_candidates,
        collect_intent_signals_fn=_collect_intent_signals,
        multi_domain_agent_priority=MULTI_DOMAIN_AGENT_PRIORITY,
    )


def _looks_like_compound_request(text: str) -> bool:
    """判断当前输入是否更适合走复合任务。"""
    return _support_looks_like_compound_request(
        text,
        analyze_request_fn=_analyze_request,
        route_strategy_complex_single=RouteStrategy.COMPLEX_SINGLE_DOMAIN.value,
        route_strategy_multi_split=RouteStrategy.MULTI_DOMAIN_SPLIT.value,
        complex_connector_hints=SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.COMPLEX_CONNECTOR_HINT],
    )


def _worker_history_message_text(msg: BaseMessage) -> str:
    try:
        return _content_to_text(msg.content).strip().lower()
    except Exception:
        content_val = getattr(msg, "content", "")
        return str(content_val or "").strip().lower()


def _is_agent_history_relevant(agent_name: str, text: str) -> bool:
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
    return True


def _build_worker_history_messages_for_agent(
        *,
        agent_name: str,
        history_messages: List[BaseMessage],
        limit: int,
        llm_config: Optional[Dict[str, Any]] = None,
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

    selected: List[BaseMessage] = []
    latest_msg = human_messages[-1]
    latest_text = _worker_history_message_text(latest_msg)
    if agent_name in {"CHAT", "chat_node", ""} or _is_agent_history_relevant(agent_name, latest_text):
        selected.append(latest_msg)

    for msg in reversed(human_messages[:-1]):
        if len(selected) >= safe_limit:
            break
        if _is_agent_history_relevant(agent_name, _worker_history_message_text(msg)):
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

    skill_prompt_blocks = [str(item or "").strip() for item in ((llm_config or {}).get("skill_prompt_blocks") or []) if str(item or "").strip()]
    if skill_prompt_blocks:
        selected = [SystemMessage(content="\n\n".join(skill_prompt_blocks))] + selected

    return selected


def _build_runtime_prefix_messages(
    *,
    llm_config: Optional[Dict[str, Any]] = None,
    dynamic_blocks: Optional[List[str]] = None,
    extra_static_blocks: Optional[List[str]] = None,
) -> List[BaseMessage]:
    config = llm_config or {}
    return list(system_prompt_registry.build_runtime_messages(
        model_config=config,
        tool_catalog=config.get("resolved_tool_catalog") or [],
        skill_prompts=config.get("skill_prompt_blocks") or [],
        skill_names=config.get("selected_skill_names") or [],
        dynamic_blocks=dynamic_blocks or [],
        extra_static_blocks=extra_static_blocks or [],
    ))


async def _invoke_structured_output_with_fallback(
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

    # 判断当前模型是否支持结构化输出
    if hasattr(llm, "with_structured_output"):
        for kwargs in ({"method": "json_mode"}, {}):
            try:
                structured_model = llm.with_structured_output(schema, **kwargs)
                result = await _ainvoke_with_timeout(
                    (prompt | structured_model).ainvoke(inputs, config=runtime_config),
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

    # 如果当前模型不支持结构化输出则回退到传统 JSON 解析
    response = await _ainvoke_with_timeout(
        (prompt | llm).ainvoke(inputs, config=runtime_config),
        timeout_sec=invoke_timeout_sec,
        timeout_label=f"{log_name}.json_fallback",
    )
    data = _parse_json_from_text(_content_to_text(getattr(response, "content", response)))
    try:
        return schema(**data)
    except Exception as exc:
        if last_structured_exc is not None:
            log.warning(
                f"[{log_name}]: 结构化输出失败，JSON 回退仍失败。structured_error={last_structured_exc}, json_error={exc}")
        raise


def _run_agent_to_completion(
        agent_name: str,
        user_input: str,
        model: BaseChatModel,
        config: RunnableConfig,
        session_id: str = "",
        history_messages: Optional[List[BaseMessage]] = None,
        context_slots: Optional[Dict[str, Any]] = None,
        context_summary: str = "",
        llm_config: Optional[Dict[str, Any]] = None,
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
        fallback_messages: List[Any] = _build_runtime_prefix_messages(
            llm_config=llm_config,
            dynamic_blocks=[get_agent_date_context(), context_summary],
            extra_static_blocks=[ChatFallbackPrompt.get_system_prompt()],
        )
        fallback_messages.append(HumanMessage(content=effective_user_input))
        # 降级为通用 CHAT
        with request_cancellation_service.bind_request(request_id):
            response = _run_async_in_new_loop(model.ainvoke(fallback_messages, config=config))
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        log.info(f"⏱️ Agent[CHAT_FALLBACK] 耗时: {elapsed_ms}ms")
        return {
            "content": _content_to_text(getattr(response, "content", "")).strip(),
            "response_metadata": dict(getattr(response, "response_metadata", {}) or {}),
        }

    req = AgentRequest(
        user_input=effective_user_input, model=model,
        session_id=session_id or config.get("configurable", {}).get("thread_id", ""),
        subgraph_id=agent_name, llm_config=llm_config or {},
        state={
            "messages": history_messages or [],
            "context_slots": context_slots or {},
            "context_summary": context_summary or "",
            "current_task": effective_user_input,
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
        return {
            "content": normalized or "已完成处理，但未生成可展示文本。请重试或更换提问方式。",
            "response_metadata": dict(getattr(final_response, "response_metadata", {}) or {}),
        }

    if request_id and request_cancellation_service.is_cancelled(request_id):
        log.info(f"Agent[{agent_name}] 结束前检测到取消信号，返回取消结果。request_id={request_id}")
        return WORKER_CANCELLED_RESULT

    # 如果没有最终 response，检查是否因为中断挂起
    if agent_error is None:
        root_thread_id = session_id or config.get("configurable", {}).get("thread_id", "")
        if root_thread_id:
            subgraph_config = {"configurable": {"thread_id": f"{root_thread_id}_{agent_name}"}}
            snapshot = _run_async_in_new_loop(agent_instance.graph.aget_state(subgraph_config))
            snapshot_interrupt = _extract_interrupt_from_snapshot(snapshot)
            if snapshot_interrupt:
                snapshot_interrupt["agent_name"] = snapshot_interrupt.get("agent_name") or agent_name
                elapsed_ms = int((time.perf_counter() - started_at) * 1000)
                log.info(f"⏱️ Agent[{agent_name}] 快照中断挂起，耗时: {elapsed_ms}ms")
                return {"type": INTERRUPT_RESULT_TYPE, "payload": snapshot_interrupt}

    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    log.warning(f"⏱️ Agent[{agent_name}] 失败，耗时: {elapsed_ms}ms")
    raise RuntimeError(f"{agent_name} 执行失败: {agent_error}")


def _task_sort_key(item: tuple[str, Any]) -> tuple[int, str]:
    """按任务编号排序（t1/t2...），保证输出顺序稳定。"""
    task_id = str(item[0])
    match = re.search(r"\d+", task_id)
    seq = int(match.group(0)) if match else 10 ** 9
    return seq, task_id


def _build_chat_retry_messages(
    prompt: str,
    recent_messages: List[Any],
    *,
    llm_config: Optional[Dict[str, Any]] = None,
    context_summary: str = "",
) -> list[Any]:
    compact_window = max(1, min(4, AGENT_LOOP_CONFIG.context_history_messages))
    compact_history = recent_messages[-compact_window:]
    return _build_runtime_prefix_messages(
        llm_config=llm_config,
        dynamic_blocks=[get_agent_date_context(), context_summary],
        extra_static_blocks=[prompt],
    ) + compact_history


def _build_failure_summary_prompt(
        *,
        current_task: str,
        current_agent: str,
        reason: str,
) -> str:
    normalized_task = str(current_task or "").strip() or "当前工具任务"
    normalized_agent = str(current_agent or "").strip() or "unknown_agent"
    return (
        f"内部诊断：在执行 [{normalized_task}] 时，工具 [{normalized_agent}] 暂时不可用。\n"
        f"原因概括：{_sanitize_error_message_text(reason)}\n\n"
        "请作为 AI 助手，用极简、委婉的语言向用户道歉，并说明该服务暂时不可用。"
        "绝对不要复述任何底层网络、数据库、端口、堆栈或代码错误信息。"
    )


# ==================== Tier-0.5: 数据域路由器 (Domain Router) ====================
# ==================== Tier-1: 意图路由器 (Intent Router) ====================
# ==================== Tier-2: Planner -> Dispatch -> Send(worker) -> Reduce ====================
def dispatch_node(state: GraphState, config: RunnableConfig) -> dict:
    """串行弹夹调度：只弹出下一任务，不再在此节点结算上一轮执行结果。"""
    plan = [str(item or "").strip() for item in list(state.get("plan") or []) if str(item or "").strip()]
    task_list = [dict(task) for task in list(state.get("task_list") or [])]
    task_results = dict(state.get("task_results") or {})

    next_task = ""
    next_task_id = ""
    next_task_agent = ""
    if plan:
        next_task = plan.pop(0)
        for task in task_list:
            if str(task.get("status") or "").strip() == TaskStatus.PENDING.value:
                next_task_id = str(task.get("id") or "").strip()
                task["status"] = TaskStatus.DISPATCHED.value
                next_task_agent = str(task.get("agent") or "").strip() or _resolve_forced_specialist_agent(next_task)
                if next_task_agent:
                    task["agent"] = next_task_agent
                break

    if next_task_id:
        log.info(f"Dispatch: next task [{next_task_id}] -> {next_task} (agent={next_task_agent or 'pending_route'})")
    else:
        log.info("Dispatch: no remaining tasks, main loop will end.")
        next_task = "END_TASK"

    return {
        "plan": plan,
        "task_list": task_list,
        "task_results": task_results,
        "active_tasks": [],
        "worker_results": [],
        "current_task": next_task,
        "current_task_id": next_task_id,
        "current_step_input": "" if next_task == "END_TASK" else next_task,
        "current_step_agent": "" if next_task == "END_TASK" else next_task_agent,
        "interrupt_payload": None,
        "last_step_result": "",
        "last_step_status": "",
        "last_step_error": "",
        "replan_reason": "",
        "error_message": None,
        "error_detail": None,
        "intent": None,
        "intent_confidence": None,
        "is_complex": False,
        "direct_answer": "",
    }


def _is_replan_worthy_failure(result_text: str, error_message: str) -> bool:
    """识别需要回退到 planner 的失败结果。"""
    normalized_error = str(error_message or "").strip()
    normalized_result = str(result_text or "").strip()
    if normalized_error:
        return True
    if not normalized_result:
        return True
    lower_text = normalized_result.lower()
    if normalized_result in {WORKER_CANCELLED_RESULT, WORKER_PENDING_APPROVAL_RESULT}:
        return True
    failure_markers = (
        "执行失败",
        "error:",
        "exception",
        "timed out",
        "超时",
        "连接异常",
        "服务暂时不可用",
        "未生成可展示",
    )
    return any(marker in lower_text for marker in failure_markers)


def _append_step_result_entry(
    memory: Dict[str, Any],
    *,
    step: str,
    agent: str,
    result: str,
    status: str,
) -> Dict[str, Any]:
    """把最近一步的执行结果沉淀到 memory.step_results。"""
    updated_memory = dict(memory or {})
    step_results = list(updated_memory.get("step_results") or [])
    entry = {
        "step": str(step or "").strip(),
        "agent": str(agent or "").strip(),
        "result": _compact_step_result_text(result),
        "status": str(status or "").strip(),
    }
    if step_results and isinstance(step_results[-1], dict):
        last_step = str(step_results[-1].get("step") or "").strip()
        if last_step == entry["step"]:
            step_results[-1] = entry
        else:
            step_results.append(entry)
    else:
        step_results.append(entry)
    updated_memory["step_results"] = step_results
    return updated_memory


def _normalize_execution_result(result: Any) -> Dict[str, Any]:
    """把 Agent 执行返回统一规整为 content/response_metadata 结构。"""
    if isinstance(result, dict):
        return {
            "content": str(result.get("content") or "").strip(),
            "response_metadata": dict(result.get("response_metadata") or {}),
        }
    return {"content": str(result or "").strip(), "response_metadata": {}}


def evaluator_node(state: GraphState) -> dict:
    """统一结算当前步骤结果，并决定继续 dispatch 还是回退 re-plan。"""
    current_task_id = str(state.get("current_task_id") or "").strip()
    current_task = str(state.get("current_task") or state.get("current_step_input") or "").strip()
    current_agent = str(state.get("current_step_agent") or state.get("intent") or "").strip()
    interrupt_payload = state.get("interrupt_payload")
    error_message = str(state.get("error_message") or "").strip()
    error_detail = state.get("error_detail")
    result_text = _extract_last_step_result(state)
    task_list = [dict(task) for task in list(state.get("task_list") or [])]
    task_results = dict(state.get("task_results") or {})
    memory = dict(state.get("memory") or {})
    current_replan_count = int(state.get("replan_count") or 0)
    max_replans = int(state.get("max_replans") or WORKFLOW_MAX_REPLANS)

    if not current_task_id:
        return {
            "last_step_result": "",
            "last_step_status": "",
            "last_step_error": "",
            "replan_reason": "",
        }

    for task in task_list:
        if str(task.get("id") or "") != current_task_id:
            continue
        if current_agent:
            task["agent"] = current_agent
        if interrupt_payload:
            task["status"] = TaskStatus.PENDING_APPROVAL.value
            task["result"] = WORKER_PENDING_APPROVAL_RESULT
            memory = _append_step_result_entry(
                memory,
                step=current_task,
                agent=current_agent,
                result=WORKER_PENDING_APPROVAL_RESULT,
                status=TaskStatus.PENDING_APPROVAL.value,
            )
            return {
                "task_list": task_list,
                "task_results": task_results,
                "memory": memory,
                "last_step_result": WORKER_PENDING_APPROVAL_RESULT,
                "last_step_status": TaskStatus.PENDING_APPROVAL.value,
                "last_step_error": "",
                "replan_reason": "",
            }

        fail_fast_error = _should_fail_fast_on_agent_error(error_detail, agent_name=current_agent)
        if fail_fast_error:
            safe_result = "由于网络或服务原因，该任务执行失败。"
            task["status"] = TaskStatus.ERROR.value
            task["result"] = safe_result
            task_results[current_task_id] = safe_result
            memory = _append_step_result_entry(
                memory,
                step=current_task,
                agent=str(fail_fast_error.get("agent") or current_agent or "").strip(),
                result=safe_result,
                status=TaskStatus.ERROR.value,
            )
            return {
                "task_list": task_list,
                "task_results": task_results,
                "memory": memory,
                "last_step_result": safe_result,
                "last_step_status": TaskStatus.ERROR.value,
                "last_step_error": "服务调用失败",
                "replan_reason": "",
                "error_message": None,
                "error_detail": None,
                "current_step_agent": str(fail_fast_error.get("agent") or current_agent or "").strip(),
            }

        if _is_replan_worthy_failure(result_text, error_message):
            safe_result = "服务不可用"
            task["status"] = TaskStatus.ERROR.value
            task["result"] = safe_result
            task_results[current_task_id] = safe_result
            memory = _append_step_result_entry(
                memory,
                step=current_task,
                agent=current_agent,
                result=safe_result,
                status=TaskStatus.ERROR.value,
            )
            return {
                "task_list": task_list,
                "task_results": task_results,
                "memory": memory,
                "last_step_result": safe_result,
                "last_step_status": TaskStatus.ERROR.value,
                "last_step_error": "服务调用失败",
                "replan_reason": "",
                "replan_count": current_replan_count,
                "max_replans": max_replans,
                "error_message": None,
                "error_detail": None,
                "current_step_agent": current_agent,
            }

        task["status"] = TaskStatus.DONE.value
        task["result"] = result_text
        task_results[current_task_id] = result_text
        memory = _append_step_result_entry(
            memory,
            step=current_task,
            agent=current_agent,
            result=result_text,
            status=TaskStatus.DONE.value,
        )
        return {
            "task_list": task_list,
            "task_results": task_results,
            "memory": memory,
            "last_step_result": result_text,
            "last_step_status": TaskStatus.DONE.value,
            "last_step_error": "",
            "replan_reason": "",
            "error_message": None,
            "error_detail": None,
        }

    return {}


# ==================== Worker & Reducer ====================
async def worker_node(
        state: WorkerState,
        config: RunnableConfig,
        model: BaseChatModel,
        router_model: BaseChatModel,
        chat_model: BaseChatModel,
) -> dict:
    """Worker 节点：对单个 current_task 执行快路由，再调用具体子 Agent。"""
    started_at = time.perf_counter()
    task_id = str(state.get("task_id") or "").strip() or "t0"
    current_task = str(state.get("current_task") or "").strip()
    log.info(f"Worker start: task=[{task_id}] current_task=[{current_task}]")
    request_id = str(
        config.get("configurable", {}).get("run_id")
        or config.get("configurable", {}).get("thread_id")
        or state.get("session_id")
        or ""
    ).strip()

    if request_id and request_cancellation_service.is_cancelled(request_id):
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return {
            "worker_results": [
                WorkerResult(
                    task_id=task_id,
                    task=current_task,
                    result=WORKER_CANCELLED_RESULT,
                    error=None,
                    error_payload=None,
                    agent=None,
                    elapsed_ms=elapsed_ms,
                    usage=None,
                )
            ]
        }

    # 这里只保留已有上下文，不把 current_task 预先塞进 history。
    # 下游 Agent/chat_node 都会单独拿到 current_task/user_input；
    # 如果这里提前追加，会把同一个子任务喂两遍，导致模型出现复读式回答。
    worker_messages = list(state.get("messages", []) or [])

    step_state: GraphState = {
        "messages": worker_messages,
        "session_id": state.get("session_id") or "",
        "llm_config": state.get("llm_config") or {},
        "context_slots": state.get("context_slots") or {},
        "context_summary": state.get("context_summary") or "",
        "current_task": current_task,
        "current_step_input": current_task,
        "current_step_agent": None,
        "direct_answer": "",
        "error_message": None,
        "error_detail": None,
    }

    domain_update = await IntentPolicy.domain_router_node(step_state, model=router_model, config=config)
    step_state.update(domain_update)
    intent_update = await IntentPolicy.intent_router_node(step_state, model=router_model, config=config)
    step_state.update(intent_update)

    route_target = _resolve_executor_route(step_state)
    if step_state.get("is_complex"):
        log.warning(f"Worker task still marked complex, forcing single execution: task={task_id}, route={route_target}")

    try:
        if route_target in MEMBERS:
            history_messages = _build_worker_history_messages_for_agent(
                agent_name=route_target,
                history_messages=worker_messages,
                limit=AGENT_LOOP_CONFIG.context_history_messages,
                llm_config=state.get("llm_config") or {},
            )
            if not runtime_tool_registry.is_agent_enabled(route_target, state.get("llm_config") or {}):
                disabled_message = runtime_tool_registry.get_disabled_agent_message(route_target)
                elapsed_ms = int((time.perf_counter() - started_at) * 1000)
                return {
                    "worker_results": [
                        WorkerResult(
                            task_id=task_id,
                            task=current_task,
                            result=disabled_message,
                            error=None,
                            error_payload=None,
                            agent=route_target,
                            elapsed_ms=elapsed_ms,
                            usage=None,
                        )
                    ]
                }
            execution_result = await asyncio.to_thread(
                _run_agent_to_completion,
                route_target,
                current_task,
                model,
                config,
                state.get("session_id") or "",
                history_messages,
                state.get("context_slots") or {},
                state.get("context_summary") or "",
                state.get("llm_config") or {},
            )
            agent_name = route_target
        else:
            chat_result = await chat_node(step_state, model=chat_model, config=config)
            if chat_result.get("error_message"):
                error_detail = chat_result.get("error_detail")
                if isinstance(error_detail, dict):
                    raise RuntimeError(_serialize_agent_error_payload(error_detail))
                raise RuntimeError(str(error_detail or chat_result.get("error_message")))
            chat_messages = chat_result.get("messages") or []
            chat_message = chat_messages[-1] if chat_messages else AIMessage(content="")
            execution_result = {
                "content": _content_to_text(getattr(chat_message, "content", "")).strip(),
                "response_metadata": dict(getattr(chat_message, "response_metadata", {}) or {}),
            }
            agent_name = "chat_node"

        if isinstance(execution_result, dict) and execution_result.get("type") == INTERRUPT_RESULT_TYPE:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            payload = _normalize_interrupt_payload(execution_result.get("payload"))
            payload["agent_name"] = payload.get("agent_name") or agent_name
            return {
                "worker_results": [
                    WorkerResult(
                        task_id=task_id,
                        task=current_task,
                        result=WORKER_PENDING_APPROVAL_RESULT,
                        error=None,
                        error_payload=None,
                        agent=agent_name,
                        elapsed_ms=elapsed_ms,
                        usage=None,
                    )
                ],
                "interrupt_payload": payload,
            }

        normalized_execution = _normalize_execution_result(execution_result)
        res_text = normalized_execution["content"]
        usage = _extract_usage_from_response_metadata(normalized_execution["response_metadata"]) or None

        if res_text == WORKER_CANCELLED_RESULT:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            return {
                "worker_results": [
                    WorkerResult(
                        task_id=task_id,
                        task=current_task,
                        result=WORKER_CANCELLED_RESULT,
                        error=None,
                        error_payload=None,
                        agent=agent_name,
                        elapsed_ms=elapsed_ms,
                        usage=usage,
                    )
                ]
            }
    except Exception as exc:
        err_msg = str(exc)
        if "Interrupt(" in err_msg or exc.__class__.__name__ == "GraphInterrupt":
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            return {
                "worker_results": [
                    WorkerResult(
                        task_id=task_id,
                        task=current_task,
                        result=WORKER_PENDING_APPROVAL_RESULT,
                        error=None,
                        error_payload=None,
                        agent=route_target,
                        elapsed_ms=elapsed_ms,
                        usage=None,
                    )
                ],
                "interrupt_payload": {
                    "message": DEFAULT_INTERRUPT_MESSAGE,
                    "allowed_decisions": list(DEFAULT_ALLOWED_DECISIONS),
                    "action_requests": [],
                    "agent_name": route_target,
                },
            }
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        error_payload = _build_agent_error_payload(
            agent_name=route_target or "worker_node",
            exc=exc,
            step_input=current_task,
        )
        log.error(f"Worker [{task_id}] error: {exc}")
        return {
            "worker_results": [
                WorkerResult(
                    task_id=task_id,
                    task=current_task,
                    result=_serialize_agent_error_payload(error_payload),
                    error=error_payload["message"],
                    error_payload=error_payload,
                    agent=route_target,
                    elapsed_ms=elapsed_ms,
                    usage=None,
                )
            ]
        }

    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    return {
        "worker_results": [
            WorkerResult(
                task_id=task_id,
                task=current_task,
                result=str(res_text or "").strip(),
                error=None,
                error_payload=None,
                agent=agent_name,
                elapsed_ms=elapsed_ms,
                usage=usage,
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
            worker_res = matched[0]  # 取出其中一个
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


def _stringify_task_result(value: Any) -> str:
    """把任务结果统一转成可判断的字符串。"""
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False, indent=2).strip()
    except Exception:
        return str(value).strip()


def _should_skip_reflection_after_parallel_success(
        tasks: List[SubTask],
        task_results: Dict[str, Any],
        *,
        planner_source: str,
) -> bool:
    """并行结果已充分时，直接进入聚合，避免额外反思超时。"""
    if len(tasks) < 2:
        return False

    if planner_source not in {
        "rule_split",
        "deterministic_fallback",
        "single_domain_guard",
        "fallback_deterministic",
    }:
        return False

    for task in tasks:
        if str(task.get("status") or "") != TaskStatus.DONE.value:
            return False
        task_id = str(task.get("id") or "")
        result_text = _stringify_task_result(task_results.get(task_id, task.get("result")))
        if (not result_text) or result_text in {WORKER_CANCELLED_RESULT, WORKER_PENDING_APPROVAL_RESULT}:
            return False
        if result_text.lower().startswith("error:"):
            return False

    return True


def _sanitize_reflection_tasks(
        reflection_tasks: List[PlannerTaskDecision],
        *,
        existing_tasks: List[SubTask],
        next_task_sequence: int,
) -> tuple[List[SubTask], int]:
    """清洗自动反思生成的新增任务，确保编号、依赖和去重可用。"""
    existing_ids = {str(task.get("id") or "") for task in existing_tasks if task.get("id")}
    existing_signatures = {
        (str(task.get("agent") or "CHAT").strip(), str(task.get("input") or "").strip())
        for task in existing_tasks
        if str(task.get("input") or "").strip()
    }
    valid_dependency_ids = set(existing_ids)
    remapped_ids: Dict[str, str] = {}
    appended_tasks: List[SubTask] = []
    sequence = max(1, int(next_task_sequence or (len(existing_tasks) + 1)))

    for candidate in list(reflection_tasks or [])[:3]:
        agent_name = str(getattr(candidate, "agent", "") or "CHAT").strip() or "CHAT"
        if agent_name not in MEMBERS and agent_name != "CHAT":
            agent_name = "CHAT"
        input_text = str(getattr(candidate, "input", "") or "").strip()
        if not input_text:
            continue
        signature = (agent_name, input_text)
        if signature in existing_signatures:
            continue

        original_id = str(getattr(candidate, "id", "") or f"r{sequence}")
        assigned_id = f"t{sequence}"
        sequence += 1
        remapped_ids[original_id] = assigned_id

        raw_dependencies = [str(dep) for dep in (getattr(candidate, "depends_on", []) or []) if str(dep).strip()]
        normalized_dependencies: List[str] = []
        for dep_id in raw_dependencies:
            mapped_dep = remapped_ids.get(dep_id, dep_id)
            if mapped_dep in valid_dependency_ids and mapped_dep not in normalized_dependencies:
                normalized_dependencies.append(mapped_dep)

        appended_tasks.append(
            {
                "id": assigned_id,
                "agent": agent_name,
                "input": input_text,
                "depends_on": normalized_dependencies,
                "status": TaskStatus.PENDING.value,
                "result": None,
            }
        )
        existing_signatures.add(signature)
        valid_dependency_ids.add(assigned_id)

    return appended_tasks, sequence


async def reflection_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
    """执行后反思：判断当前结果是否需要自动追加下一轮任务。"""
    tasks = list(state.get("task_list") or [])
    if not tasks:
        return {}

    status_values = [str(task.get("status") or "") for task in tasks]
    if any(status in PENDING_TASK_STATUSES for status in status_values):
        return {
            "reflection_source": "skipped_pending",
            "reflection_summary": "仍有未收敛任务，暂不触发自动反思。",
        }

    if not WORKFLOW_REFLECTION_CONFIG.enabled:
        return {
            "reflection_source": "disabled",
            "reflection_summary": "自动反思已关闭，直接进入结果收敛。",
        }

    current_round = max(0, int(state.get("reflection_round") or 0))
    max_rounds = max(0, int(state.get("max_reflection_rounds") or WORKFLOW_REFLECTION_CONFIG.max_rounds))
    if current_round >= max_rounds:
        return {
            "reflection_source": "limit_reached",
            "reflection_summary": "已达到自动反思轮次上限，停止追加任务。",
        }

    task_results = dict(state.get("task_results", {}) or {})
    planner_source = str(state.get("planner_source") or "").strip()
    if _should_skip_reflection_after_parallel_success(
            tasks,
            task_results,
            planner_source=planner_source,
    ):
        return {
            "reflection_source": "parallel_converged",
            "reflection_summary": "并行子任务已全部完成且结果充分，直接进入最终汇总。",
        }

    if (not WORKFLOW_REFLECTION_CONFIG.llm_enabled) or model is None:
        return {
            "reflection_source": "llm_disabled",
            "reflection_summary": "未启用 LLM 反思器，当前任务直接收敛。",
        }

    if not task_results:
        return {
            "reflection_source": "empty_results",
            "reflection_summary": "暂无可用执行结果，直接收敛。",
        }

    preview_max_chars = int(WORKFLOW_REFLECTION_CONFIG.result_preview_max_chars)
    task_lines: List[str] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        if not task_id:
            continue
        result_preview = _normalize_aggregator_result_text(
            task_id,
            task_results.get(task_id, task.get("result")),
            preview_max_chars,
        )
        task_lines.append(
            "\n".join(
                [
                    f"任务ID: {task_id}",
                    f"Agent: {task.get('agent')}",
                    f"状态: {task.get('status')}",
                    f"输入: {str(task.get('input') or '')[:200]}",
                    f"结果: {result_preview}",
                ]
            )
        )

    latest_user_text = _latest_human_message(state.get("messages", []))
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ReflectionPrompt.get_system_prompt()),
            (
                "human",
                "用户原始请求：\n{user_request}\n\n"
                "当前已经执行完成的任务如下：\n{task_summaries}\n\n"
                "请判断：是否还需要自动追加下一轮任务，还是已经足够收敛。",
            ),
        ]
    )

    try:
        structured = await _invoke_structured_output_with_fallback(
            prompt=prompt,
            model=model,
            schema=ReflectionDecision,
            inputs={
                "user_request": latest_user_text,
                "task_summaries": "\n\n---\n\n".join(task_lines),
            },
            config=config,
            max_tokens=256,
            log_name="workflow_reflection",
        )
    except Exception as exc:
        log.warning(f"Reflection fallback: {exc}")
        return {
            "reflection_source": "llm_error",
            "reflection_summary": f"自动反思失败，直接收敛。原因: {exc}",
        }

    next_task_sequence = int(state.get("next_task_sequence") or (len(tasks) + 1))
    appended_tasks, next_task_sequence = _sanitize_reflection_tasks(
        list(structured.tasks or []),
        existing_tasks=tasks,
        next_task_sequence=next_task_sequence,
    )
    reflection_summary = str(structured.summary or "").strip() or "自动反思判断当前阶段已收敛。"
    if (not structured.continue_execution) or (not appended_tasks):
        return {
            "reflection_round": current_round + 1,
            "reflection_source": "llm_converged",
            "reflection_summary": reflection_summary,
            "next_task_sequence": next_task_sequence,
        }

    updated_tasks = tasks + appended_tasks
    updated_max_waves = max(
        int(state.get("max_waves") or 0),
        int(state.get("current_wave") or 0) + len(appended_tasks) * 2 + 2,
    )
    return {
        "task_list": updated_tasks,
        "max_waves": updated_max_waves,
        "reflection_round": current_round + 1,
        "reflection_source": "llm_appended",
        "reflection_summary": reflection_summary,
        "next_task_sequence": next_task_sequence,
    }


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


def _humanize_aggregator_error_text(text: str) -> str:
    """将内部错误文本转成更适合直出给用户的表述。"""
    normalized = (text or "").strip()
    if normalized.lower().startswith("error:"):
        normalized = normalized[6:].strip()

    normalized = re.sub(r"^[a-z_]+\s+执行失败[:：]\s*", "", normalized, flags=re.I)
    normalized = normalized.replace(
        "StructuredTool does not support sync invocation.",
        "联网检索工具调用异常，请稍后重试。",
    )
    normalized = normalized.replace("chatprompttemplate is missing variables", "提示词模板变量配置异常")
    normalized = normalized.strip(" 。\n\t")
    if not normalized:
        return "有一项子任务执行失败，请稍后重试。"
    if "联网检索" in normalized or "搜索" in normalized:
        return normalized
    return f"有一项子任务未能完成：{normalized}。"


def _clean_aggregated_task_text(task: Optional[SubTask], text: str) -> str:
    """清洗子任务结果中的内部噪音，保留用户可读内容。"""
    normalized = str(text or "").replace("\r\n", "\n").strip()
    if not normalized:
        return "（该子任务未返回可展示内容）"

    if normalized.lower().startswith("error:"):
        return _humanize_aggregator_error_text(normalized)

    normalized = re.sub(r"(?im)^\s*\[object Object\],?\s*$", "", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip(" \n,")

    if "[object Object]" in normalized:
        lower_text = normalized.lower()
        agent_name = str((task or {}).get("agent") or "").strip()
        if agent_name in {"sql_agent", "yunyou_agent"} or any(
                marker in lower_text for marker in ("sql", "select ", " from ", "holter")
        ):
            return "该查询子任务返回了损坏的结构化片段，暂时无法整理成可直接展示的结果。"

    return normalized or "（该子任务未返回可展示内容）"


def _build_aggregated_task_title(task_id: str, task: Optional[SubTask]) -> str:
    """为聚合后的子任务结果生成更友好的小节标题。"""
    agent_name = str((task or {}).get("agent") or "").strip()
    agent_label = {
        "search_agent": "联网检索",
        "weather_agent": "天气信息",
        "yunyou_agent": "业务数据",
        "sql_agent": "数据库查询",
        "medical_agent": "医疗建议",
        "code_agent": "代码处理",
        "CHAT": "综合答复",
        "ChatAgent": "综合答复",
        "chat_node": "综合答复",
    }.get(agent_name, agent_name or f"任务 {task_id}")

    raw_focus = str((task or {}).get("input") or "").strip()
    focus = re.split(r"[。！？!?；;\n]", raw_focus)[0].strip()
    focus = re.sub(r"^(请|帮我|帮忙|查询|检索|回答|说明|判断|告诉我)\s*", "", focus)
    focus = focus.strip("，,。；;：: ")
    if len(focus) > 24:
        focus = focus[:24].rstrip("，,。；;：: ") + "..."

    if focus and focus != agent_label:
        return f"{agent_label} · {focus}"
    return agent_label


def _build_deterministic_aggregation(
        user_request: str,
        normalized_results: List[tuple[str, str]],
        tasks_by_id: Optional[Dict[str, SubTask]] = None,
) -> str:
    """构造稳定、快速、不依赖额外大模型调用的最终聚合结果。"""
    if not normalized_results:
        return "没有任何子任务结果可以聚合。"

    # 单任务场景直接返回正文，避免重复包裹影响阅读。
    if len(normalized_results) == 1:
        task_id, text = normalized_results[0]
        task = (tasks_by_id or {}).get(task_id)
        return _clean_aggregated_task_text(task, text)

    success_parts: List[str] = []
    failure_parts: List[str] = []
    for index, (task_id, text) in enumerate(normalized_results, start=1):
        task = (tasks_by_id or {}).get(task_id)
        normalized_text = _clean_aggregated_task_text(task, text)
        if not normalized_text:
            continue
        if normalized_text.startswith("有一项子任务未能完成") or normalized_text.startswith("联网检索工具调用异常"):
            failure_parts.append(normalized_text)
            continue
        section_title = _build_aggregated_task_title(task_id, task)
        success_parts.append(f"### {index}. {section_title}\n{normalized_text}")

    if success_parts:
        merged_parts = [
            f"## 已并行完成 {len(normalized_results)} 个子任务",
            "> 系统已按可并行部分同时执行，并整理为以下合并结果。",
            *success_parts,
        ]
        if failure_parts:
            merged_parts.append("### 需要注意")
            merged_parts.extend(f"- {item}" for item in failure_parts)
        return "\n\n".join(merged_parts).strip()
    if failure_parts:
        return "\n".join(failure_parts).strip()
    return "暂未获得可直接展示的结果，请稍后重试。"


async def aggregator_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
    """聚合节点：将多个子任务结果整合成最终回答"""
    started_at = time.perf_counter()
    worker_items = list(state.get("worker_results") or [])
    tasks_by_id: Dict[str, SubTask] = {}
    results: Dict[str, Any] = {}

    if worker_items:
        for item in worker_items:
            if not isinstance(item, dict):
                continue
            task_id = str(item.get("task_id") or "")
            if not task_id:
                continue
            task_input = str(item.get("task") or "").strip()
            agent_name = str(item.get("agent") or "CHAT").strip() or "CHAT"
            error_text = str(item.get("error") or "").strip()
            result_text = str(item.get("result") or "").strip()
            tasks_by_id[task_id] = {
                "id": task_id,
                "agent": agent_name,
                "input": task_input,
                "depends_on": [],
                "status": TaskStatus.ERROR.value if error_text else TaskStatus.DONE.value,
                "result": result_text,
            }
            if error_text:
                results[task_id] = f"Error: {error_text}"
            elif result_text:
                results[task_id] = result_text

    if not results:
        tasks = state.get("task_list", []) or []
        pending_like = [t for t in tasks if str(t.get("status") or "") in PENDING_TASK_STATUSES]
        if pending_like:
            log.info("Aggregator: 检测到未终态任务（含 pending/pending_approval/dispatched），本轮不输出正文。")
            return {"direct_answer": ""}

        results = state.get("task_results", {}) or {}
        tasks_by_id = {
            str(task.get("id") or ""): task
            for task in tasks
            if isinstance(task, dict) and task.get("id")
        }

    if not results:
        msg = AIMessage(
            content="没有任何子任务结果可以聚合。",
            name="Aggregator",
            response_metadata={"synthetic": True, "force_emit": True},
        )
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        log.info(f"⏱️ Aggregator 空结果耗时: {elapsed_ms}ms")
        return {"messages": [msg], "direct_answer": "没有任何子任务结果可以聚合。"}

    max_chars = AGGREGATOR_CONFIG.max_result_chars
    normalized_results: List[tuple[str, str]] = []
    for task_id, value in sorted(results.items(), key=_task_sort_key):
        normalized_results.append(
            (str(task_id), _normalize_aggregator_result_text(str(task_id), value, max_chars))
        )

    user_request = _latest_human_message(state.get("messages", []))
    deterministic_answer = _build_deterministic_aggregation(
        user_request,
        normalized_results,
        tasks_by_id=tasks_by_id,
    )

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
    aggregation_usage: Dict[str, int] = {}
    try:
        res_list = [f"【任务 {task_id} 的反馈】:\n{text}" for task_id, text in normalized_results]
        agg_msg = "\n\n---\n\n".join(res_list)

        prompt = ChatPromptTemplate.from_messages([
            ("system", AggregatorPrompt.get_system_prompt()),
            ("human", f"用户的原始请求：\n{user_request}\n\n执行结果反馈：\n{agg_msg}")
        ])

        response = await (prompt | model).ainvoke({}, config=config)
        parsed_content = _content_to_text(getattr(response, "content", "")).strip()
        if parsed_content:
            final_content = parsed_content
            agg_mode = "llm"
            aggregation_usage = _extract_usage_from_response_metadata(
                getattr(response, "response_metadata", {}) or {},
            )
    except Exception as exc:
        log.warning(f"Aggregator LLM 聚合失败，已回退确定性聚合: {exc}")

    msg = AIMessage(
        content=final_content,
        name="Aggregator",
        response_metadata={
            "synthetic": True,
            "force_emit": True,
            "usage": aggregation_usage,
        },
    )
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    log.info(f"⏱️ Aggregator [{agg_mode}] 耗时: {elapsed_ms}ms")
    return {"messages": [msg], "direct_answer": final_content}


async def memory_manager_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
    """结束前统一执行滑动摘要记忆收敛，控制多轮对话窗口。"""
    messages = list(state.get("messages", []) or [])
    round_count = count_dialog_rounds(messages)
    if round_count <= HISTORY_SUMMARY_TRIGGER_ROUNDS:
        return {}

    compact_messages = await build_sliding_summary_messages(
        messages,
        model=model,
        config=config,
        trigger_rounds=HISTORY_SUMMARY_TRIGGER_ROUNDS,
        summarize_rounds=HISTORY_SUMMARY_BATCH_ROUNDS,
        keep_recent_rounds=HISTORY_SUMMARY_KEEP_RECENT_ROUNDS,
    )
    if compact_messages == messages:
        return {}

    log.info(
        "Memory Manager: summarized dialog rounds=%s -> compact_messages=%s",
        round_count,
        len(compact_messages),
    )
    return {
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *compact_messages],
    }


async def chat_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
    """聊天节点：处理通用对话"""
    started_at = time.perf_counter()
    direct_ans = state.get("direct_answer", "")
    current_step_input = str(state.get("current_task") or state.get("current_step_input") or "").strip()
    if direct_ans and len(direct_ans.strip()) > 3:
        msg = AIMessage(content=direct_ans, name="ChatAgent", response_metadata={"synthetic": True})
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        log.info(f"⏱️ chat_node 直出耗时: {elapsed_ms}ms")
    else:
        prompt = ChatFallbackPrompt.get_system_prompt()
        # Chat 兜底只保留最近窗口，避免超长历史拖慢响应。
        recent_messages = state.get("messages", [])[-AGENT_LOOP_CONFIG.context_history_messages:]
        if current_step_input:
            last_message = recent_messages[-1] if recent_messages else None
            last_is_same_human = (
                isinstance(last_message, HumanMessage)
                and _content_to_text(getattr(last_message, "content", "")).strip() == current_step_input
            )
            if not last_is_same_human:
                recent_messages = list(recent_messages) + [HumanMessage(content=current_step_input)]
        request_messages = _build_runtime_prefix_messages(
            llm_config=state.get("llm_config") or {},
            dynamic_blocks=[get_agent_date_context(), str(state.get("context_summary") or "").strip()],
            extra_static_blocks=[ChatFallbackPrompt.get_system_prompt()],
        ) + recent_messages
        runtime_config = _build_non_streaming_config(config)
        run_id = str(
            config.get("configurable", {}).get("run_id")
            or state.get("session_id")
            or config.get("configurable", {}).get("thread_id")
            or ""
        ).strip()

        total_timeout_sec = float(CHAT_NODE_TOTAL_TIMEOUT_SEC)

        msg: Optional[AIMessage] = None
        final_error: Optional[Exception] = None

        try:
            response = await _ainvoke_with_timeout(
                model.ainvoke(request_messages, config=runtime_config),
                timeout_sec=max(1.0, total_timeout_sec),
                timeout_label="chat_node.ainvoke",
            )
            content = _content_to_text(getattr(response, "content", "")).strip()
            if content:
                msg = AIMessage(
                    content=content,
                    name="ChatAgent",
                    response_metadata={
                        **dict(getattr(response, "response_metadata", {}) or {}),
                        "usage": _extract_usage_from_response_metadata(
                            getattr(response, "response_metadata", {}) or {},
                        ),
                        "synthetic": True,
                    },
                )
            else:
                final_error = RuntimeError("chat_node.empty_response")
        except Exception as exc:
            final_error = exc
            if total_timeout_sec >= 1.0 and not request_cancellation_service.is_cancelled(run_id):
                try:
                    retry_response = await _ainvoke_with_timeout(
                        model.ainvoke(
                            _build_chat_retry_messages(
                                prompt,
                                recent_messages,
                                llm_config=state.get("llm_config") or {},
                                context_summary=str(state.get("context_summary") or "").strip(),
                            ),
                            config=runtime_config,
                        ),
                        timeout_sec=min(max(total_timeout_sec * 0.5, 10.0), 30.0),
                        timeout_label="chat_node.retry_ainvoke",
                    )
                    retry_content = _content_to_text(getattr(retry_response, "content", "")).strip()
                    if retry_content:
                        msg = AIMessage(
                            content=retry_content,
                            name="ChatAgent",
                            response_metadata={
                                **dict(getattr(retry_response, "response_metadata", {}) or {}),
                                "usage": _extract_usage_from_response_metadata(
                                    getattr(retry_response, "response_metadata", {}) or {},
                                ),
                                "synthetic": True,
                            },
                        )
                        final_error = None
                except Exception as retry_exc:
                    final_error = retry_exc

        if msg is None:
            graceful_reply = _build_graceful_chat_fallback(
                current_step_input or _latest_human_message(state.get("messages", []))
            )
            if graceful_reply:
                elapsed_ms = int((time.perf_counter() - started_at) * 1000)
                log.error(
                    "⚠️ [底层报错] ChatAgent 调用失败，已走友好降级回复。",
                    exc_info=_exc_info_tuple(final_error),
                )
                return {
                    "messages": [
                        AIMessage(
                            content=graceful_reply,
                            name="ChatAgent",
                            response_metadata={"synthetic": True, "force_emit": True, "fallback": True},
                        )
                    ],
                    "current_step_agent": "chat_node",
                    "error_message": None,
                    "error_detail": None,
                    "error_payload": None,
                }
            error_payload = _build_agent_error_payload(
                agent_name="ChatAgent",
                exc=final_error or RuntimeError("chat_node.empty_response"),
                step_input=current_step_input or _latest_human_message(state.get("messages", [])),
            )
            log.error("⚠️ [底层报错] ChatAgent 执行失败。", exc_info=_exc_info_tuple(final_error))
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            log.info(f"⏱️ chat_node 模型耗时: {elapsed_ms}ms")
            return {
                "error_message": error_payload["message"],
                "error_detail": error_payload,
                "error_payload": error_payload,
                "current_step_agent": "chat_node",
            }

        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        log.info(f"⏱️ chat_node 模型耗时: {elapsed_ms}ms")
    return {
        "messages": [msg],
        "current_step_agent": "chat_node",
        "error_message": None,
        "error_detail": None,
        "error_payload": None,
    }


async def single_agent_node(
        state: GraphState,
        agent_name: str,
        model: BaseChatModel,
        config: RunnableConfig,
) -> dict:
    """单一Agent节点：执行单个专业Agent"""
    planned_step_input = str(state.get("current_task") or state.get("current_step_input") or "").strip()
    user_input = planned_step_input or _latest_human_message(state.get("messages", []))
    try:
        history_messages = _build_worker_history_messages_for_agent(
            agent_name=agent_name,
            history_messages=state.get("messages", []) or [],
            limit=AGENT_LOOP_CONFIG.context_history_messages,
            llm_config=state.get("llm_config") or {},
        )
        if not runtime_tool_registry.is_agent_enabled(agent_name, state.get("llm_config") or {}):
            content = runtime_tool_registry.get_disabled_agent_message(agent_name)
        else:
            content = await asyncio.to_thread(
                _run_agent_to_completion,
                agent_name,
                user_input,
                model,
                config,
                state.get("session_id") or "",
                history_messages,
                state.get("context_slots") or {},
                state.get("context_summary") or "",
                state.get("llm_config") or {},
            )
        if content == WORKER_CANCELLED_RESULT:
            return {"direct_answer": "", "current_step_agent": agent_name}
        if isinstance(content, dict) and content.get("type") == INTERRUPT_RESULT_TYPE:
            payload = _normalize_interrupt_payload(content.get("payload"))
            payload["agent_name"] = payload.get("agent_name") or agent_name
            return {"interrupt_payload": payload, "current_step_agent": agent_name}
        normalized_execution = _normalize_execution_result(content)
        normalized_content = normalized_execution["content"]
        response_metadata = {
            **normalized_execution["response_metadata"],
            "usage": _extract_usage_from_response_metadata(normalized_execution["response_metadata"]),
        }
        run_id = str(
            config.get("configurable", {}).get("run_id")
            or state.get("session_id")
            or config.get("configurable", {}).get("thread_id")
            or ""
        ).strip()
        live_streamed = AGENT_LIVE_STREAM_ENABLED and agent_stream_bus.has_streamed(run_id, agent_name)
        res_msg = AIMessage(
            content=normalized_content,
            name=agent_name,
            response_metadata={
                **response_metadata,
                "synthetic": True,
                "live_streamed": live_streamed,
            },
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
                },
                "current_step_agent": agent_name,
            }
        lowered_err = err_msg.lower()
        if agent_name in {
            "weather_agent",
            "sql_agent",
            "yunyou_agent",
            "search_agent",
            "medical_agent",
            "code_agent",
        }:
            error_payload = _build_agent_error_payload(
                agent_name=agent_name,
                exc=exc,
                step_input=user_input,
            )
            error_payload["message"] = _build_tool_failure_message(agent_name)
            error_payload["retryable"] = bool(
                error_payload.get("retryable")
                or isinstance(exc, ConnectionError)
                or "host is down" in lowered_err
                or any(marker in lowered_err for marker in SUPERVISOR_AGENT_FAILURE_CONNECTION_MARKERS)
            )
            log.error(f"⚠️ [底层报错] Agent {agent_name} 执行失败: {err_msg}", exc_info=True)
            return {
                "error_message": error_payload["message"],
                "error_detail": error_payload,
                "error_payload": error_payload,
                "current_step_agent": agent_name,
            }
        error_payload = _build_agent_error_payload(
            agent_name=agent_name,
            exc=exc,
            step_input=user_input,
        )
        log.error(f"⚠️ [底层报错] Agent {agent_name} 执行失败: {err_msg}", exc_info=True)
        return {
            "error_message": error_payload["message"],
            "error_detail": error_payload,
            "error_payload": error_payload,
            "current_step_agent": agent_name,
        }
    return {"messages": [res_msg], "current_step_agent": agent_name}


# ==================== 路由与编排逻辑 ====================
def _extract_last_step_result(state: GraphState) -> str:
    """抽取最近一步执行结果，写入 memory 供后续步骤参考。"""
    error_message = str(state.get("error_message") or "").strip()
    if error_message:
        return f"执行失败：{error_message}"

    for message in reversed(state.get("messages", []) or []):
        if not isinstance(message, AIMessage):
            continue
        text = _content_to_text(getattr(message, "content", "")).strip()
        if text:
            return text
    return ""


def _resolve_forced_specialist_agent(step_text: str, planned_agent: str = "") -> str:
    """对 SQL / 云柚步骤做硬路由，避免路由器误判导致子 Agent 失活。"""
    normalized_planned_agent = str(planned_agent or "").strip()
    if normalized_planned_agent in {"sql_agent", "yunyou_agent"}:
        return normalized_planned_agent

    normalized_step = str(step_text or "").strip()
    if not normalized_step:
        return ""

    if _looks_like_sql_request(normalized_step):
        return "sql_agent"
    if _looks_like_holter_request(normalized_step):
        return "yunyou_agent"
    return ""


def _resolve_executor_route(step_state: GraphState) -> str:
    """根据当前原子步骤的意图结果决定下一跳节点。"""
    forced_agent = _resolve_forced_specialist_agent(
        str(step_state.get("current_task") or step_state.get("current_step_input") or "").strip(),
        str(step_state.get("current_step_agent") or "").strip(),
    )
    if forced_agent:
        return forced_agent

    intent = str(step_state.get("intent") or "").strip()
    if intent in MEMBERS:
        return intent
    return "chat_node"


async def executor_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
    """
    Plan-and-Execute 执行器。

    1. 先把上一轮步骤结果沉淀到 memory；
    2. 再从 plan 队列取出下一步；
    3. 对该步骤重跑一次 IntentPolicy；
    4. 决定派给哪个单兵 Agent 或 chat_node。
    """
    memory = dict(state.get("memory") or {})
    step_results = list(memory.get("step_results") or [])

    previous_step = str(state.get("current_step_input") or "").strip()
    if previous_step:
        last_entry = step_results[-1] if step_results else None
        if not isinstance(last_entry, dict) or str(last_entry.get("step") or "").strip() != previous_step:
            step_results.append(
                {
                    "step": previous_step,
                    "agent": str(state.get("current_step_agent") or "").strip(),
                    "result": _compact_step_result_text(_extract_last_step_result(state)),
                }
            )
        memory["step_results"] = step_results

    remaining_plan = [str(item or "").strip() for item in list(state.get("plan") or []) if str(item or "").strip()]
    if not remaining_plan:
        return {
            "memory": memory,
            "executor_active": False,
            "current_step_input": None,
            "current_step_agent": None,
            "next": "__end__",
            "error_message": None,
            "error_detail": None,
        }

    step = remaining_plan.pop(0)
    step_state: GraphState = dict(state)
    step_state["messages"] = list(state.get("messages", []) or []) + [HumanMessage(content=step)]
    step_state["error_message"] = None
    step_state["error_detail"] = None
    step_state["interrupt_payload"] = None
    step_state["direct_answer"] = ""

    domain_update = await IntentPolicy.domain_router_node(step_state, model=model, config=config)
    step_state.update(domain_update)
    intent_update = await IntentPolicy.intent_router_node(step_state, model=model, config=config)
    step_state.update(intent_update)

    route_target = _resolve_executor_route(step_state)
    if step_state.get("is_complex"):
        log.warning(f"Executor 收到仍被判定为复杂的步骤，强制按单步执行: step={step}, intent={route_target}")

    memory["last_planned_step"] = step
    memory["step_results"] = step_results

    return {
        "plan": remaining_plan,
        "memory": memory,
        "data_domain": step_state.get("data_domain"),
        "domain_confidence": step_state.get("domain_confidence"),
        "domain_route_source": step_state.get("domain_route_source"),
        "domain_candidates": step_state.get("domain_candidates"),
        "intent_candidates": step_state.get("intent_candidates"),
        "route_strategy": step_state.get("route_strategy"),
        "route_reason": step_state.get("route_reason"),
        "domain_elapsed_ms": step_state.get("domain_elapsed_ms"),
        "intent": step_state.get("intent"),
        "intent_confidence": step_state.get("intent_confidence"),
        "is_complex": step_state.get("is_complex"),
        "direct_answer": step_state.get("direct_answer"),
        "intent_elapsed_ms": step_state.get("intent_elapsed_ms"),
        "current_step_input": step,
        "current_step_agent": route_target,
        "executor_active": True,
        "next": route_target,
        "error_message": None,
        "error_detail": None,
        "interrupt_payload": None,
    }


def _route_after_executor(state: GraphState) -> str:
    next_node = str(state.get("next") or "").strip()
    if (not state.get("executor_active")) or next_node == "__end__":
        return "__end__"
    if next_node in MEMBERS or next_node == "chat_node":
        return next_node
    return "chat_node"


def _route_after_step_completion(state: GraphState) -> str:
    if state.get("interrupt_payload"):
        return "__end__"
    if state.get("executor_active"):
        return "executor_node"
    return "__end__"


def _route_after_intent(state: GraphState) -> str:
    """
    条件边：Intent_Router → 单兵 / Planner / 聊天

    决策逻辑（与架构规范对齐）：
    1. is_complex=True → Parent_Planner_Node（Plan-and-Execute）
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


def _route_after_dispatch(state: GraphState) -> str:
    """弹夹循环：有 current_task 则进入路由，无任务或中断则结束。"""
    if state.get("interrupt_payload"):
        return "__end__"
    current_task = str(state.get("current_task") or "").strip()
    if (not current_task) or current_task == "END_TASK":
        return "__end__"
    return "Domain_Router_Node"


def _route_current_task_after_intent(state: GraphState) -> str:
    """当前原子任务只允许落到单兵 Agent 或 chat_node，不再二次进入 planner。"""
    forced_agent = _resolve_forced_specialist_agent(
        str(state.get("current_task") or state.get("current_step_input") or "").strip(),
        str(state.get("current_step_agent") or "").strip(),
    )
    if forced_agent:
        log.info(f"当前任务硬路由: {forced_agent}")
        return forced_agent

    conf = float(state.get("intent_confidence") or 0.0)
    intent = str(state.get("intent") or "CHAT").strip() or "CHAT"
    if conf >= 0.7 and intent in MEMBERS:
        log.info(f"当前任务路由: {intent} (conf={conf:.2f})")
        return intent
    log.info(f"当前任务路由: chat_node (intent={intent}, conf={conf:.2f})")
    return "chat_node"


def _route_after_evaluator(state: GraphState) -> str:
    """根据 evaluator 结论决定是继续 dispatch、回到 planner，还是终止。"""
    if state.get("interrupt_payload"):
        return "__end__"

    last_status = str(state.get("last_step_status") or "").strip()
    if last_status == TaskStatus.ERROR.value:
        remaining_plan = [str(item or "").strip() for item in list(state.get("plan") or []) if str(item or "").strip()]
        if remaining_plan:
            return "dispatch_node"
        pending_tasks = [
            task for task in list(state.get("task_list") or [])
            if str(task.get("status") or "").strip() in PENDING_TASK_STATUSES
        ]
        if pending_tasks:
            return "dispatch_node"
        return "__end__"

    if last_status == TaskStatus.DONE.value:
        return "dispatch_node"

    return "__end__"


def create_graph(model_config: Optional[dict] = None):
    """构建统一入口 + 串行弹夹循环的 Supervisor 主图。"""
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

    try:
        router_config = dict(config_dict)
        if router_model_name:
            router_config["model"] = router_model_name
        router_config["model_size"] = "fast"
        router_model, _ = create_model_from_config(**router_config)
        log.info("Tier-1/Tier-2 路由与规划统一挂载 fast model")
    except Exception as e:
        log.warning(f"挂载 fast 路由模型失败，回退至主模型: {e}")

    if simple_chat_model_name and simple_chat_model_name != config_dict.get("model"):
        if simple_chat_model_name == router_model_name and router_model is not model:
            chat_model = router_model
            log.info(f"简单对话引擎复用路由小模型: {simple_chat_model_name}")
        else:
            try:
                chat_config = dict(config_dict)
                chat_config["model"] = simple_chat_model_name
                chat_config["model_size"] = "fast"
                temp_chat_model, _ = create_model_from_config(**chat_config)
                chat_model = temp_chat_model
                log.info(f"简单对话引擎已挂载小模型: {simple_chat_model_name}")
            except Exception as e:
                log.warning(f"挂载简单对话小模型 [{simple_chat_model_name}] 失败，回退至主模型: {e}")
    elif router_model is not model:
        chat_model = router_model

    workflow = StateGraph(GraphState)

    # 上层的决策分析
    workflow.add_node(
        "Domain_Router_Node",
        functools.partial(IntentPolicy.domain_router_node, model=router_model),
        retry_policy=GRAPH_RETRY_POLICY,
    )
    workflow.add_node(
        "Intent_Router_Node",
        functools.partial(IntentPolicy.intent_router_node, model=router_model),
        retry_policy=GRAPH_RETRY_POLICY,
    )
    workflow.add_node(
        "Parent_Planner_Node",
        functools.partial(PlannerNode.planner_node, model=router_model),
        retry_policy=GRAPH_RETRY_POLICY,
    )
    workflow.add_node("dispatch_node", dispatch_node, retry_policy=GRAPH_RETRY_POLICY)
    workflow.add_node("evaluator_node", evaluator_node, retry_policy=GRAPH_RETRY_POLICY)

    workflow.add_node("chat_node", functools.partial(chat_node, model=chat_model), retry_policy=GRAPH_RETRY_POLICY)
    for name in MEMBERS:
        workflow.add_node(
            name,
            functools.partial(single_agent_node, agent_name=name, model=model),
            retry_policy=GRAPH_RETRY_POLICY,
        )
    # ================= 编织拓扑关系 =================
    workflow.add_edge(START, "Parent_Planner_Node")
    workflow.add_edge("Parent_Planner_Node", "dispatch_node")
    workflow.add_conditional_edges(
        "dispatch_node",
        _route_after_dispatch,
        {"Domain_Router_Node": "Domain_Router_Node", "__end__": END},
    )
    workflow.add_edge("Domain_Router_Node", "Intent_Router_Node")

    router_options = {name: name for name in MEMBERS}
    router_options.update({"chat_node": "chat_node"})
    workflow.add_conditional_edges("Intent_Router_Node", _route_current_task_after_intent, router_options)

    workflow.add_edge("chat_node", "evaluator_node")
    for name in MEMBERS:
        workflow.add_edge(name, "evaluator_node")
    workflow.add_conditional_edges(
        "evaluator_node",
        _route_after_evaluator,
        {
            "dispatch_node": "dispatch_node",
            "chat_node": "chat_node",
            "__end__": END,
        },
    )

    return workflow.compile(checkpointer=get_checkpointer("supervisor"))
