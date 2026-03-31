import asyncio
import functools
import re
import time
from typing import Any, Optional, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from common.llm.unified_loader import create_model_from_config
from common.utils.custom_logger import get_logger
from common.utils.retry_utils import GRAPH_RETRY_POLICY
from config.constants.agent_registry_keywords import AGENT_KEYWORDS, AgentKeywordGroup
from config.constants.supervisor_keywords import (
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
    RouteStrategy,
    WORKFLOW_INTERRUPT_RESULT_TYPE,
)
from config.runtime_settings import ROUTER_POLICY_CONFIG
from models.schemas.supervisor_schemas import RequestAnalysisDecision
from planner.planner import PlannerNode
from supervisor.agents.code_agent import CodeAgent
from supervisor.agents.medical_agent import MedicalAgent
from supervisor.agents.search_agent import SearchAgent
from supervisor.agents.sql_agent import SqlAgent
from supervisor.agents.weather_agent import WeatherAgent
from supervisor.agents.yunyou_agent import YunyouAgent
from supervisor.checkpointer import get_checkpointer
from app.supervisor.graph_state import AgentRequest, AgentState
from supervisor.registry import MEMBERS, agent_classes
from supervisor.supervisor_rule_support import (
    analyze_request_payload as _support_analyze_request_payload,
    collect_intent_signals as _support_collect_intent_signals,
    dedupe_keep_order as _support_dedupe_keep_order,
    has_dependency_hint as _support_has_dependency_hint,
    is_explicit_request_clause as _support_is_explicit_request_clause,
    looks_like_compound_request as _support_looks_like_compound_request,
    split_query_clauses as _support_split_query_clauses,
)
from supervisor.supervisor_support import (
    build_non_streaming_config as _support_build_non_streaming_config,
    can_reuse_weather_context as _support_can_reuse_weather_context,
    content_to_text as _support_content_to_text,
    extract_city_from_context_slots as _support_extract_city_from_context_slots,
    extract_recent_city_from_history as _support_extract_recent_city_from_history,
    has_recent_weather_fact as _support_has_recent_weather_fact,
    history_hint_intent as _support_history_hint_intent,
    history_requests_location as _support_history_requests_location,
    input_has_location_anchor as _support_input_has_location_anchor,
    is_followup_supplement as _support_is_followup_supplement,
    latest_human_message as _support_latest_human_message,
    looks_like_location_fragment as _support_looks_like_location_fragment,
    looks_like_weather_reuse_query as _support_looks_like_weather_reuse_query,
    parse_json_from_text as _support_parse_json_from_text,
    wants_weather_refresh as _support_wants_weather_refresh,
)

log = get_logger(__name__)
INTERRUPT_RESULT_TYPE = WORKFLOW_INTERRUPT_RESULT_TYPE


class Plan(BaseModel):
    steps: list[str] = Field(default_factory=list, description="原子的、可独立执行的子任务步骤列表")


_SIMPLE_REQUEST_CONNECTOR_RE = re.compile(
    r"(?:然后|接着|并且|以及|同时|顺便|后面|随后|再|\band\b|\bthen\b|(?<![A-Za-z0-9])和(?![A-Za-z0-9]))",
    flags=re.IGNORECASE,
)

_PLANNER_SYSTEM_PROMPT = (
    "你是一个任务拆解专家，请将用户的复杂请求拆分为多个原子的、可独立执行的子任务步骤。\n"
    "1. 只输出 JSON 对象，格式必须是 {\"steps\": [\"步骤1\", \"步骤2\"]}。\n"
    "2. 每个步骤都必须是单一动作，不能把多个动作塞进同一步。\n"
    "3. 必须保留原请求中的顺序关系、地点实体、对象实体和限定条件。\n"
    "4. 如果请求本身已经是单一步骤，则只返回一个步骤。\n"
    "5. 不要解释，不要输出 Markdown。"
)

_AGGREGATOR_SYSTEM_PROMPT = (
    "你是总管汇总官。请基于已经完成的全部子任务结果，直接给老板一版最终答复。"
    "只输出最终结论，不要重复拆解过程，不要再发号施令。"
)


def _parse_json_from_text(text: str) -> dict:
    return _support_parse_json_from_text(text, log=log)


def _build_non_streaming_config(config: RunnableConfig) -> RunnableConfig:
    return _support_build_non_streaming_config(config)


async def _ainvoke_with_timeout(awaitable, *, timeout_sec: float, timeout_label: str):
    try:
        return await asyncio.wait_for(awaitable, timeout=max(0.1, float(timeout_sec)))
    except asyncio.TimeoutError as exc:
        raise TimeoutError(timeout_label) from exc


def _content_to_text(content: Any) -> str:
    return _support_content_to_text(content)


def _latest_human_message(messages: list[BaseMessage]) -> str:
    return _support_latest_human_message(messages)


def _history_requests_location(messages: Optional[list[BaseMessage]]) -> bool:
    return _support_history_requests_location(messages)


def _looks_like_location_fragment(text: str) -> bool:
    return _support_looks_like_location_fragment(text)


def _extract_recent_city_from_history(messages: list[BaseMessage]) -> Optional[str]:
    return _support_extract_recent_city_from_history(messages)


def _extract_city_from_context_slots(context_slots: Optional[dict[str, Any]]) -> Optional[str]:
    return _support_extract_city_from_context_slots(context_slots)


def _input_has_location_anchor(text: str) -> bool:
    return _support_input_has_location_anchor(text)


def _history_hint_intent(messages: list[BaseMessage], latest_user_text: str = "") -> Optional[str]:
    return _support_history_hint_intent(messages, latest_user_text)


def _has_recent_weather_fact(messages: list[BaseMessage]) -> bool:
    return _support_has_recent_weather_fact(messages)


def _looks_like_weather_reuse_query(text: str) -> bool:
    return _support_looks_like_weather_reuse_query(text)


def _wants_weather_refresh(text: str) -> bool:
    return _support_wants_weather_refresh(text)


def _can_reuse_weather_context(messages: list[BaseMessage], latest_user_text: str) -> bool:
    return _support_can_reuse_weather_context(messages, latest_user_text)


def _looks_like_holter_request(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    keywords = SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.HOLTER_DOMAIN]
    return any(keyword in normalized for keyword in keywords)


def _looks_like_sql_request(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    if _looks_like_holter_request(normalized):
        return any(re.search(anchor, normalized) for anchor in SUPERVISOR_SQL_EXPLICIT_ANCHORS)
    patterns = SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.SQL_REGEX_PATTERNS]
    return any(re.search(pattern, normalized) for pattern in patterns)


def _looks_like_weather_request(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    keywords = SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.WEATHER_DOMAIN]
    return any(keyword in normalized for keyword in keywords)


def _looks_like_search_request(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    keywords = SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.SEARCH_DOMAIN]
    if any(keyword in normalized for keyword in keywords):
        return True
    if any(hint in normalized for hint in SUPERVISOR_SEARCH_GENERIC_QUERY_HINTS):
        if _looks_like_holter_request(normalized) or _looks_like_sql_request(normalized) or _looks_like_weather_request(
            normalized
        ):
            return False
        return True
    return False


def _is_weather_actionable_clause(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized or not _looks_like_weather_request(normalized):
        return False
    if _looks_like_weather_reuse_query(normalized) or _wants_weather_refresh(normalized):
        return True
    if any(mark in normalized for mark in ("?", "？")):
        return True
    if normalized.endswith(("吗", "么", "呢")):
        return True
    return any(re.search(pattern, normalized) for pattern in SUPERVISOR_WEATHER_ACTION_PATTERNS)


def _is_search_actionable_clause(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized or not _looks_like_search_request(normalized):
        return False
    if any(mark in normalized for mark in ("?", "？")):
        return True
    return any(hint in normalized for hint in SUPERVISOR_SEARCH_ACTION_HINTS)


def _looks_like_medical_request(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    keywords = tuple(keyword.lower() for keyword in AGENT_KEYWORDS[AgentKeywordGroup.MEDICAL])
    return any(keyword in normalized for keyword in keywords)


def _looks_like_code_request(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    keywords = tuple(keyword.lower() for keyword in AGENT_KEYWORDS[AgentKeywordGroup.CODE])
    if not any(keyword in normalized for keyword in keywords):
        return False
    if any(hint in normalized for hint in SUPERVISOR_CODE_ACTION_HINTS):
        return True
    if any(hint in normalized for hint in SUPERVISOR_CODE_LEARNING_HINTS):
        return False
    if any(re.search(pattern, normalized) for pattern in SUPERVISOR_CODE_SNIPPET_PATTERNS):
        return True
    if any(marker in normalized for marker in SUPERVISOR_CODE_LANGUAGE_ONLY_MARKERS):
        return False
    return any(keyword in normalized for keyword in keywords if keyword not in {"python", "java", "编程"})


def _looks_like_general_chat_request(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return True
    return not any(
        (
            _looks_like_holter_request(normalized),
            _looks_like_sql_request(normalized),
            _looks_like_weather_request(normalized) and _is_weather_actionable_clause(normalized),
            _looks_like_search_request(normalized) and _is_search_actionable_clause(normalized),
            _looks_like_medical_request(normalized),
            _looks_like_code_request(normalized),
        )
    )


def _collect_intent_signals(text: str) -> list[str]:
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


def _dedupe_keep_order(values: list[str]) -> list[str]:
    return _support_dedupe_keep_order(values)


def _has_dependency_hint(text: str) -> bool:
    return _support_has_dependency_hint(text)


def _is_explicit_request_clause(text: str) -> bool:
    return _support_is_explicit_request_clause(
        text,
        request_action_hints=SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.REQUEST_ACTION_HINT],
    )


def _split_query_clauses(user_text: str) -> list[str]:
    return _support_split_query_clauses(user_text)


def _analyze_request(user_text: str) -> RequestAnalysisDecision:
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


def _looks_like_compound_request(text: str) -> bool:
    return _support_looks_like_compound_request(
        text,
        analyze_request_fn=_analyze_request,
        route_strategy_complex_single=RouteStrategy.COMPLEX_SINGLE_DOMAIN.value,
        route_strategy_multi_split=RouteStrategy.MULTI_DOMAIN_SPLIT.value,
        complex_connector_hints=SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.COMPLEX_CONNECTOR_HINT],
    )


async def _invoke_structured_output_with_fallback(
    *,
    prompt: ChatPromptTemplate,
    model: BaseChatModel,
    schema: Type[BaseModel],
    inputs: dict[str, Any],
    config: RunnableConfig,
    max_tokens: int,
    log_name: str,
) -> BaseModel:
    runtime_config = _build_non_streaming_config(config)
    invoke_timeout_sec = max(1.0, float(ROUTER_POLICY_CONFIG.router_llm_timeout_sec))
    llm = model
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
                "[%s] structured output failed, json fallback also failed. structured_error=%s json_error=%s",
                log_name,
                last_structured_exc,
                exc,
            )
        raise


def _normalize_steps(steps: list[str], fallback_text: str) -> list[str]:
    normalized: list[str] = []
    for step in steps or []:
        text = str(step or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    if normalized:
        return normalized
    fallback = str(fallback_text or "").strip()
    return [fallback] if fallback else []


def _fallback_steps(user_text: str) -> list[str]:
    return _normalize_steps(_split_query_clauses(user_text), user_text)


def _is_simple_request(user_text: str) -> bool:
    normalized = str(user_text or "").strip()
    if not normalized:
        return True
    return not _SIMPLE_REQUEST_CONNECTOR_RE.search(normalized)


def _build_task_list(steps: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"t{index + 1}",
            "agent": "",
            "input": step,
            "depends_on": [],
            "status": "pending",
            "result": None,
        }
        for index, step in enumerate(steps)
    ]


def _classify_agent_failure(exc: Exception) -> tuple[str, str]:
    detail = str(exc or "").strip()
    if isinstance(exc, TimeoutError):
        return "处理超时，请稍后重试。", detail
    if isinstance(exc, ConnectionError):
        return "模型服务连接异常，请稍后重试。", detail
    return "底层服务执行异常，请稍后重试。", detail


def _build_graceful_chat_fallback(user_text: str) -> str:
    normalized = (user_text or "").strip()
    if not normalized:
        return ""
    if _looks_like_weather_request(normalized) and not _is_weather_actionable_clause(normalized):
        return "要是你想查某个城市的实时天气、降雨或未来预报，直接告诉我城市就行。"
    if any(token in normalized for token in ("在吗", "有人吗", "你怎么了", "你还好吗")):
        return "我在。你把问题直接发出来，我继续处理。"
    if any(token in normalized for token in ("累", "烦", "迷茫", "崩溃", "难受")):
        return "我在。你可以继续说具体发生了什么，我帮你一起理一理。"
    return ""


def _select_agent_for_task(task: str) -> tuple[str, float]:
    normalized = str(task or "").strip()
    if _looks_like_holter_request(normalized):
        return "yunyou_agent", 0.98
    if _looks_like_sql_request(normalized):
        return "sql_agent", 0.95
    if _looks_like_weather_request(normalized) and _is_weather_actionable_clause(normalized):
        return "weather_agent", 0.93
    if _looks_like_search_request(normalized) and _is_search_actionable_clause(normalized):
        return "search_agent", 0.91
    if _looks_like_medical_request(normalized):
        return "medical_agent", 0.90
    if _looks_like_code_request(normalized):
        return "code_agent", 0.90
    return "chat_node", 0.50


def _dispatch_instruction_count(messages: list[BaseMessage]) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message, SystemMessage) and "【调度器指令】" in _content_to_text(getattr(message, "content", ""))
    )


def _should_aggregate(state: AgentState) -> bool:
    messages = list(state.get("messages", []) or [])
    if _dispatch_instruction_count(messages) < 2:
        return False
    return any(isinstance(message, AIMessage) for message in messages)


def _build_named_ai_message(content: Any, *, name: str, metadata: Optional[dict[str, Any]] = None) -> AIMessage:
    return AIMessage(
        content=_content_to_text(content).strip(),
        name=name,
        response_metadata={"synthetic": True, **(metadata or {})},
    )


def _run_agent_to_completion(
    agent_name: str,
    user_input: str,
    model: BaseChatModel,
    config: RunnableConfig,
    session_id: str = "",
    history_messages: Optional[list[BaseMessage]] = None,
) -> Any:
    req = AgentRequest(
        user_input=user_input,
        model=model,
        session_id=session_id,
        subgraph_id=agent_name,
        llm_config={},
        state={
            "messages": history_messages or [],
            "current_task": user_input,
        },
    )
    agent_instance = agent_classes[agent_name].cls(req)
    final_response: Optional[AIMessage] = None
    last_error = ""

    for event in agent_instance.run(req, config=config):
        if not isinstance(event, dict):
            continue
        if "error" in event:
            last_error = str(event.get("error") or "").strip()
        if "interrupt" in event:
            return {"type": INTERRUPT_RESULT_TYPE, "payload": event.get("interrupt")}
        for node_val in event.values():
            if not isinstance(node_val, dict):
                continue
            for message in node_val.get("messages", []):
                if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
                    final_response = message

    if final_response is not None:
        final_text = _content_to_text(getattr(final_response, "content", "")).strip()
        if final_text:
            return final_text
    if last_error:
        raise RuntimeError(last_error)
    raise RuntimeError(f"{agent_name} 未生成最终响应。")


async def planner_node(state: AgentState, config: RunnableConfig, req: AgentRequest) -> dict:
    planner_update = await PlannerNode.planner_node(state, model=req.model, config=config)
    return {
        "plan": list(planner_update.get("plan", []) or []),
        "current_task": str(planner_update.get("current_task") or ""),
    }


async def dispatch_node(state: AgentState, config: RunnableConfig):
    plan = state.get("plan", [])
    if not plan:
        return {"current_task": "END_TASK"}
    new_plan = list(plan)
    task = new_plan.pop(0)
    task_msg = SystemMessage(content=f"【调度器指令】：请专注执行当前子任务 -> {task}")
    return {
        "plan": new_plan,
        "current_task": task,
        "messages": [task_msg],
    }


def intent_policy(state: AgentState) -> str:
    current_task = str(state.get("current_task") or "").strip()
    if not current_task or current_task == "END_TASK":
        return END
    if _looks_like_holter_request(current_task):
        return "yunyou_agent"
    if _looks_like_sql_request(current_task):
        return "sql_agent"
    if _looks_like_weather_request(current_task):
        return "weather_agent"
    if _looks_like_search_request(current_task):
        return "search_agent"
    if _looks_like_medical_request(current_task):
        return "medical_agent"
    if _looks_like_code_request(current_task):
        return "code_agent"
    return "search_agent"


def route_after_dispatch(state: AgentState) -> str:
    if state.get("current_task") == "END_TASK":
        return END
    return intent_policy(state)


def build_supervisor_graph(req: AgentRequest):
    def _agent_req(agent_name: str) -> AgentRequest:
        return AgentRequest(
            user_input=req.user_input,
            state=req.state,
            session_id=req.session_id,
            subgraph_id=agent_name,
            model=req.model,
            llm_config=req.llm_config,
        )

    workflow = StateGraph(AgentState)

    workflow.add_node("planner_node", functools.partial(planner_node, req=req))
    workflow.add_node("dispatch_node", dispatch_node)

    workflow.add_node("weather_agent", WeatherAgent(_agent_req("weather_agent")).graph)
    workflow.add_node("search_agent", SearchAgent(_agent_req("search_agent")).graph)
    workflow.add_node("sql_agent", SqlAgent(_agent_req("sql_agent")).graph)
    workflow.add_node("code_agent", CodeAgent(_agent_req("code_agent")).graph)
    workflow.add_node("medical_agent", MedicalAgent(_agent_req("medical_agent")).graph)
    workflow.add_node("yunyou_agent", YunyouAgent(_agent_req("yunyou_agent")).graph)

    workflow.add_edge(START, "planner_node")
    workflow.add_edge("planner_node", "dispatch_node")
    workflow.add_conditional_edges("dispatch_node", route_after_dispatch)

    workflow.add_edge("weather_agent", "dispatch_node")
    workflow.add_edge("search_agent", "dispatch_node")
    workflow.add_edge("sql_agent", "dispatch_node")
    workflow.add_edge("code_agent", "dispatch_node")
    workflow.add_edge("medical_agent", "dispatch_node")
    workflow.add_edge("yunyou_agent", "dispatch_node")

    return workflow.compile(checkpointer=get_checkpointer("supervisor"))


def create_graph(model_config: Optional[dict] = None):
    model, _provider = create_model_from_config(**(model_config or {}))
    req = AgentRequest(
        user_input="",
        model=model,
        session_id="",
        subgraph_id="supervisor",
        llm_config=model_config or {},
    )
    return build_supervisor_graph(req)
