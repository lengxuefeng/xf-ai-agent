import functools
import queue
import json
import re
import threading
import time
from typing import Optional, List, Dict, Any, TypedDict, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START
from langgraph.constants import Send
from pydantic import BaseModel

from agent.graph_state import AgentRequest
from agent.graphs.checkpointer import get_checkpointer
from agent.graphs.state import GraphState, SubTask, WorkerResult
from agent.graphs.supervisor_support import (
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
from agent.graphs.supervisor_rule_support import (
    analyze_request_payload as _support_analyze_request_payload,
    build_agent_specific_task_input as _support_build_agent_specific_task_input,
    build_planner_fallback_tasks as _support_build_planner_fallback_tasks,
    build_rule_based_multidomain_tasks as _support_build_rule_based_multidomain_tasks,
    collect_intent_signals as _support_collect_intent_signals,
    dedupe_keep_order as _support_dedupe_keep_order,
    extract_agent_focus_text as _support_extract_agent_focus_text,
    has_dependency_hint as _support_has_dependency_hint,
    is_explicit_request_clause as _support_is_explicit_request_clause,
    looks_like_compound_request as _support_looks_like_compound_request,
    select_primary_agent_for_clause as _support_select_primary_agent_for_clause,
    split_query_clauses as _support_split_query_clauses,
)
from agent.llm.unified_loader import create_model_from_config
from agent.prompts.domain_prompt import DomainPrompt
from agent.registry import agent_classes, MEMBERS
from agent.prompts.supervisor_prompt import (
    IntentRouterPrompt,
    ChatFallbackPrompt,
    PlannerPrompt,
    ReflectionPrompt,
    AggregatorPrompt,
)
from constants.agent_registry_keywords import AGENT_KEYWORDS, AgentKeywordGroup
from constants.approval_constants import DEFAULT_ALLOWED_DECISIONS, DEFAULT_INTERRUPT_MESSAGE
from constants.supervisor_keywords import (
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
    WORKFLOW_REFLECTION_CONFIG,
)
from enums.agent_enum import AgentTypeEnum
from schemas.supervisor_schemas import (
    DomainDecision,
    IntentDecision,
    PlannerDecision,
    PlannerTaskDecision,
    ReflectionDecision,
    RequestAnalysisDecision,
)
from services.route_metrics_service import route_metrics_service
from services.agent_stream_bus import agent_stream_bus
from services.request_cancellation_service import request_cancellation_service
from utils.custom_logger import get_logger
from utils.date_utils import get_agent_date_context


from utils.custom_logger import get_logger
log = get_logger(__name__)


class PlannerNode:
    @staticmethod
    def _build_planner_state_payload(task_list: List[SubTask], *, planner_source: str, elapsed_ms: int) -> dict:
        """为 DAG 规划节点构造统一初始状态。"""
        return {
            "task_list": task_list,
            "task_results": {},
            "current_wave": 0,
            "max_waves": len(task_list) * 2 + 2,
            "planner_source": planner_source,
            "planner_elapsed_ms": elapsed_ms,
            "reflection_round": 0,
            "max_reflection_rounds": WORKFLOW_REFLECTION_CONFIG.max_rounds,
            "next_task_sequence": len(task_list) + 1,
            "reflection_source": "",
            "reflection_summary": "",
        }


    @staticmethod
    def parent_planner_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
        from agent.graphs.supervisor import (
            _latest_human_message, _invoke_structured_output_with_fallback,
            _build_rule_based_multidomain_tasks, _build_planner_fallback_tasks
        )
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
            return PlannerNode._build_planner_state_payload(rule_tasks, planner_source="rule_split", elapsed_ms=elapsed_ms)

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
            log.info(
                f"Tier-2 Planner: Generated {len(deterministic_tasks)} tasks -> {[t['id'] for t in deterministic_tasks]}")
            return PlannerNode._build_planner_state_payload(
                deterministic_tasks,
                planner_source="single_domain_guard",
                elapsed_ms=elapsed_ms,
            )

        # 默认关闭 LLM Planner：生产场景优先稳定性与可解释性，避免模型规划漂移。
        if not ROUTER_POLICY_CONFIG.planner_llm_fallback_enabled:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            log.info(f"⏱️ Parent Planner [deterministic_fallback] 耗时: {elapsed_ms}ms")
            log.info(
                f"Tier-2 Planner: Generated {len(deterministic_tasks)} tasks -> {[t['id'] for t in deterministic_tasks]}")
            return PlannerNode._build_planner_state_payload(
                deterministic_tasks,
                planner_source="deterministic_fallback",
                elapsed_ms=elapsed_ms,
            )

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
            for idx, t in enumerate(tasks_data, start=1):
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
        return PlannerNode._build_planner_state_payload(task_list, planner_source=planner_source, elapsed_ms=elapsed_ms)



