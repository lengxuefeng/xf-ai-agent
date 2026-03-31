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

from supervisor.graph_state import AgentRequest
from supervisor.checkpointer import get_checkpointer
from supervisor.state import GraphState, SubTask, WorkerResult

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
    CHAT_NODE_FIRST_TOKEN_TIMEOUT_SEC,
    CHAT_NODE_STREAM_ENABLED,
    CHAT_NODE_TOTAL_TIMEOUT_SEC,
    MODEL_TIERING_CONFIG,
    ROUTER_POLICY_CONFIG,
    WORKFLOW_REFLECTION_CONFIG,
)
from common.enums.agent_enum import AgentTypeEnum
from models.schemas.supervisor_schemas import (
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
from common.utils.custom_logger import get_logger
from common.utils.date_utils import get_agent_date_context


from common.utils.custom_logger import get_logger
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
        from supervisor.supervisor import (
            _latest_human_message, _invoke_structured_output_with_fallback
        )
        from planner.task_builder import (
            build_rule_based_multidomain_tasks as _build_rule_based_multidomain_tasks,
            build_planner_fallback_tasks as _build_planner_fallback_tasks
        )
        from supervisor.supervisor import (
             _split_query_clauses, _collect_intent_signals,
             _is_explicit_request_clause, _select_primary_agent_for_clause,
             _dedupe_keep_order, _has_dependency_hint
        )
        
        def local_build_rule_tasks(user_text, candidate_agents, route_strategy):
            return _build_rule_based_multidomain_tasks(
                user_text,
                candidate_agents=candidate_agents,
                route_strategy=route_strategy,
                split_query_clauses_fn=_split_query_clauses,
                collect_intent_signals_fn=_collect_intent_signals,
                is_explicit_request_clause_fn=_is_explicit_request_clause,
                select_primary_agent_for_clause_fn=_select_primary_agent_for_clause,
                dedupe_keep_order_fn=_dedupe_keep_order,
                has_dependency_hint_fn=_has_dependency_hint,
                route_strategy_single=RouteStrategy.SINGLE_DOMAIN.value,
                route_strategy_complex_single=RouteStrategy.COMPLEX_SINGLE_DOMAIN.value,
                route_strategy_multi_split=RouteStrategy.MULTI_DOMAIN_SPLIT.value,
                pending_status=TaskStatus.PENDING.value,
            )
            
        def local_build_fallback_tasks(user_text, intent_candidates, route_strategy, fallback_intent):
            from supervisor.registry import MEMBERS
            return _build_planner_fallback_tasks(
                user_text=user_text,
                intent_candidates=intent_candidates,
                route_strategy=route_strategy,
                fallback_intent=fallback_intent,
                has_dependency_hint_fn=_has_dependency_hint,
                route_strategy_single=RouteStrategy.SINGLE_DOMAIN.value,
                route_strategy_multi_split=RouteStrategy.MULTI_DOMAIN_SPLIT.value,
                multi_domain_agent_priority=MULTI_DOMAIN_AGENT_PRIORITY,
                members=MEMBERS,
                pending_status=TaskStatus.PENDING.value,
            )
        """第三级：父规划器 (Parent_Planner_Node)"""
        started_at = time.perf_counter()
        messages = state.get("messages", [])
        latest_user_text = _latest_human_message(messages)
        route_strategy = (state.get("route_strategy") or RouteStrategy.SINGLE_DOMAIN.value).strip()
        intent_candidates = [str(item) for item in (state.get("intent_candidates") or []) if isinstance(item, str)]
        intent_name = str(state.get("intent") or "").strip()

        # 根因修复：优先使用统一规则规划器，避免多域问题被 LLM 规划器抖动影响。
        rule_tasks = local_build_rule_tasks(
            latest_user_text,
            candidate_agents=intent_candidates,
            route_strategy=route_strategy,
        )
        if rule_tasks:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            log.info(f"⏱️ Parent Planner [rule_split] 耗时: {elapsed_ms}ms")
            log.info(f"Tier-2 Planner: Generated {len(rule_tasks)} tasks -> {[t['id'] for t in rule_tasks]}")
            return PlannerNode._build_planner_state_payload(rule_tasks, planner_source="rule_split", elapsed_ms=elapsed_ms)

        deterministic_tasks = local_build_fallback_tasks(
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



