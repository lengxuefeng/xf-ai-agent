import re
import time
from typing import List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig

from supervisor.state import GraphState
from prompts.agent_prompts.domain_prompt import DomainPrompt
from supervisor.registry import MEMBERS
from prompts.agent_prompts.supervisor_prompt import (
    IntentRouterPrompt,
)
from config.constants.workflow_constants import (
    AGENT_DOMAIN_MAP,
    RouteStrategy,
)
from config.runtime_settings import (
    ROUTER_POLICY_CONFIG,
)
from common.enums.agent_enum import AgentTypeEnum
from models.schemas.supervisor_schemas import (
    DomainDecision,
    IntentDecision,
    RequestAnalysisDecision,
)
from services.route_metrics_service import route_metrics_service
from common.utils.location_parser import (
    extract_valid_city_candidate,
)
from common.utils.custom_logger import get_logger
log = get_logger(__name__)


class IntentPolicy:
    _COMPOUND_CONNECTOR_RE = re.compile(
        r"(?:\b(?:and|then)\b|然后|接着|并且|以及|同时|顺便|后面|随后|先[^。！？\n]{0,80}?再)",
        flags=re.IGNORECASE,
    )
    _TOOL_INTENT_PRIORITY = (
        "yunyou_agent",
        "sql_agent",
        "weather_agent",
        "search_agent",
        "medical_agent",
        "code_agent",
    )
    _TOOL_INTENT_CONFIDENCE = {
        "yunyou_agent": 0.97,
        "sql_agent": 0.96,
        "weather_agent": 0.94,
        "search_agent": 0.91,
        "medical_agent": 0.9,
        "code_agent": 0.9,
    }
    _CHAT_BLOCKED_TOOL_INTENTS = frozenset({
        "yunyou_agent",
        "sql_agent",
        "weather_agent",
    })

    @staticmethod
    def _select_priority_tool_intent(
            *,
            latest_user_text: str,
            data_domain: str,
            intent_candidates: List[str],
            looks_like_holter_request,
            looks_like_sql_request,
            looks_like_weather_request,
            is_weather_actionable_clause,
            looks_like_search_request,
            is_search_actionable_clause,
            looks_like_medical_request,
            looks_like_code_request,
    ) -> str:
        lowered_text = str(latest_user_text or "").lower()
        # 最高优先级硬拦截，防止闲聊兜底误判。
        if "天气" in lowered_text and any(token in lowered_text for token in ["查", "预报", "南京", "北京", "上海"]):
            return "weather_agent"
        if "holter" in lowered_text or "云柚" in lowered_text:
            return "yunyou_agent"
        if "数据库" in lowered_text or "sql" in lowered_text:
            return "sql_agent"

        ordered_candidates: List[str] = []

        def _prepend(candidate_name: str) -> None:
            normalized_name = str(candidate_name or "").strip()
            if not normalized_name or normalized_name == "CHAT":
                return
            if normalized_name in ordered_candidates:
                ordered_candidates.remove(normalized_name)
            ordered_candidates.insert(0, normalized_name)

        def _append(candidate_name: str) -> None:
            normalized_name = str(candidate_name or "").strip()
            if not normalized_name or normalized_name == "CHAT" or normalized_name in ordered_candidates:
                return
            ordered_candidates.append(normalized_name)

        for candidate in intent_candidates or []:
            _append(candidate)

        if looks_like_holter_request(latest_user_text) or data_domain == "YUNYOU_DB":
            _prepend("yunyou_agent")
        if looks_like_sql_request(latest_user_text) or data_domain == "LOCAL_DB":
            _prepend("sql_agent")
        if looks_like_weather_request(latest_user_text) and is_weather_actionable_clause(latest_user_text):
            _prepend("weather_agent")
        if looks_like_search_request(latest_user_text) and is_search_actionable_clause(latest_user_text):
            _prepend("search_agent")
        if looks_like_medical_request(latest_user_text) and "medical_agent" in MEMBERS:
            _prepend("medical_agent")
        if looks_like_code_request(latest_user_text) and "code_agent" in MEMBERS:
            _prepend("code_agent")

        for candidate_name in IntentPolicy._TOOL_INTENT_PRIORITY:
            if candidate_name in ordered_candidates:
                return candidate_name
        return ""

    @staticmethod
    def _extract_location_entities(text: str) -> List[str]:
        entities: List[str] = []
        chunks = re.split(
            r"(?:然后|接着|并且|以及|同时|顺便|后面|随后|和|及|与|,|，|;|；|\band\b|\bthen\b|先|再)",
            text or "",
            flags=re.IGNORECASE,
        )
        for chunk in chunks:
            normalized_chunk = re.sub(
                r"^(?:先|再|接着|然后|请|帮我|帮忙|查一下|查|查询|看一下|看|搜一下|搜索|搜|讲个|讲|说个|说)\s*",
                "",
                str(chunk or "").strip(),
                flags=re.IGNORECASE,
            )
            city = extract_valid_city_candidate(normalized_chunk)
            if city and city not in entities:
                entities.append(city)
        return entities

    @staticmethod
    def _has_compound_structure(text: str) -> bool:
        normalized = (text or "").strip()
        if not normalized:
            return False
        if len(normalized) >= 6 and IntentPolicy._COMPOUND_CONNECTOR_RE.search(normalized):
            return True
        return False

    @staticmethod
    def _resolve_complex_intent(
            *,
            latest_user_text: str,
            data_domain: str,
            intent_candidates: List[str],
            looks_like_holter_request,
            looks_like_sql_request,
            looks_like_weather_request,
            looks_like_search_request,
            looks_like_medical_request,
            looks_like_code_request,
    ) -> str:
        if intent_candidates:
            return intent_candidates[0]
        if looks_like_holter_request(latest_user_text):
            return "yunyou_agent"
        if looks_like_sql_request(latest_user_text):
            return "sql_agent"
        if looks_like_weather_request(latest_user_text):
            return "weather_agent"
        if looks_like_search_request(latest_user_text):
            return "search_agent"
        if looks_like_medical_request(latest_user_text) and "medical_agent" in MEMBERS:
            return "medical_agent"
        if looks_like_code_request(latest_user_text) and "code_agent" in MEMBERS:
            return "code_agent"
        if data_domain == "YUNYOU_DB":
            return "yunyou_agent"
        if data_domain == "LOCAL_DB":
            return "sql_agent"
        if data_domain == "WEB_SEARCH":
            return "search_agent"
        return "CHAT"

    @staticmethod
    def _finalize_domain_decision(
            *,
            decision: DomainDecision,
            analysis: RequestAnalysisDecision,
            session_id: str,
            latest_user_text: str,
            started_at: float,
    ) -> dict:
        route_metrics_service.record_domain_decision(
            session_id=session_id,
            user_text=latest_user_text,
            domain=decision.data_domain,
            confidence=decision.confidence,
            source=decision.source,
        )
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        domain_candidates = analysis.candidate_domains or [decision.data_domain]
        intent_candidates = analysis.candidate_agents or []
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
            "route_strategy": analysis.route_strategy,
            "route_reason": analysis.reason,
            "domain_elapsed_ms": elapsed_ms,
        }


    @staticmethod
    def _finalize_intent_decision(
            *,
            intent: str,
            confidence: float,
            is_complex: bool,
            source: str,
            session_id: str,
            latest_user_text: str,
            started_at: float,
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

    @staticmethod
    def _build_routing_messages(state: GraphState, *, window: int) -> tuple[List, str]:
        messages = list(state.get("messages", []) or [])
        trimmed_messages = messages[-window:]
        current_task = str(state.get("current_task") or state.get("current_step_input") or "").strip()
        if current_task:
            trimmed_messages = trimmed_messages + [HumanMessage(content=current_task)]
            return trimmed_messages, current_task
        from supervisor.supervisor import _latest_human_message
        return trimmed_messages, _latest_human_message(trimmed_messages)


    @staticmethod
    async def domain_router_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
        from supervisor.supervisor import (
            _looks_like_holter_request, _looks_like_sql_request,
            _looks_like_weather_request, _is_weather_actionable_clause,
            _looks_like_search_request, _is_search_actionable_clause,
            _looks_like_medical_request, _looks_like_code_request,
            _looks_like_general_chat_request, _looks_like_compound_request,
            _latest_human_message, _support_is_followup_supplement,
            _history_hint_intent, _can_reuse_weather_context,
            _looks_like_location_fragment, _extract_city_from_context_slots,
            _extract_recent_city_from_history, _input_has_location_anchor,
            _analyze_request, _invoke_structured_output_with_fallback
        )
        """
        先识别“去哪类数据域”，再做意图路由。

        数据域定义：
        1. YUNYOU_DB: 云柚/holter 业务域
        2. LOCAL_DB: 本地业务库 SQL 域
        3. WEB_SEARCH: 互联网检索域
        4. GENERAL: 通用对话域
        """
        # 分类窗口：避免把过长历史塞给路由器，降低延迟和漂移概率。
        classify_window = max(3, int(ROUTER_POLICY_CONFIG.classifier_history_messages))
        classify_messages, latest_user_text = IntentPolicy._build_routing_messages(
            state,
            window=classify_window,
        )
        session_id = state.get("session_id") or config.get("configurable", {}).get("thread_id", "")
        request_analysis = _analyze_request(latest_user_text)
        started_at = time.perf_counter()

        # follow-up 补充句优先继承上轮域
        if _support_is_followup_supplement(latest_user_text, classify_messages):
            hinted_intent = _history_hint_intent(classify_messages, latest_user_text)
            if hinted_intent == AgentTypeEnum.YUNYOU.code:
                decision = DomainDecision(data_domain="YUNYOU_DB", confidence=0.94, source="history")
            elif hinted_intent == AgentTypeEnum.SQL.code:
                decision = DomainDecision(data_domain="LOCAL_DB", confidence=0.93, source="history")
            elif hinted_intent in {AgentTypeEnum.WEATHER.code, AgentTypeEnum.SEARCH.code}:
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
                return IntentPolicy._finalize_domain_decision(
                    decision=decision,
                    analysis=followup_analysis,
                    session_id=session_id,
                    latest_user_text=latest_user_text,
                    started_at=started_at,
                )

        # 多意图请求在 Domain 层即标记为“需要拆分”，防止被单域强路由吞掉。
        if (
                request_analysis.route_strategy == RouteStrategy.MULTI_DOMAIN_SPLIT.value
                and len(request_analysis.candidate_agents) >= 2
        ):
            primary_domain = (request_analysis.candidate_domains or ["GENERAL"])[0]
            decision = DomainDecision(data_domain=primary_domain, confidence=0.97, source="rule_multi_domain")
            return IntentPolicy._finalize_domain_decision(
                decision=decision,
                analysis=request_analysis,
                session_id=session_id,
                latest_user_text=latest_user_text,
                started_at=started_at,
            )

        # 规则优先
        if _looks_like_holter_request(latest_user_text):
            decision = DomainDecision(data_domain="YUNYOU_DB", confidence=0.98, source="rule")
        elif _looks_like_sql_request(latest_user_text):
            decision = DomainDecision(data_domain="LOCAL_DB", confidence=0.95, source="rule")
        elif (
                (_looks_like_weather_request(latest_user_text) and _is_weather_actionable_clause(latest_user_text))
                or (_looks_like_search_request(latest_user_text) and _is_search_actionable_clause(latest_user_text))
        ):
            decision = DomainDecision(data_domain="WEB_SEARCH", confidence=0.9, source="rule")
        else:
            # 默认策略：未知场景直接归入 GENERAL，避免每轮都走慢速 LLM 分类。
            # 只有显式打开开关才启用 LLM 兜底。
            if not ROUTER_POLICY_CONFIG.domain_llm_fallback_enabled:
                decision = DomainDecision(data_domain="GENERAL", confidence=0.88, source="rule_default_general")
            else:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", DomainPrompt.SYSTEM),
                    MessagesPlaceholder(variable_name="messages"),
                ])
                try:
                    structured = await _invoke_structured_output_with_fallback(
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
        return IntentPolicy._finalize_domain_decision(
            decision=decision,
            analysis=single_domain_analysis,
            session_id=session_id,
            latest_user_text=latest_user_text,
            started_at=started_at,
        )


    @staticmethod
    async def intent_router_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
        from supervisor.supervisor import (
            _looks_like_holter_request, _looks_like_sql_request,
            _looks_like_weather_request, _is_weather_actionable_clause,
            _looks_like_search_request, _is_search_actionable_clause,
            _looks_like_medical_request, _looks_like_code_request,
            _looks_like_general_chat_request, _looks_like_compound_request,
            _latest_human_message, _support_is_followup_supplement,
            _history_hint_intent, _can_reuse_weather_context,
            _looks_like_location_fragment, _extract_city_from_context_slots,
            _extract_recent_city_from_history, _input_has_location_anchor,
            _analyze_request, _invoke_structured_output_with_fallback
        )
        """第二级：意图识别节点 (Intent_Router_Node)"""
        session_id = state.get("session_id") or config.get("configurable", {}).get("thread_id", "")
        data_domain = (state.get("data_domain") or "GENERAL").upper()
        route_strategy = (state.get("route_strategy") or RouteStrategy.SINGLE_DOMAIN.value).strip()
        route_reason = (state.get("route_reason") or "").strip()
        intent_candidates = [str(item) for item in (state.get("intent_candidates") or []) if isinstance(item, str)]
        domain_conf = float(state.get("domain_confidence") or 0.0)
        domain_source = state.get("domain_route_source") or "unknown"
        # 保留最近多轮上下文，避免“用户补充参数”被误判为新问题导致重复追问。
        classify_window = max(3, int(ROUTER_POLICY_CONFIG.classifier_history_messages))
        trimmed_messages, latest_user_text = IntentPolicy._build_routing_messages(
            state,
            window=classify_window,
        )
        started_at = time.perf_counter()
        compound_signal = IntentPolicy._has_compound_structure(latest_user_text) or _looks_like_compound_request(
            latest_user_text
        )
        location_entities = IntentPolicy._extract_location_entities(latest_user_text)
        multi_location_signal = len(location_entities) >= 2
        strict_tool_intent = IntentPolicy._select_priority_tool_intent(
            latest_user_text=latest_user_text,
            data_domain=data_domain,
            intent_candidates=intent_candidates,
            looks_like_holter_request=_looks_like_holter_request,
            looks_like_sql_request=_looks_like_sql_request,
            looks_like_weather_request=_looks_like_weather_request,
            is_weather_actionable_clause=_is_weather_actionable_clause,
            looks_like_search_request=_looks_like_search_request,
            is_search_actionable_clause=_is_search_actionable_clause,
            looks_like_medical_request=_looks_like_medical_request,
            looks_like_code_request=_looks_like_code_request,
        )

        # 根因修复：多意图输入在 Intent 层直接标记复杂任务，强制走 Parent Planner。
        if route_strategy == RouteStrategy.MULTI_DOMAIN_SPLIT.value and len(intent_candidates) >= 2:
            log.info(
                "Intent multi-domain split: 命中多意图候选，转入 Parent Planner "
                f"(candidates={intent_candidates}, reason={route_reason})"
            )
            return IntentPolicy._finalize_intent_decision(
                intent="CHAT",
                confidence=max(domain_conf, 0.96),
                is_complex=True,
                source="analysis_multi_domain",
                direct_answer="",
                session_id=session_id,
                latest_user_text=latest_user_text,
                started_at=started_at,
            )

        # 单域但有先后依赖，也强制走规划节点，避免“先查再总结”被误判为单兵。
        if route_strategy == RouteStrategy.COMPLEX_SINGLE_DOMAIN.value and len(intent_candidates) == 1:
            planned_intent = intent_candidates[0]
            return IntentPolicy._finalize_intent_decision(
                intent=planned_intent,
                confidence=max(domain_conf, 0.93),
                is_complex=True,
                source="analysis_complex_single",
                direct_answer="",
                session_id=session_id,
                latest_user_text=latest_user_text,
                    started_at=started_at,
                )

        # 强制复杂任务：复合连接词、多地点实体、或复杂单域策略都必须进入 Planner。
        forced_complex = (
                compound_signal
                or multi_location_signal
                or route_strategy == RouteStrategy.COMPLEX_SINGLE_DOMAIN.value
        )
        if forced_complex:
            planned_intent = IntentPolicy._resolve_complex_intent(
                latest_user_text=latest_user_text,
                data_domain=data_domain,
                intent_candidates=intent_candidates,
                looks_like_holter_request=_looks_like_holter_request,
                looks_like_sql_request=_looks_like_sql_request,
                looks_like_weather_request=_looks_like_weather_request,
                looks_like_search_request=_looks_like_search_request,
                looks_like_medical_request=_looks_like_medical_request,
                looks_like_code_request=_looks_like_code_request,
            )
            complex_reason_parts = []
            if compound_signal:
                complex_reason_parts.append("compound_connector")
            if multi_location_signal:
                complex_reason_parts.append(f"multi_location:{','.join(location_entities)}")
            if route_strategy == RouteStrategy.COMPLEX_SINGLE_DOMAIN.value:
                complex_reason_parts.append("complex_single_domain")
            complex_reason = "|".join(complex_reason_parts) or route_reason or "forced_complex"
            log.info(
                "Intent forced complex: 命中复合任务信号，转入 Planner "
                f"(intent={planned_intent}, reason={complex_reason})"
            )
            return IntentPolicy._finalize_intent_decision(
                intent=planned_intent,
                confidence=max(domain_conf, 0.95 if planned_intent != "CHAT" else 0.9),
                is_complex=True,
                source=f"forced_complex[{complex_reason}]",
                direct_answer="",
                session_id=session_id,
                latest_user_text=latest_user_text,
                started_at=started_at,
            )

        # 天气追问优化：若上轮已给出天气事实，优先复用上下文做建议，不再重复调 weather_agent。
        if (
                _can_reuse_weather_context(trimmed_messages, latest_user_text)
                and strict_tool_intent not in IntentPolicy._CHAT_BLOCKED_TOOL_INTENTS
        ):
            log.info("Intent weather reuse: 复用最近天气上下文，直接路由 CHAT")
            return IntentPolicy._finalize_intent_decision(
                intent="CHAT",
                confidence=0.94,
                is_complex=False,
                source="history_weather_reuse",
                direct_answer="",
                session_id=session_id,
                latest_user_text=latest_user_text,
                started_at=started_at,
            )

        # 先处理“补充参数”场景：继承上轮领域，避免补充句被 SQL fast-path 截走。
        if _support_is_followup_supplement(latest_user_text, trimmed_messages):
            hinted_intent = _history_hint_intent(trimmed_messages, latest_user_text)
            if hinted_intent:
                log.info(f"Intent follow-up carry-over: route to [{hinted_intent}]")
                return IntentPolicy._finalize_intent_decision(
                    intent=hinted_intent,
                    confidence=0.93,
                    is_complex=False,
                    source="followup_history",
                    direct_answer="",
                    session_id=session_id,
                    latest_user_text=latest_user_text,
                    started_at=started_at,
                )

        # 先按数据域强约束路由，避免跨域误查
        if data_domain == "YUNYOU_DB":
            return IntentPolicy._finalize_intent_decision(
                intent="yunyou_agent",
                confidence=max(domain_conf, 0.95),
                is_complex=False,
                source=f"domain_{domain_source}",
                direct_answer="",
                session_id=session_id,
                latest_user_text=latest_user_text,
                started_at=started_at,
            )
        if data_domain == "LOCAL_DB" and _looks_like_sql_request(latest_user_text):
            return IntentPolicy._finalize_intent_decision(
                intent="sql_agent",
                confidence=max(domain_conf, 0.94),
                is_complex=False,
                source=f"domain_{domain_source}",
                direct_answer="",
                session_id=session_id,
                latest_user_text=latest_user_text,
                started_at=started_at,
            )
        if data_domain == "WEB_SEARCH":
            hinted_intent = _history_hint_intent(trimmed_messages, latest_user_text)
            if _looks_like_location_fragment(latest_user_text) and hinted_intent in {"weather_agent", "search_agent"}:
                fallback_agent = hinted_intent
            else:
                fallback_agent = "weather_agent" if _looks_like_weather_request(latest_user_text) else "search_agent"
            return IntentPolicy._finalize_intent_decision(
                intent=fallback_agent,
                confidence=max(domain_conf, 0.9),
                is_complex=False,
                source=f"domain_{domain_source}",
                direct_answer="",
                session_id=session_id,
                latest_user_text=latest_user_text,
                started_at=started_at,
            )

        prioritized_tool_intent = strict_tool_intent
        if prioritized_tool_intent:
            return IntentPolicy._finalize_intent_decision(
                intent=prioritized_tool_intent,
                confidence=max(
                    domain_conf,
                    IntentPolicy._TOOL_INTENT_CONFIDENCE.get(prioritized_tool_intent, 0.9),
                ),
                is_complex=False,
                source="priority_tool_intent",
                direct_answer="",
                session_id=session_id,
                latest_user_text=latest_user_text,
                started_at=started_at,
            )

        # WEB/GENERAL 的短问题兜底规则：优先走单兵，避免误判成复杂 DAG。
        if _looks_like_weather_request(latest_user_text) and (not _is_weather_actionable_clause(latest_user_text)):
            return IntentPolicy._finalize_intent_decision(
                intent="CHAT",
                confidence=max(domain_conf, 0.9),
                is_complex=False,
                source="rule_weather_smalltalk",
                direct_answer="",
                session_id=session_id,
                latest_user_text=latest_user_text,
                started_at=started_at,
            )
        if _looks_like_weather_request(latest_user_text):
            return IntentPolicy._finalize_intent_decision(
                intent="weather_agent",
                confidence=max(domain_conf, 0.9),
                is_complex=False,
                source="rule_weather_fastpath",
                direct_answer="",
                session_id=session_id,
                latest_user_text=latest_user_text,
                started_at=started_at,
            )
        if data_domain == "GENERAL" and _looks_like_search_request(latest_user_text):
            return IntentPolicy._finalize_intent_decision(
                intent="search_agent",
                confidence=max(domain_conf, 0.88),
                is_complex=False,
                source="rule_search_fastpath",
                direct_answer="",
                session_id=session_id,
                latest_user_text=latest_user_text,
                started_at=started_at,
            )

        # 规则快路由：医疗/代码问题优先落到垂直 Agent（若已注册）。
        if _looks_like_medical_request(latest_user_text) and "medical_agent" in MEMBERS:
            return IntentPolicy._finalize_intent_decision(
                intent="medical_agent",
                confidence=max(domain_conf, 0.9),
                is_complex=False,
                source="rule_medical_fastpath",
                direct_answer="",
                session_id=session_id,
                latest_user_text=latest_user_text,
                started_at=started_at,
            )
        if _looks_like_code_request(latest_user_text) and "code_agent" in MEMBERS:
            return IntentPolicy._finalize_intent_decision(
                intent="code_agent",
                confidence=max(domain_conf, 0.9),
                is_complex=False,
                source="rule_code_fastpath",
                direct_answer="",
                session_id=session_id,
                latest_user_text=latest_user_text,
                started_at=started_at,
            )

        # 业务域优先路由：Holter/云柚相关查询，优先进入 yunyou_agent。
        # 注意：必须放在 SQL fast-path 之前，否则会被“order by/limit/数据库”误路由到 sql_agent。
        if _looks_like_holter_request(latest_user_text):
            log.info("Intent fast-path: 命中 Holter/云柚业务域，直接路由 yunyou_agent")
            return IntentPolicy._finalize_intent_decision(
                intent="yunyou_agent",
                confidence=0.96,
                is_complex=False,
                source="rule_holter",
                direct_answer="",
                session_id=session_id,
                latest_user_text=latest_user_text,
                started_at=started_at,
            )

        # SQL 快速路由：用户明确表达了 SQL/排序/TopN 查询诉求时，优先进入 sql_agent。
        # 业务上“按 id 倒序/前 N 条/order by/limit”这类语句通常是直接查库意图。
        if _looks_like_sql_request(latest_user_text):
            log.info("Intent fast-path: 命中 SQL 语义特征，直接路由 sql_agent")
            return IntentPolicy._finalize_intent_decision(
                intent="sql_agent",
                confidence=0.95,
                is_complex=False,
                source="rule_sql",
                direct_answer="",
                session_id=session_id,
                latest_user_text=latest_user_text,
                started_at=started_at,
            )

        # GENERAL 快速通道：纯闲聊默认直达 CHAT，避免慢速 LLM 分类。
        if (
                data_domain == "GENERAL"
                and ROUTER_POLICY_CONFIG.general_chat_fastpath_enabled
                and _looks_like_general_chat_request(latest_user_text)
        ):
            direct_complex = _looks_like_compound_request(latest_user_text)
            return IntentPolicy._finalize_intent_decision(
                intent="CHAT",
                confidence=0.92,
                is_complex=direct_complex,
                source="rule_general_chat_fastpath",
                direct_answer="",
                session_id=session_id,
                latest_user_text=latest_user_text,
                started_at=started_at,
            )

        # 默认不走 LLM 意图分类，直接按规则兜底。
        if not ROUTER_POLICY_CONFIG.intent_llm_fallback_enabled:
            direct_complex = _looks_like_compound_request(latest_user_text)
            return IntentPolicy._finalize_intent_decision(
                intent="CHAT",
                confidence=0.85,
                is_complex=direct_complex,
                source="rule_default_chat",
                direct_answer="",
                session_id=session_id,
                latest_user_text=latest_user_text,
                started_at=started_at,
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", IntentRouterPrompt.get_system_prompt()),
            MessagesPlaceholder(variable_name="messages"),
        ])

        decision_source = "llm"

        try:
            structured = await _invoke_structured_output_with_fallback(
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

        if decision.intent == "CHAT" and strict_tool_intent in IntentPolicy._CHAT_BLOCKED_TOOL_INTENTS:
            log.info(
                "Intent chat override: 命中强工具意图，覆盖 CHAT "
                f"(tool={strict_tool_intent}, original_source={decision_source})"
            )
            decision.intent = strict_tool_intent
            decision.confidence = max(
                float(decision.confidence or 0.0),
                IntentPolicy._TOOL_INTENT_CONFIDENCE.get(strict_tool_intent, 0.94),
            )
            decision.is_complex = False
            decision.direct_answer = ""
            decision_source = f"{decision_source}_tool_override"

        return IntentPolicy._finalize_intent_decision(
            intent=decision.intent,
            confidence=decision.confidence,
            is_complex=decision.is_complex,
            source=decision_source,
            direct_answer=decision.direct_answer,
            session_id=session_id,
            latest_user_text=latest_user_text,
            started_at=started_at,
        )
