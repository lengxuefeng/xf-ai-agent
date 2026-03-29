import re
from typing import Any, Callable, Dict, List, Optional, Sequence


def dedupe_keep_order(values: Sequence[str]) -> List[str]:
    """按首次出现顺序去重，保证规则规划阶段输出稳定。"""
    deduped: List[str] = []
    for value_item in values:
        if value_item not in deduped:
            deduped.append(value_item)
    return deduped


def has_dependency_hint(text: str) -> bool:
    """识别“先做 A 再做 B”这类明显带有先后依赖的表达。"""
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


def is_explicit_request_clause(text: str, *, request_action_hints: Sequence[str]) -> bool:
    """判断一个子句是否明确要求系统执行动作，而不是单纯背景描述。"""
    normalized_text = (text or "").strip().lower()
    if not normalized_text:
        return False
    if any(mark in normalized_text for mark in ("?", "？")):
        return True
    if normalized_text.endswith(("吗", "么", "呢")):
        return True

    interrogative_patterns = (
        r"^(怎么|如何|怎样|为何|为什么|能否|可否|是否|要不要|有没有|需不需要)",
        r"(怎么|如何|怎样|为何|为什么|能否|可否|是否|要不要|有没有|需不需要).{0,12}(做|办|处理|操作|安排|推进|落地|实现|解决)",
    )
    if any(re.search(pattern, normalized_text) for pattern in interrogative_patterns):
        return True
    return any(hint in normalized_text for hint in request_action_hints)


def split_query_clauses(user_text: str) -> List[str]:
    """把复合问题拆成可供规则规划器消费的子句列表。"""
    normalized_text = (user_text or "").strip()
    if not normalized_text:
        return []

    normalized_text = re.sub(r"[，,；;。！？?!]+", "|", normalized_text)
    normalized_text = re.sub(
        r"(并且|而且|同时|然后|再帮我|顺便|另外|以及)",
        "|",
        normalized_text,
    )
    clauses = [item.strip(" ,，\n\t") for item in normalized_text.split("|")]
    return [item for item in clauses if item]


def collect_intent_signals(
    text: str,
    *,
    looks_like_holter_request: Callable[[str], bool],
    looks_like_sql_request: Callable[[str], bool],
    looks_like_weather_request: Callable[[str], bool],
    looks_like_search_request: Callable[[str], bool],
    looks_like_medical_request: Callable[[str], bool],
    looks_like_code_request: Callable[[str], bool],
    is_weather_actionable_clause: Callable[[str], bool],
    is_search_actionable_clause: Callable[[str], bool],
    multi_domain_agent_priority: Sequence[str],
) -> List[str]:
    """汇总一条输入里命中的业务意图，并按统一优先级输出。"""
    normalized_text = (text or "").strip().lower()
    if not normalized_text:
        return []

    signals: List[str] = []
    holter_hit = looks_like_holter_request(normalized_text)
    sql_hit = looks_like_sql_request(normalized_text)
    weather_hit = looks_like_weather_request(normalized_text)
    search_hit = looks_like_search_request(normalized_text)
    medical_hit = looks_like_medical_request(normalized_text)
    code_hit = looks_like_code_request(normalized_text)

    if holter_hit:
        signals.append("yunyou_agent")
    if sql_hit:
        signals.append("sql_agent")

    weather_is_explicit = is_weather_actionable_clause(normalized_text)
    strong_business_present = holter_hit or sql_hit or medical_hit or code_hit
    if weather_hit and (weather_is_explicit or (not strong_business_present)):
        signals.append("weather_agent")

    search_is_explicit = is_search_actionable_clause(normalized_text)
    if search_hit and (search_is_explicit or (not strong_business_present)):
        signals.append("search_agent")
    if medical_hit:
        signals.append("medical_agent")
    if code_hit:
        signals.append("code_agent")

    ordered_unique: List[str] = []
    for candidate_name in multi_domain_agent_priority:
        if candidate_name in signals and candidate_name not in ordered_unique:
            ordered_unique.append(candidate_name)
    return ordered_unique


def analyze_request_payload(
    user_text: str,
    *,
    split_query_clauses_fn: Callable[[str], List[str]],
    collect_intent_signals_fn: Callable[[str], List[str]],
    is_explicit_request_clause_fn: Callable[[str], bool],
    is_weather_actionable_clause_fn: Callable[[str], bool],
    is_search_actionable_clause_fn: Callable[[str], bool],
    has_dependency_hint_fn: Callable[[str], bool],
    dedupe_keep_order_fn: Callable[[Sequence[str]], List[str]],
    agent_domain_map: Dict[str, str],
    route_strategy_single: str,
    route_strategy_complex_single: str,
    route_strategy_multi_split: str,
) -> Dict[str, Any]:
    """统一分析请求复杂度与候选意图，供 Domain/Intent/Planner 共享复用。"""
    normalized_text = (user_text or "").strip().lower()
    if not normalized_text:
        return {}

    clauses = split_query_clauses_fn(normalized_text) or [normalized_text]
    candidate_agents = collect_intent_signals_fn(normalized_text)

    explicit_clause_agents: List[str] = []
    for clause in clauses:
        if not is_explicit_request_clause_fn(clause):
            continue
        clause_agents = collect_intent_signals_fn(clause)
        if clause_agents:
            explicit_clause_agents.extend(clause_agents)
        else:
            explicit_clause_agents.append("CHAT")
    explicit_clause_agents = dedupe_keep_order_fn(explicit_clause_agents)
    if explicit_clause_agents:
        candidate_agents = explicit_clause_agents

    strong_agents = {"yunyou_agent", "sql_agent", "medical_agent", "code_agent"}
    if any(agent_name in strong_agents for agent_name in candidate_agents):
        filtered_candidates = list(candidate_agents)
        if "weather_agent" in filtered_candidates:
            weather_explicit = any(is_weather_actionable_clause_fn(clause) for clause in clauses)
            if not weather_explicit:
                filtered_candidates = [
                    agent_name for agent_name in filtered_candidates if agent_name != "weather_agent"
                ]
        if "search_agent" in filtered_candidates:
            search_explicit = any(is_search_actionable_clause_fn(clause) for clause in clauses)
            if not search_explicit:
                filtered_candidates = [
                    agent_name for agent_name in filtered_candidates if agent_name != "search_agent"
                ]
        candidate_agents = filtered_candidates

    candidate_domains = dedupe_keep_order_fn(
        [agent_domain_map.get(agent_name, "GENERAL") for agent_name in candidate_agents]
    )
    multi_intent = len(candidate_agents) >= 2
    multi_domain = len([domain for domain in candidate_domains if domain != "GENERAL"]) >= 2
    dependency_hint = has_dependency_hint_fn(normalized_text)

    if multi_intent:
        return {
            "candidate_agents": candidate_agents,
            "candidate_domains": candidate_domains,
            "is_multi_intent": True,
            "is_multi_domain": multi_domain,
            "has_dependency_hint": dependency_hint,
            "route_strategy": route_strategy_multi_split,
            "reason": "multiple_intents_detected",
        }

    if dependency_hint:
        return {
            "candidate_agents": candidate_agents,
            "candidate_domains": candidate_domains or ["GENERAL"],
            "is_multi_intent": False,
            "is_multi_domain": False,
            "has_dependency_hint": True,
            "route_strategy": route_strategy_complex_single,
            "reason": "dependency_hint_detected",
        }

    return {
        "candidate_agents": candidate_agents,
        "candidate_domains": candidate_domains or ["GENERAL"],
        "is_multi_intent": False,
        "is_multi_domain": False,
        "has_dependency_hint": False,
        "route_strategy": route_strategy_single,
        "reason": "single_domain",
    }


def select_primary_agent_for_clause(
    clause_text: str,
    fallback_candidates: Sequence[str],
    *,
    collect_intent_signals_fn: Callable[[str], List[str]],
    multi_domain_agent_priority: Sequence[str],
) -> Optional[str]:
    """为单个子句选最合适的执行 Agent，优先信任子句内显式命中。"""
    clause_candidates = collect_intent_signals_fn(clause_text)
    if clause_candidates:
        return clause_candidates[0]
    for candidate_name in fallback_candidates:
        if candidate_name in multi_domain_agent_priority:
            return candidate_name
    return None


def extract_agent_focus_text(
    agent_name: str,
    clause_text: str,
    full_text: str,
    *,
    collect_intent_signals_fn: Callable[[str], List[str]],
    dedupe_keep_order_fn: Callable[[Sequence[str]], List[str]],
) -> str:
    """从复合子句里筛出更贴近目标 Agent 的那部分片段。"""
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
        if agent_name in collect_intent_signals_fn(part):
            focused_segments.append(part)
    focused_segments = dedupe_keep_order_fn(focused_segments)
    if focused_segments:
        return "；".join(focused_segments)
    return normalized_clause


def build_agent_specific_task_input(
    agent_name: str,
    clause_text: str,
    full_text: str,
    *,
    extract_agent_focus_text_fn: Callable[[str, str, str], str],
) -> str:
    """根据目标 Agent 包装出更明确、更自包含的任务输入。"""
    normalized_clause = extract_agent_focus_text_fn(agent_name, clause_text, full_text)
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


def build_rule_based_multidomain_tasks(
    user_text: str,
    *,
    candidate_agents: Optional[List[str]],
    route_strategy: str,
    split_query_clauses_fn: Callable[[str], List[str]],
    collect_intent_signals_fn: Callable[[str], List[str]],
    is_explicit_request_clause_fn: Callable[[str], bool],
    select_primary_agent_for_clause_fn: Callable[[str, Sequence[str]], Optional[str]],
    build_agent_specific_task_input_fn: Callable[[str, str, str], str],
    dedupe_keep_order_fn: Callable[[Sequence[str]], List[str]],
    has_dependency_hint_fn: Callable[[str], bool],
    route_strategy_single: str,
    route_strategy_complex_single: str,
    route_strategy_multi_split: str,
    pending_status: str,
) -> Optional[List[Dict[str, Any]]]:
    """基于规则把复合输入拆成可执行任务列表，尽量避免回退到慢规划。"""
    normalized_text = (user_text or "").strip()
    if not normalized_text:
        return None

    fallback_candidates = dedupe_keep_order_fn(candidate_agents or collect_intent_signals_fn(normalized_text))

    if route_strategy == route_strategy_complex_single and len(fallback_candidates) == 1:
        primary_agent = fallback_candidates[0]
        first_task: Dict[str, Any] = {
            "id": "t1",
            "agent": primary_agent,
            "input": build_agent_specific_task_input_fn(primary_agent, normalized_text, normalized_text),
            "depends_on": [],
            "status": pending_status,
            "result": None,
        }
        summary_hints = ("总结", "分析", "解释", "建议", "报告", "结论", "对比", "方案", "归纳")
        needs_chat_summary = any(hint in normalized_text for hint in summary_hints)
        if not needs_chat_summary:
            return [first_task]

        second_task: Dict[str, Any] = {
            "id": "t2",
            "agent": "CHAT",
            "input": f"基于 t1 的执行结果，给出用户需要的最终建议与结论：{normalized_text}",
            "depends_on": ["t1"],
            "status": pending_status,
            "result": None,
        }
        return [first_task, second_task]

    if route_strategy != route_strategy_multi_split and len(fallback_candidates) < 2:
        return None

    clauses = split_query_clauses_fn(normalized_text) or [normalized_text]
    clause_request_flags = [is_explicit_request_clause_fn(clause) for clause in clauses]
    has_explicit_clause = any(clause_request_flags)

    clause_pairs: List[tuple[str, str]] = []
    for idx, clause in enumerate(clauses):
        if has_explicit_clause and (not clause_request_flags[idx]):
            continue

        clause_candidates = collect_intent_signals_fn(clause)
        if clause_request_flags[idx] and len(clause_candidates) >= 2:
            for candidate_name in clause_candidates:
                clause_pairs.append((candidate_name, clause))
            continue
        if clause_request_flags[idx]:
            if clause_candidates:
                clause_pairs.append((clause_candidates[0], clause))
            else:
                clause_pairs.append(("CHAT", clause))
            continue

        selected_agent = select_primary_agent_for_clause_fn(clause, fallback_candidates)
        if selected_agent:
            clause_pairs.append((selected_agent, clause))

    if not clause_pairs:
        full_agent = select_primary_agent_for_clause_fn(normalized_text, fallback_candidates)
        if full_agent:
            clause_pairs = [(full_agent, normalized_text)]
        else:
            clause_pairs = [(agent_name, normalized_text) for agent_name in fallback_candidates]

    merged_pairs: List[tuple[str, str]] = []
    for agent_name, clause in clause_pairs:
        if merged_pairs and merged_pairs[-1][0] == agent_name:
            merged_agent, merged_clause = merged_pairs[-1]
            merged_pairs[-1] = (merged_agent, f"{merged_clause}；{clause}")
        else:
            merged_pairs.append((agent_name, clause))

    if len(merged_pairs) < 2 and len(fallback_candidates) >= 2 and (not has_explicit_clause):
        merged_pairs = [(fallback_candidates[0], normalized_text)]

    if len(merged_pairs) < 1:
        return None

    sequential_mode = has_dependency_hint_fn(normalized_text)
    task_list: List[Dict[str, Any]] = []
    previous_task_id: Optional[str] = None
    for index, (agent_name, clause) in enumerate(merged_pairs, start=1):
        task_id = f"t{index}"
        depends_on = [previous_task_id] if (sequential_mode and previous_task_id) else []
        task_list.append(
            {
                "id": task_id,
                "agent": agent_name,
                "input": build_agent_specific_task_input_fn(agent_name, clause, normalized_text),
                "depends_on": depends_on,
                "status": pending_status,
                "result": None,
            }
        )
        previous_task_id = task_id
    return task_list


def build_planner_fallback_tasks(
    *,
    user_text: str,
    intent_candidates: List[str],
    route_strategy: str,
    fallback_intent: str,
    build_agent_specific_task_input_fn: Callable[[str, str, str], str],
    has_dependency_hint_fn: Callable[[str], bool],
    route_strategy_single: str,
    route_strategy_multi_split: str,
    multi_domain_agent_priority: Sequence[str],
    members: Sequence[str],
    pending_status: str,
) -> List[Dict[str, Any]]:
    """为 Planner 不可用时构造稳定可解释的确定性兜底任务。"""
    normalized_text = (user_text or "").strip() or "请基于当前会话给出可执行结论。"

    deduped_candidates: List[str] = []
    for candidate_name in intent_candidates:
        if (
            candidate_name in multi_domain_agent_priority
            or candidate_name == "CHAT"
        ) and candidate_name not in deduped_candidates:
            deduped_candidates.append(candidate_name)

    if route_strategy == route_strategy_single and len(deduped_candidates) == 1:
        only_agent = deduped_candidates[0]
        return [
            {
                "id": "t1",
                "agent": only_agent,
                "input": build_agent_specific_task_input_fn(only_agent, normalized_text, normalized_text),
                "depends_on": [],
                "status": pending_status,
                "result": None,
            }
        ]

    if route_strategy == route_strategy_multi_split and len(deduped_candidates) >= 2:
        sequential_mode = has_dependency_hint_fn(normalized_text)
        fallback_tasks: List[Dict[str, Any]] = []
        previous_task_id: Optional[str] = None
        for index, agent_name in enumerate(deduped_candidates, start=1):
            task_id = f"t{index}"
            depends_on = [previous_task_id] if (sequential_mode and previous_task_id) else []
            fallback_tasks.append(
                {
                    "id": task_id,
                    "agent": agent_name,
                    "input": build_agent_specific_task_input_fn(agent_name, normalized_text, normalized_text),
                    "depends_on": depends_on,
                    "status": pending_status,
                    "result": None,
                }
            )
            previous_task_id = task_id
        return fallback_tasks

    if deduped_candidates:
        primary_agent = deduped_candidates[0]
    elif fallback_intent in members:
        primary_agent = fallback_intent
    else:
        primary_agent = "CHAT"

    return [
        {
            "id": "t1",
            "agent": primary_agent,
            "input": build_agent_specific_task_input_fn(primary_agent, normalized_text, normalized_text),
            "depends_on": [],
            "status": pending_status,
            "result": None,
        }
    ]


def looks_like_compound_request(
    text: str,
    *,
    analyze_request_fn: Callable[[str], Any],
    route_strategy_complex_single: str,
    route_strategy_multi_split: str,
    complex_connector_hints: Sequence[str],
) -> bool:
    """判断输入是否更适合走复合任务编排，而不是单轮直接回答。"""
    analysis = analyze_request_fn(text)
    route_strategy = getattr(analysis, "route_strategy", None) or (
        analysis.get("route_strategy") if isinstance(analysis, dict) else None
    )
    if route_strategy in {route_strategy_multi_split, route_strategy_complex_single}:
        return True

    normalized_text = (text or "").strip().lower()
    if not normalized_text:
        return False
    if len(re.findall(r"[?？]", normalized_text)) >= 2:
        return True
    if any(token in normalized_text for token in complex_connector_hints):
        return len(normalized_text) >= 30
    return False
