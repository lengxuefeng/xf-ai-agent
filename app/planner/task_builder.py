import re
from typing import Any, Callable, Dict, List, Optional, Sequence
from config.constants.supervisor_keywords import SUPERVISOR_SUMMARY_HINTS
from common.enums.agent_enum import AGENT_MEMBERS_TUPLE, AgentTypeEnum
def extract_agent_focus_text(
        agent_name: str,
        clause_text: str,
        full_text: str,
        *,
        collect_intent_signals_fn: Callable[[str], List[str]],
        dedupe_keep_order_fn: Callable[[Sequence[str]], List[str]],
) -> str:
    """
    从复合子句里筛出更贴近目标 Agent 的那部分片段。

    设计要点：
    1. 按连接词拆分子句
    2. 检查每个片段是否命中目标Agent
    3. 返回命中的片段或完整子句

    设计理由：
    1. 复合子句可能包含多个Agent的内容
    2. 需要提取与目标Agent相关的部分
    3. 提高任务输入的准确性

    拆分策略：
    - 连接词：和、并且、以及、然后、再
    - 标点符号：，、；;

    Args:
        agent_name: 目标Agent名称
        clause_text: 子句文本
        full_text: 完整文本
        collect_intent_signals_fn: 收集意图信号的函数
        dedupe_keep_order_fn: 去重函数

    Returns:
        str: 提取的Agent相关文本

    示例：
    >>> extract_agent_focus_text("weather_agent", "郑州天气和北京天气", "郑州天气和北京天气", ...)
    "郑州天气"

    使用场景：
    - 任务拆解时优化任务输入
    - 提高Agent执行的准确性
    """
    normalized_clause = clause_text.strip() or full_text.strip()
    if not normalized_clause:
        return full_text.strip()

    # 1. 按连接词拆分
    segment_parts = [
        item.strip(" ,，;；\n\t")
        for item in re.split(r"(?:和|并且|以及|然后|再|,|，|;|；)", normalized_clause)
    ]
    segment_parts = [item for item in segment_parts if item]
    if not segment_parts:
        return normalized_clause

    # 2. 检查每个片段是否命中目标Agent
    focused_segments: List[str] = []
    for part in segment_parts:
        if agent_name in collect_intent_signals_fn(part):
            focused_segments.append(part)
    focused_segments = dedupe_keep_order_fn(focused_segments)

    # 3. 返回命中的片段或完整子句
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
    """
    根据目标 Agent 包装出更明确、更自包含的任务输入。

    设计要点：
    1. 提取Agent相关的文本
    2. 为每个Agent添加特定的提示
    3. 限制Agent的输出范围，避免跑题

    设计理由：
    1. 明确的任务边界，提高执行准确性
    2. 特定的提示，减少幻觉和错误
    3. 自包含的任务输入，不依赖上下文

    提示策略：
    - 天气Agent：只输出天气结论和出行建议
    - 搜索Agent：只输出用户明确要求的检索结果
    - 云柚Agent：只输出Holter数据查询结果
    - SQL Agent：只处理SQL查询相关诉求
    - 医疗Agent：只处理医疗健康分析相关诉求
    - 代码Agent：只处理代码开发相关诉求

    Args:
        agent_name: 目标Agent名称
        clause_text: 子句文本
        full_text: 完整文本
        extract_agent_focus_text_fn: 提取Agent相关文本的函数

    Returns:
        str: 包装后的任务输入

    示例：
    >>> build_agent_specific_task_input("weather_agent", "郑州天气", "郑州天气怎么样", ...)
    "你是天气执行子任务，仅允许输出天气结论与出行建议。不要输出旅游路线、景点攻略或与天气无关内容。\n用户子任务：郑州天气"

    使用场景：
    - 任务拆解时为每个Agent构造任务输入
    - 提高Agent执行的准确性和可控性
    """
    normalized_clause = extract_agent_focus_text_fn(agent_name, clause_text, full_text)

    # 为每个Agent添加特定的提示
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
    """
    基于规则把复合输入拆成可执行任务列表，尽量避免回退到慢规划。

    设计要点：
    1. 纯规则匹配，不调用LLM
    2. 支持多种路由策略
    3. 智能合并相同Agent的子句
    4. 支持依赖关系的任务编排

    设计理由：
    1. 零LLM成本，快速响应
    2. 可解释性强，容易调试
    3. 兜底能力，LLM不可用时仍可工作

    任务结构：
    {
        "id": 任务ID（t1, t2, ...）
        "agent": Agent名称
        "input": 任务输入
        "depends_on": 依赖的任务ID列表
        "status": 任务状态
        "result": 任务结果（初始为None）
    }

    Args:
        user_text: 用户输入的文本
        candidate_agents: 候选Agent列表
        route_strategy: 路由策略
        split_query_clauses_fn: 拆分子句的函数
        collect_intent_signals_fn: 收集意图信号的函数
        is_explicit_request_clause_fn: 判断显式请求子句的函数
        select_primary_agent_for_clause_fn: 选择主Agent的函数
        build_agent_specific_task_input_fn: 构建任务输入的函数
        dedupe_keep_order_fn: 去重函数
        has_dependency_hint_fn: 判断依赖提示的函数
        route_strategy_single: 单域路由策略
        route_strategy_complex_single: 复杂单域路由策略
        route_strategy_multi_split: 多域拆分路由策略
        pending_status: 待处理状态

    Returns:
        Optional[List[Dict[str, Any]]]: 任务列表，无法拆解则返回None

    示例：
    >>> build_rule_based_multidomain_tasks("先查天气，再查景点", ...)
    [
        {"id": "t1", "agent": "weather_agent", "input": "...", "depends_on": [], "status": "pending", "result": None},
        {"id": "t2", "agent": "search_agent", "input": "...", "depends_on": ["t1"], "status": "pending", "result": None}
    ]

    使用场景：
    - 规则规划器的核心逻辑
    - 任务拆解的主要实现
    """
    normalized_text = (user_text or "").strip()
    if not normalized_text:
        return None

    # 1. 确定候选Agent列表
    fallback_candidates = dedupe_keep_order_fn(candidate_agents or collect_intent_signals_fn(normalized_text))

    # 2. 处理复杂单域策略（单一Agent但需要规划）
    if route_strategy == route_strategy_complex_single and len(fallback_candidates) == 1:
        primary_agent = fallback_candidates[0]

        # 创建第一个任务（主要Agent）
        first_task: Dict[str, Any] = {
            "id": "t1",
            "agent": primary_agent,
            "input": build_agent_specific_task_input_fn(primary_agent, normalized_text, normalized_text),
            "depends_on": [],
            "status": pending_status,
            "result": None,
        }

        # 检查是否需要Chat总结
        needs_chat_summary = any(hint in normalized_text for hint in SUPERVISOR_SUMMARY_HINTS)
        if not needs_chat_summary:
            return [first_task]

        # 创建第二个任务（Chat总结）
        second_task: Dict[str, Any] = {
            "id": "t2",
            "agent": "CHAT",
            "input": f"基于 t1 的执行结果，给出用户需要的最终建议与结论：{normalized_text}",
            "depends_on": ["t1"],
            "status": pending_status,
            "result": None,
        }
        return [first_task, second_task]

    # 3. 如果不是多域拆分且候选Agent少于2个，返回None
    if route_strategy != route_strategy_multi_split and len(fallback_candidates) < 2:
        return None

    # 4. 拆分子句，为每个子句选择Agent
    clauses = split_query_clauses_fn(normalized_text) or [normalized_text]
    clause_request_flags = [is_explicit_request_clause_fn(clause) for clause in clauses]
    has_explicit_clause = any(clause_request_flags)

    # 5. 为每个子句选择Agent
    clause_pairs: List[tuple[str, str]] = []
    for idx, clause in enumerate(clauses):
        # 跳过非显式请求的子句（当有显式子句时）
        if has_explicit_clause and (not clause_request_flags[idx]):
            continue

        # 检查子句命中的Agent
        clause_candidates = collect_intent_signals_fn(clause)
        if clause_request_flags[idx] and len(clause_candidates) >= 2:
            # 显式子句命中多个Agent，全部添加
            for candidate_name in clause_candidates:
                clause_pairs.append((candidate_name, clause))
            continue

        if clause_request_flags[idx]:
            # 显式子句命中一个Agent
            if clause_candidates:
                clause_pairs.append((clause_candidates[0], clause))
            else:
                clause_pairs.append(("CHAT", clause))
            continue

        # 非显式子句，从fallback中选择
        selected_agent = select_primary_agent_for_clause_fn(clause, fallback_candidates)
        if selected_agent:
            clause_pairs.append((selected_agent, clause))

    # 6. 如果没有确定任何Agent，使用fallback
    if not clause_pairs:
        full_agent = select_primary_agent_for_clause_fn(normalized_text, fallback_candidates)
        if full_agent:
            clause_pairs = [(full_agent, normalized_text)]
        else:
            clause_pairs = [(agent_name, normalized_text) for agent_name in fallback_candidates]

    # 7. 合并相同Agent的连续子句
    merged_pairs: List[tuple[str, str]] = []
    for agent_name, clause in clause_pairs:
        if merged_pairs and merged_pairs[-1][0] == agent_name:
            # 合并相同Agent的子句
            merged_agent, merged_clause = merged_pairs[-1]
            merged_pairs[-1] = (merged_agent, f"{merged_clause}；{clause}")
        else:
            merged_pairs.append((agent_name, clause))

    # 8. 特殊处理：如果只有一个任务且有多个候选，使用第一个
    if len(merged_pairs) < 2 and len(fallback_candidates) >= 2 and (not has_explicit_clause):
        merged_pairs = [(fallback_candidates[0], normalized_text)]

    # 9. 确保至少有一个任务
    if len(merged_pairs) < 1:
        return None

    # 10. 构建任务列表，考虑依赖关系
    sequential_mode = has_dependency_hint_fn(normalized_text)
    task_list: List[Dict[str, Any]] = []
    previous_task_id: Optional[str] = None
    for index, (agent_name, clause) in enumerate(merged_pairs, start=1):
        task_id = f"t{index}"
        # 顺序模式时，依赖上一个任务
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
    """
    为 Planner 不可用时构造稳定可解释的确定性兜底任务。

    设计要点：
    1. Planner不可用时的兜底方案
    2. 确定性的任务构造，结果可复现
    3. 支持多种路由策略
    4. 优先级排序，保证输出稳定

    设计理由：
    1. 确保系统在任何情况下都能工作
    2. 兜底方案简单可靠，便于调试
    3. 保持与主逻辑一致的行为

    Args:
        user_text: 用户输入的文本
        intent_candidates: 意图候选列表
        route_strategy: 路由策略
        fallback_intent: 兜底意图
        build_agent_specific_task_input_fn: 构建任务输入的函数
        has_dependency_hint_fn: 判断依赖提示的函数
        route_strategy_single: 单域路由策略
        route_strategy_multi_split: 多域拆分路由策略
        multi_domain_agent_priority: Agent优先级列表
        members: Agent成员列表
        pending_status: 待处理状态

    Returns:
        List[Dict[str, Any]]: 任务列表

    示例：
    >>> build_planner_fallback_tasks(user_text="郑州天气", intent_candidates=["weather_agent"], ...)
    [{"id": "t1", "agent": "weather_agent", "input": "...", "depends_on": [], "status": "pending", "result": None}]

    使用场景：
    - Planner不可用时的兜底方案
    - 测试和调试时的确定性输出
    """
    normalized_text = (user_text or "").strip() or "请基于当前会话给出可执行结论。"

    # 1. 去重候选Agent，按优先级排序
    deduped_candidates: List[str] = []
    for candidate_name in intent_candidates:
        if (
                candidate_name in multi_domain_agent_priority
                or candidate_name == "CHAT"
        ) and candidate_name not in deduped_candidates:
            deduped_candidates.append(candidate_name)

    # 2. 单域策略：只有一个候选Agent
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

    # 3. 多域拆分策略：多个候选Agent
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

    # 4. 其他情况：使用第一个候选Agent或兜底意图
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
        needs_chat_summary = any(hint in normalized_text for hint in SUPERVISOR_SUMMARY_HINTS)
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


