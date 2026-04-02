# -*- coding: utf-8 -*-
"""
Supervisor 规则支持模块（Supervisor Rule Support）。

提供基于规则的前置匹配能力，用于快速路由和任务拆解。
这是Supervisor的Zero-LLM层，在调用昂贵的LLM之前先进行规则匹配。

设计要点：
1. 零LLM成本：纯规则匹配，不调用LLM
2. 快速响应：规则匹配速度极快，用户体验好
3. 可解释性强：规则明确，容易理解和调试
4. 兜底能力：LLM不可用时，规则层仍可工作

使用场景：
- 快速路由：根据关键词快速选择Agent
- 任务拆解：根据连接词和依赖提示拆解复合请求
- 规则降级：LLM不可用时使用规则作为兜底

架构说明：
- 本模块提供纯规则匹配能力
- supervisor.py结合规则和LLM，提供智能路由
- 规则层和LLM层互补，既保证速度又保证准确性
"""
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from config.constants.supervisor_keywords import (
    SUPERVISOR_CLAUSE_CONNECTOR_PATTERNS,
    SUPERVISOR_SEQUENTIAL_HINT_PAIRS,
    SUPERVISOR_SUMMARY_HINTS,
)
from common.enums.agent_enum import AGENT_MEMBERS_TUPLE, AgentTypeEnum


def dedupe_keep_order(values: Sequence[str]) -> List[str]:
    """
    按首次出现顺序去重，保证规则规划阶段输出稳定。

    设计要点：
    1. 保留首次出现的元素
    2. 保持原始顺序
    3. 去除重复元素

    设计理由：
    1. 规则匹配的结果可能是重复的，需要去重
    2. 保留顺序，保证输出的一致性
    3. 便于调试和复现

    Args:
        values: 要去重的字符串序列

    Returns:
        List[str]: 去重后的列表，保持原始顺序

    示例：
    >>> dedupe_keep_order(["a", "b", "a", "c", "b"])
    ["a", "b", "c"]

    使用场景：
    - 规则匹配后去重候选Agent
    - 任务拆解后去重候选任务
    """
    deduped: List[str] = []
    for value_item in values:
        if value_item not in deduped:
            deduped.append(value_item)
    return deduped


def has_dependency_hint(text: str) -> bool:
    """
    识别"先做 A 再做 B"这类明显带有先后依赖的表达。

    设计要点：
    1. 检查预定义的依赖提示词对
    2. 同时出现两个提示词，判断有依赖

    设计理由：
    1. 依赖关系影响任务拆解策略
    2. 有依赖的任务需要顺序执行
    3. 无依赖的任务可以并行执行

    支持的依赖提示：
    - 先...再...
    - 先...然后...
    - 首先...其次...
    - 之前...之后...

    Args:
        text: 要分析的文本

    Returns:
        bool: True表示有依赖提示，False表示无

    示例：
    >>> has_dependency_hint("先查天气，再查附近景点")
    True
    >>> has_dependency_hint("郑州的天气怎么样？")
    False

    使用场景：
    - 任务拆解时判断是否需要顺序执行
    - 路由策略选择
    """
    normalized_text = (text or "").strip().lower()
    if not normalized_text:
        return False

    # 检查所有预定义的依赖提示词对
    for first_hint, second_hint in SUPERVISOR_SEQUENTIAL_HINT_PAIRS:
        if first_hint in normalized_text and second_hint in normalized_text:
            return True
    return False


def is_explicit_request_clause(text: str, *, request_action_hints: Sequence[str]) -> bool:
    """
    判断一个子句是否明确要求系统执行动作，而不是单纯背景描述。

    设计要点：
    1. 检查疑问句（？、吗、么、呢）
    2. 检查疑问词（怎么、如何、为什么等）
    3. 检查查询词（什么、哪些、多少等）
    4. 检查动作提示词

    设计理由：
    1. 区分"背景描述"和"明确请求"
    2. 明确请求需要调用Agent
    3. 背景描述可以忽略或作为上下文

    判断标准：
    1. 包含疑问符号或疑问词
    2. 包含明确的查询词
    3. 包含动作提示词

    Args:
        text: 要判断的文本
        request_action_hints: 动作提示词列表

    Returns:
        bool: True表示是明确请求，False表示不是

    示例：
    >>> is_explicit_request_clause("郑州天气怎么样？", request_action_hints=["查询"])
    True
    >>> is_explicit_request_clause("我在郑州", request_action_hints=["查询"])
    False

    使用场景：
    - 子句过滤，只保留明确请求的子句
    - 任务拆解时的子句分类
    """
    # 标准化输入文本：转小写、去除首尾空格
    normalized_text = (text or "").strip().lower()
    if not normalized_text:
        return False

    # 1. 检查疑问符号
    # 疑问符号是最明确的请求信号，如"？"、"吗"结尾的句子
    # 这些符号几乎总是表示用户在提问，需要系统响应
    if any(mark in normalized_text for mark in ("?", "？")):
        return True
    if normalized_text.endswith(("吗", "么", "呢")):
        return True

    # 2. 检查疑问词模式
    # 疑问词用于询问"如何"、"为什么"等开放性问题
    # 第一个模式：匹配开头的疑问词，如"怎么"、"如何"等
    # 第二个模式：匹配疑问词后跟动词，如"怎么做"、"如何处理"等
    interrogative_patterns = (
        r"^(怎么|如何|怎样|为何|为什么|能否|可否|是否|要不要|有没有|需不需要)",
        r"(怎么|如何|怎样|为何|为什么|能否|可否|是否|要不要|有没有|需不需要).{0,12}(做|办|处理|操作|安排|推进|落地|实现|解决)",
    )
    if any(re.search(pattern, normalized_text) for pattern in interrogative_patterns):
        return True

    # 3. 检查查询词模式
    # 查询词用于询问具体事实，如时间、数量、内容等
    # 这些词通常表示用户在询问信息，需要系统查询或检索
    factoid_patterns = (
        r"(什么时间|什么时候|何时|哪天)",  # 时间查询
        r"(几号|几月几号|几月|几日|日期)",  # 日期查询
        r"(星期几|周几|礼拜几|几点)",  # 星期时间查询
        r"(有什么|有啥|哪些|多少)",  # 数量或内容查询
    )
    if any(re.search(pattern, normalized_text) for pattern in factoid_patterns):
        return True

    # 4. 检查动作提示词
    # 动作提示词由调用方传入，如"查询"、"搜索"、"执行"等
    # 这些词明确表示用户希望系统执行某个动作
    return any(hint in normalized_text for hint in request_action_hints)


def split_query_clauses(user_text: str) -> List[str]:
    """
    把复合问题拆成可供规则规划器消费的子句列表。

    设计要点：
    1. 根据标点符号和连接词拆分
    2. 清理空白的子句
    3. 保留原始语义

    设计理由：
    1. 复合问题需要拆解为多个子任务
    2. 每个子句对应一个Agent调用
    3. 提高任务拆解的准确性

    拆分策略：
    1. 标点符号：，、；。！？
    2. 连接词：和、并且、以及、然后、再
    3. 短横线：-

    Args:
        user_text: 用户输入的文本

    Returns:
        List[str]: 拆分后的子句列表

    示例：
    >>> split_query_clauses("郑州天气怎么样，有什么好玩的？")
    ["郑州天气怎么样", "有什么好玩的"]
    >>> split_query_clauses("先查天气再查景点")
    ["先查天气", "再查景点"]

    使用场景：
    - 规则规划器的输入预处理
    - 任务拆解的基础步骤
    """
    # 标准化输入文本：去除首尾空格
    normalized_text = (user_text or "").strip()
    if not normalized_text:
        return []

    # 1. 根据标点符号替换为分隔符
    # 将中文和英文的逗号、分号、句号、感叹号、问号统一替换为"|"
    # 这样可以将复合问题按标点拆分为多个子句
    # 例如："郑州天气怎么样，有什么好玩的？" -> "郑州天气怎么样|有什么好玩的？"
    normalized_text = re.sub(r"[，,；;。！？？！]+", "|", normalized_text)

    # 2. 根据连接词替换为分隔符
    # 将逻辑连接词（和、并且、以及、然后、再）替换为"|"
    # 这样可以将复合问题按逻辑关系拆分
    # 例如："先查天气和查景点" -> "先查天气|查景点"
    # 使用re.escape转义特殊字符，避免正则表达式错误
    connector_pattern = "|".join(re.escape(item) for item in SUPERVISOR_CLAUSE_CONNECTOR_PATTERNS)
    normalized_text = re.sub(rf"({connector_pattern})", "|", normalized_text)

    # 3. 分割并清理子句
    # 按"|"分割文本
    # 去除每个子句的首尾空格、逗号、换行符、制表符等
    # 过滤掉空的子句
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
    """
    汇总一条输入里命中的业务意图，并按统一优先级输出。

    设计要点：
    1. 检查各种Agent的关键词匹配
    2. 按优先级排序（云柚、SQL、天气、搜索、医疗、代码）
    3. 天气和搜索Agent需要额外的显式检查

    设计理由：
    1. 优先级反映了业务重要性
    2. 天气和搜索容易被误匹配，需要额外检查
    3. 多意图需要按优先级处理

    匹配策略：
    1. 云柚、SQL、医疗、代码：关键词匹配即可
    2. 天气：关键词匹配 + 显式动作检查（或无其他强业务）
    3. 搜索：关键词匹配 + 显式动作检查（或无其他强业务）

    Args:
        text: 要分析的文本
        looks_like_holter_request: 检查是否为云柚请求
        looks_like_sql_request: 检查是否为SQL请求
        looks_like_weather_request: 检查是否为天气请求
        looks_like_search_request: 检查是否为搜索请求
        looks_like_medical_request: 检查是否为医疗请求
        looks_like_code_request: 检查是否为代码请求
        is_weather_actionable_clause: 检查是否为显式天气动作
        is_search_actionable_clause: 检查是否为显式搜索动作
        multi_domain_agent_priority: Agent优先级列表

    Returns:
        List[str]: 按优先级排序的Agent名称列表

    示例：
    >>> collect_intent_signals("郑州天气怎么样，云柚设备数据", ...)
    ["yunyou_agent", "weather_agent"]

    使用场景：
    - 规则匹配收集候选Agent
    - 路由决策的输入
    """
    # 标准化输入文本：转小写、去除首尾空格
    normalized_text = (text or "").strip().lower()
    if not normalized_text:
        return []

    signals: List[str] = []
 
    # 1. 检查各Agent的关键词匹配
    # 这些检查会调用各Agent的looks_like_*方法
    # 每个Agent都有自己的关键词列表，通过正则或包含判断匹配
    # 云柚、SQL、医疗、代码Agent匹配即可，因为它们的业务特征明确
    # 天气和搜索Agent需要额外的显式检查，因为它们的关键词较通用
    holter_hit = looks_like_holter_request(normalized_text)
    sql_hit = looks_like_sql_request(normalized_text)
    weather_hit = looks_like_weather_request(normalized_text)
    search_hit = looks_like_search_request(normalized_text)
    medical_hit = looks_like_medical_request(normalized_text)
    code_hit = looks_like_code_request(normalized_text)
 
    # 2. 添加命中的强业务Agent
    # 云柚：Holter设备、心电报告等医疗设备相关业务
    # SQL：数据库查询、数据表等数据相关业务
    # 医疗：医疗健康、症状分析、药物咨询等专业业务
    # 代码：Python代码编写、调试、优化等开发相关业务
    # 这些Agent的业务特征明确，关键词匹配即可，不需要额外检查
    if holter_hit:
        signals.append("yunyou_agent")
    if sql_hit:
        signals.append("sql_agent")
    if medical_hit:
        signals.append("medical_agent")
    if code_hit:
        signals.append("code_agent")
 
    # 3. 添加天气Agent（需要额外检查）
    # 天气Agent的关键词较通用，容易误匹配
    # 只有以下两种情况才添加天气Agent：
    #   a) 有显式的天气动作（如"查天气"、"天气怎么样"）
    #   b) 没有其他强业务Agent（云柚、SQL、医疗、代码）命中
    # 场景："郑州的房价怎么样"不会添加天气Agent，因为有"房价"这个非天气关键词
    # 场景："郑州天气怎么样，云柚设备数据"会添加天气Agent，因为有显式天气动作
    weather_is_explicit = is_weather_actionable_clause(normalized_text)
    strong_business_present = holter_hit or sql_hit or medical_hit or code_hit
    if weather_hit and (weather_is_explicit or (not strong_business_present)):
        signals.append("weather_agent")
 
    # 4. 添加搜索Agent（需要额外检查）
    # 搜索Agent的关键词也较通用，容易误匹配
    # 只有以下两种情况才添加搜索Agent：
    #   a) 有显式的搜索动作（如"搜索"、"查一下"）
    #   b) 没有其他强业务Agent命中
    # 场景："郑州的房价怎么样"不会添加搜索Agent
    # 场景："郑州有什么好玩的，搜索一下景点"会添加搜索Agent，因为有显式搜索动作
    search_is_explicit = is_search_actionable_clause(normalized_text)
    if search_hit and (search_is_explicit or (not strong_business_present)):
        signals.append("search_agent")
 
    # 5. 按优先级排序并去重
    # 优先级顺序在multi_domain_agent_priority中定义
    # 云柚 > SQL > 天气 > 搜索 > 医疗 > 代码
    # 这个顺序反映了业务的重要性和执行成本
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
    """
    统一分析请求复杂度与候选意图，供 Domain/Intent/Planner 共享复用。

    设计要点：
    1. 拆分子句，分别分析意图
    2. 识别多意图和多域场景
    3. 判断依赖关系
    4. 选择合适的路由策略

    设计理由：
    1. 统一的分析逻辑，避免各模块重复实现
    2. 支持多种路由策略（单域、多域、复杂单域）
    3. 提供丰富的分析结果，便于调试和监控

    路由策略：
    - single_domain: 单一领域单一意图
    - multi_domain_split: 多域/多意图，需要拆分
    - complex_single_domain: 单域但有依赖，需要规划

    Args:
        user_text: 用户输入的文本
        split_query_clauses_fn: 拓分子句的函数
        collect_intent_signals_fn: 收集意图信号的函数
        is_explicit_request_clause_fn: 判断显式请求子句的函数
        is_weather_actionable_clause_fn: 判断显式天气动作的函数
        is_search_actionable_clause_fn: 判断显式搜索动作的函数
        has_dependency_hint_fn: 判断依赖提示的函数
        dedupe_keep_order_fn: 去重函数
        agent_domain_map: Agent到数据域的映射
        route_strategy_single: 单域路由策略
        route_strategy_complex_single: 复杂单域路由策略
        route_strategy_multi_split: 多域拆分路由策略

    Returns:
        Dict[str, Any]: 分析结果字典
            - candidate_agents: 候选Agent列表
            - candidate_domains: 候选数据域列表
            - is_multi_intent: 是否多意图
            - is_multi_domain: 是否多数据域
            - has_dependency_hint: 是否有依赖提示
            - route_strategy: 推荐的路由策略
            - reason: 路由策略选择的原因

    示例：
    >>> analyze_request_payload("郑州天气怎么样，有什么好玩的？", ...)
    {
        "candidate_agents": ["weather_agent", "search_agent"],
        "candidate_domains": ["WEB_SEARCH"],
        "is_multi_intent": True,
        "is_multi_domain": False,
        "has_dependency_hint": False,
        "route_strategy": "multi_domain_split",
        "reason": "multiple_intents_detected"
    }

    使用场景：
    - 路由决策的核心逻辑
    - 任务规划的前置分析
    - 调试和排错
    """
    # 标准化输入文本：转小写、去除首尾空格
    normalized_text = (user_text or "").strip().lower()
    if not normalized_text:
        return {}
 
    # 1. 拆分子句，收集候选Agent
    # 首先尝试按标点和连接词拆分子句
    # 如果无法拆分，则将整个文本作为一个子句
    # 然后收集所有命中的Agent候选
    clauses = split_query_clauses_fn(normalized_text) or [normalized_text]
    candidate_agents = collect_intent_signals_fn(normalized_text)
 
    # 2. 检查是否有显式请求的子句
    # 显式请求是指明确要求系统执行某个动作的子句
    # 例如："查询天气"、"搜索景点"等
    # 这样可以过滤掉纯描述性的背景信息，提高任务拆解的准确性
    explicit_clause_agents: List[str] = []
    for clause in clauses:
        # 检查子句是否为显式请求
        if not is_explicit_request_clause_fn(clause):
            continue  # 跳过非显式请求的子句
        
        # 检查这个子句是否命中某个Agent
        clause_agents = collect_intent_signals_fn(clause)
        if clause_agents:
            # 命中了具体Agent，全部添加到候选列表
            explicit_clause_agents.extend(clause_agents)
        else:
            # 没有命中具体Agent，默认用CHAT处理
            explicit_clause_agents.append(AgentTypeEnum.CHAT.code)
    
    # 去重并保持顺序
    explicit_clause_agents = dedupe_keep_order_fn(explicit_clause_agents)
    
    # 如果有显式请求的子句，优先使用显式Agent候选
    # 因为显式请求更准确地反映了用户的意图
    if explicit_clause_agents:
        candidate_agents = explicit_clause_agents
 
    # 3. 过滤弱匹配的天气和搜索Agent
    # 天气和搜索Agent的关键词较通用，容易误匹配
    # 只有显式明确的动作才保留，避免误路由
    # 只有当有命中的Agent时才进行过滤
    if any(agent_name in AGENT_MEMBERS_TUPLE for agent_name in candidate_agents):
        filtered_candidates = list(candidate_agents)
 
        # 3.1 检查天气Agent是否为弱匹配
        # 天气Agent只有在显式查询天气时才保留
        # 避免在"郑州的房价怎么样？"这种问题中误匹配天气Agent
        weather_agent = AgentTypeEnum.WEATHER.code
        if weather_agent in filtered_candidates:
            # 检查是否有显式的天气动作
            weather_explicit = any(is_weather_actionable_clause_fn(clause) for clause in clauses)
            if not weather_explicit:
                # 弱匹配，过滤掉天气Agent
                filtered_candidates = [
                    agent_name for agent_name in filtered_candidates if agent_name != weather_agent
                ]
 
        # 3.2 检查搜索Agent是否为弱匹配
        # 搜索Agent只有在显式搜索时才保留
        # 避免在"郑州的房价怎么样？"这种问题中误匹配搜索Agent
        search_agent = AgentTypeEnum.SEARCH.code
        if search_agent in filtered_candidates:
            # 检查是否有显式的搜索动作
            search_explicit = any(is_search_actionable_clause_fn(clause) for clause in clauses)
            if not search_explicit:
                # 弱匹配，过滤掉搜索Agent
                filtered_candidates = [
                    agent_name for agent_name in filtered_candidates if agent_name != search_agent
                ]
 
        candidate_agents = filtered_candidates
 
    # 4. 将Agent转换为对应的数据域
    # 数据域分类：YUNYOU_DB、LOCAL_DB、WEB_SEARCH、GENERAL
    # 用于判断是否跨多个数据域，影响路由策略
    candidate_domains = dedupe_keep_order_fn(
        [agent_domain_map.get(agent_name, "GENERAL") for agent_name in candidate_agents]
    )
 
    # 5. 判断请求复杂度
    # 这些复杂度指标用于选择合适的路由策略
    multi_intent = len(candidate_agents) >= 2  # 是否多意图：命中2个或更多Agent
    multi_domain = len([domain for domain in candidate_domains if domain != "GENERAL"]) >= 2  # 是否多数据域：命中2个或更多非通用域
    dependency_hint = has_dependency_hint_fn(normalized_text)  # 是否有依赖提示

    # 6. 选择路由策略
    if multi_intent:
        # 多意图，使用多域拆分策略
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
        # 有依赖提示，使用复杂单域策略
        return {
            "candidate_agents": candidate_agents,
            "candidate_domains": candidate_domains or ["GENERAL"],
            "is_multi_intent": False,
            "is_multi_domain": False,
            "has_dependency_hint": True,
            "route_strategy": route_strategy_complex_single,
            "reason": "dependency_hint_detected",
        }

    # 单意图，使用单域策略
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
    """
    为单个子句选最合适的执行 Agent，优先信任子句内显式命中。

    设计要点：
    1. 优先使用子句内显式命中的Agent
    2. 没有显式命中则按优先级从fallback中选择
    3. 返回None表示无法确定Agent

    设计理由：
    1. 子句内的显式命中最准确
    2. fallback按优先级保证输出稳定
    3. None表示无法确定，需要其他机制处理

    Args:
        clause_text: 子句文本
        fallback_candidates: 候选Agent列表
        collect_intent_signals_fn: 收集意图信号的函数
        multi_domain_agent_priority: Agent优先级列表

    Returns:
        Optional[str]: 选定的Agent名称，无法确定则返回None

    示例：
    >>> select_primary_agent_for_clause("郑州天气", ["weather_agent", "search_agent"], ...)
    "weather_agent"

    使用场景：
    - 任务拆解时为每个子句选择Agent
    - 规则规划器的核心逻辑
    """
    # 1. 检查子句内是否有显式命中
    clause_candidates = collect_intent_signals_fn(clause_text)
    if clause_candidates:
        return clause_candidates[0]

    # 2. 从fallback中按优先级选择
    for candidate_name in fallback_candidates:
        if candidate_name in multi_domain_agent_priority:
            return candidate_name

    return None


def looks_like_compound_request(
        text: str,
        *,
        analyze_request_fn: Callable[[str], Any],
        route_strategy_complex_single: str,
        route_strategy_multi_split: str,
        complex_connector_hints: Sequence[str],
) -> bool:
    """
    判断输入是否更适合走复合任务编排，而不是单轮直接回答。

    设计要点：
    1. 检查路由策略是否为复合策略
    2. 检查是否有多个疑问句
    3. 检查是否有连接词且文本足够长

    设计理由：
    1. 复合请求需要任务编排，不适合单轮回答
    2. 提前识别复合请求，选择合适的处理路径
    3. 提高路由准确性

    判断标准：
    1. 路由策略为multi_domain_split或complex_single_domain
    2. 包含多个疑问句
    3. 包含连接词且文本长度>=30

    Args:
        text: 要判断的文本
        analyze_request_fn: 分析请求的函数
        route_strategy_complex_single: 复杂单域路由策略
        route_strategy_multi_split: 多域拆分路由策略
        complex_connector_hints: 复杂连接词提示

    Returns:
        bool: True表示是复合请求，False表示不是

    示例：
    >>> looks_like_compound_request("郑州天气怎么样，有什么好玩的？", ...)
    True
    >>> looks_like_compound_request("你好", ...)
    False

    使用场景：
    - 判断是否需要任务编排
    - 路由策略选择的辅助判断
    """
    # 1. 检查路由策略
    analysis = analyze_request_fn(text)
    route_strategy = getattr(analysis, "route_strategy", None) or (
        analysis.get("route_strategy") if isinstance(analysis, dict) else None
    )
    if route_strategy in {route_strategy_multi_split, route_strategy_complex_single}:
        return True

    # 2. 检查是否有多个疑问句
    normalized_text = (text or "").strip().lower()
    if not normalized_text:
        return False
    if len(re.findall(r"[?？]", normalized_text)) >= 2:
        return True

    # 3. 检查是否有连接词且文本足够长
    if any(token in normalized_text for token in complex_connector_hints):
        return len(normalized_text) >= 30

    return False

    for first_hint, second_hint in SUPERVISOR_SEQUENTIAL_HINT_PAIRS:
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
    factoid_patterns = (
        r"(什么时间|什么时候|何时|哪天)",
        r"(几号|几月几号|几月|几日|日期)",
        r"(星期几|周几|礼拜几|几点)",
        r"(有什么|有啥|哪些|多少)",
    )
    if any(re.search(pattern, normalized_text) for pattern in factoid_patterns):
        return True
    imperative_patterns = (
        r"^(?:请)?(?:告诉我|告知我|说一下|说说|介绍一下|提供|给出|写一个|写个|查一下|查下|搜一下|搜下)",
        r"(?:告诉我|告知我|说一下|说说|介绍一下|提供|给出).{0,18}(?:路线|自驾|驾车|导航|方案|代码|示例|信息|资料)",
    )
    if any(re.search(pattern, normalized_text) for pattern in imperative_patterns):
        return True
    return any(hint in normalized_text for hint in request_action_hints)


def split_query_clauses(user_text: str) -> List[str]:
    """把复合问题拆成可供规则规划器消费的子句列表。"""
    normalized_text = (user_text or "").strip()
    if not normalized_text:
        return []

    normalized_text = re.sub(r"[，,；;。！？?!]+", "|", normalized_text)
    connector_pattern = "|".join(re.escape(item) for item in SUPERVISOR_CLAUSE_CONNECTOR_PATTERNS)
    normalized_text = re.sub(rf"({connector_pattern})", "|", normalized_text)
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
            explicit_clause_agents.append(AgentTypeEnum.CHAT.code)
    explicit_clause_agents = dedupe_keep_order_fn(explicit_clause_agents)
    if explicit_clause_agents:
        candidate_agents = explicit_clause_agents

    if any(agent_name in AGENT_MEMBERS_TUPLE for agent_name in candidate_agents):
        filtered_candidates = list(candidate_agents)
        weather_agent = AgentTypeEnum.WEATHER.code
        if weather_agent in filtered_candidates:
            weather_explicit = any(is_weather_actionable_clause_fn(clause) for clause in clauses)
            if not weather_explicit:
                filtered_candidates = [
                    agent_name for agent_name in filtered_candidates if agent_name != weather_agent
                ]
        search_agent = AgentTypeEnum.SEARCH.code
        if search_agent in filtered_candidates:
            search_explicit = any(is_search_actionable_clause_fn(clause) for clause in clauses)
            if not search_explicit:
                filtered_candidates = [
                    agent_name for agent_name in filtered_candidates if agent_name != search_agent
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
        return len(normalized_text) >= 18
    for first_hint, second_hint in SUPERVISOR_SEQUENTIAL_HINT_PAIRS:
        if first_hint in normalized_text and second_hint in normalized_text:
            return True
    return False
