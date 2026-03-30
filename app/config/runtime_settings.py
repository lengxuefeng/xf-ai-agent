# -*- coding: utf-8 -*-
"""
运行时参数配置中心。

说明：
1. 将高频硬编码参数统一收口，便于线上调参与回归。
2. 仅放“运行期策略参数”，不放业务数据。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple


def _as_int(name: str, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    """从环境变量读取整数值并校验范围"""
    # 尝试读取环境变量，如果不存在或为空则使用默认值
    raw = os.getenv(name)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = int(raw)  # 转换为整数
        except Exception:
            value = default  # 转换失败则使用默认值
    # 限制最小值
    if min_value is not None:
        value = max(min_value, value)
    # 限制最大值
    if max_value is not None:
        value = min(max_value, value)
    return value


def _as_float(name: str, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
    """从环境变量读取浮点数并校验范围"""
    # 尝试读取环境变量，如果不存在或为空则使用默认值
    raw = os.getenv(name)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = float(raw)  # 转换为浮点数
        except Exception:
            value = default  # 转换失败则使用默认值
    # 限制最小值
    if min_value is not None:
        value = max(min_value, value)
    # 限制最大值
    if max_value is not None:
        value = min(max_value, value)
    return value


def _as_bool(name: str, default: bool) -> bool:
    """从环境变量读取布尔值"""
    raw = os.getenv(name)
    if raw is None:
        return default
    # 检查字符串是否为真值（支持多种格式）
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_csv_tuple(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    """从环境变量读取逗号分隔的字符串并转换为元组"""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    # 按逗号分割并去除空白
    return tuple(x.strip() for x in raw.split(",") if x.strip())


@dataclass(frozen=True)
class GraphRunnerTuning:
    """GraphRunner 图执行器的调优参数配置"""

    # 规则引擎扫描用户输入的最大长度，超出则跳过规则匹配
    rule_scan_max_len: int

    # 估算每个意图占用的字符数，用于计算规则匹配覆盖率
    chars_per_intent: int

    # 队列轮询超时时间（秒），从事件队列获取数据的最长等待时间
    queue_poll_timeout_sec: float

    # 空闲心跳间隔（秒），长时间无事件时发送心跳保持连接
    idle_heartbeat_sec: float

    # 空闲超时时间（秒），超时则强制结束图执行
    idle_timeout_sec: float

    # 是否启用空闲超时（启用后长工具调用可能被误判）
    idle_timeout_enabled: bool

    # 硬超时时间（秒），图执行的最长总时长限制
    hard_timeout_sec: float

    # 是否启用“流结束后的全量子图中断扫描”（默认关闭，避免额外 Agent 初始化和数据库开销）
    post_run_interrupt_scan_enabled: bool


@dataclass(frozen=True)
class YunyouHttpConfig:
    """云柚系统 HTTP 调用的参数配置"""

    # HTTP 请求超时时间（秒）
    timeout_seconds: int

    # 是否允许自动跟随重定向
    allow_redirects: bool

    # 是否验证 SSL 证书
    verify_ssl: bool

    # 是否禁用代理
    disable_proxy: bool

    # HTTP User-Agent 请求头
    user_agent: str

    # HTTP Connection 请求头
    connection_header: str

    # 错误信息预览的最大字符数，用于日志输出
    error_preview_chars: int

    # 单次调用最大重试次数（包含首试）
    retry_attempts: int

    # 重试基础退避时间（毫秒）
    retry_backoff_ms: int

    # 熔断触发阈值（连续失败次数）
    circuit_breaker_threshold: int

    # 熔断窗口时长（秒）
    circuit_open_seconds: int

    # 正向缓存 TTL（秒）
    cache_ttl_seconds: int

    # 失败时允许使用的陈旧缓存窗口（秒）
    cache_stale_seconds: int


@dataclass(frozen=True)
class YunyouDbPoolConfig:
    """云柚数据库连接池配置和查询限制参数"""

    # 连接回收时间（秒），超过此时间的连接会被回收重建
    pool_recycle_seconds: int

    # 连接池基础大小
    pool_size: int

    # 连接池最大溢出连接数，允许超过基础大小的连接数
    max_overflow: int

    # Holter 表的表名
    holter_table_name: str

    # Holter 查询的默认返回条数
    holter_default_limit: int

    # Holter 查询的最大返回条数限制
    holter_max_limit: int


@dataclass(frozen=True)
class RouteMetricsConfig:
    """路由指标服务配置参数"""

    # 识别用户纠正意图的关键词列表
    correction_keywords: Tuple[str, ...]

    # 最大缓存事件数，超过则丢弃旧事件
    max_events: int


@dataclass(frozen=True)
class SemanticCacheConfig:
    """语义缓存服务配置参数"""

    # 默认缓存存活时间（秒）
    default_ttl_seconds: int

    # 缓存最大条目数，超过则使用 LRU 策略清理
    max_size: int


@dataclass(frozen=True)
class AgentLoopConfig:
    """工具型 Agent 的循环限制和上下文管理参数"""

    # 搜索 Agent 最大工具调用循环次数
    search_max_tool_loops: int

    # 天气 Agent 最大工具调用循环次数
    weather_max_tool_loops: int

    # 云柚 Agent 最大工具调用循环次数
    yunyou_max_tool_loops: int

    # 上下文中保留的历史消息条数
    context_history_messages: int

    # 上下文压缩时允许的最大 Token 数
    context_compress_max_tokens: int

    # 上下文压缩时允许的最大字符数
    context_compress_max_chars: int

    # 相关性筛选时，始终保留的尾部消息条数
    context_relevance_tail_messages: int

    # 相关性筛选的最小词长度（字符）
    context_relevance_min_token_chars: int


@dataclass(frozen=True)
class ModelTieringConfig:
    """模型分层调度配置（简单问答/路由/复杂任务分流）"""

    # Tier-1 路由小模型名称；为空则复用主模型
    router_model: str

    # 简单对话小模型名称；为空则复用主模型
    simple_chat_model: str


@dataclass(frozen=True)
class RouterPolicyConfig:
    """路由器策略配置（性能/稳定性优先）。"""

    # 是否启用 Domain Router 的 LLM 兜底分类（关闭可显著降低时延）
    domain_llm_fallback_enabled: bool

    # 是否启用 Intent Router 的 LLM 兜底分类（关闭后主要依赖规则路由）
    intent_llm_fallback_enabled: bool

    # 路由分类器可见历史窗口大小（条）
    classifier_history_messages: int

    # 是否启用 GENERAL 域直达 CHAT 的快速通道
    general_chat_fastpath_enabled: bool

    # 是否启用 Parent Planner 的 LLM 兜底（关闭后全部走确定性任务编排）
    planner_llm_fallback_enabled: bool

    # 路由器/规划器结构化调用的单次超时（秒）
    router_llm_timeout_sec: float


@dataclass(frozen=True)
class AggregatorConfig:
    """结果聚合器配置。"""

    # 是否启用大模型聚合（关闭时走快速确定性聚合，时延更低）
    use_llm_aggregation: bool

    # 单个子任务结果参与聚合的最大字符数，避免超长输入拖慢聚合
    max_result_chars: int


@dataclass(frozen=True)
class WorkflowReflectionConfig:
    """多步执行反思配置。"""

    # 是否启用执行后反思与追加任务
    enabled: bool

    # 是否允许使用 LLM 反思器决定是否追加任务
    llm_enabled: bool

    # 最大反思轮次，防止无限追加
    max_rounds: int

    # 反思阶段单次 LLM 超时
    llm_timeout_sec: float

    # 传给反思器的单任务结果预览最大字符数
    result_preview_max_chars: int


# 图执行器调优参数配置实例
GRAPH_RUNNER_TUNING = GraphRunnerTuning(
    # 规则扫描最大长度（环境变量：ROUTER_RULE_SCAN_MAX_LEN，默认 60，范围 1-500）
    rule_scan_max_len=_as_int("ROUTER_RULE_SCAN_MAX_LEN", 60, min_value=1, max_value=500),

    # 每个意图占用字符数（环境变量：ROUTER_CHARS_PER_INTENT，默认 15，范围 1-200）
    chars_per_intent=_as_int("ROUTER_CHARS_PER_INTENT", 15, min_value=1, max_value=200),

    # 队列轮询超时（环境变量：GRAPH_QUEUE_POLL_TIMEOUT_SEC，默认 1.0，范围 0.1-10）
    queue_poll_timeout_sec=_as_float("GRAPH_QUEUE_POLL_TIMEOUT_SEC", 1.0, min_value=0.1, max_value=10),

    # 空闲心跳间隔（环境变量：GRAPH_IDLE_HEARTBEAT_SEC，默认 10.0，范围 1-120）
    idle_heartbeat_sec=_as_float("GRAPH_IDLE_HEARTBEAT_SEC", 10.0, min_value=1, max_value=120),

    # 空闲超时时间（环境变量：GRAPH_IDLE_TIMEOUT_SEC，默认 45.0，范围 5-600）
    idle_timeout_sec=_as_float("GRAPH_IDLE_TIMEOUT_SEC", 45.0, min_value=5, max_value=600),

    # 是否启用空闲超时（环境变量：GRAPH_IDLE_TIMEOUT_ENABLED，默认 false）
    idle_timeout_enabled=_as_bool("GRAPH_IDLE_TIMEOUT_ENABLED", False),

    # 硬超时时间（环境变量：GRAPH_HARD_TIMEOUT_SEC，默认 300.0，范围 30-3600）
    hard_timeout_sec=_as_float("GRAPH_HARD_TIMEOUT_SEC", 300.0, min_value=30, max_value=3600),

    # 流后全量扫描开关（环境变量：GRAPH_POST_RUN_INTERRUPT_SCAN_ENABLED，默认 false）
    post_run_interrupt_scan_enabled=_as_bool("GRAPH_POST_RUN_INTERRUPT_SCAN_ENABLED", False),
)

# 云柚 HTTP 调用配置实例
YUNYOU_HTTP_CONFIG = YunyouHttpConfig(
    # 请求超时（环境变量：YUNYOU_HTTP_TIMEOUT_SECONDS，默认 8，范围 3-120）
    timeout_seconds=_as_int("YUNYOU_HTTP_TIMEOUT_SECONDS", 8, min_value=3, max_value=120),

    # 允许重定向（环境变量：YUNYOU_HTTP_ALLOW_REDIRECTS，默认 False）
    allow_redirects=_as_bool("YUNYOU_HTTP_ALLOW_REDIRECTS", False),

    # 验证 SSL（环境变量：YUNYOU_HTTP_VERIFY_SSL，默认 False）
    verify_ssl=_as_bool("YUNYOU_HTTP_VERIFY_SSL", False),

    # 禁用代理（环境变量：YUNYOU_HTTP_DISABLE_PROXY，默认 True）
    disable_proxy=_as_bool("YUNYOU_HTTP_DISABLE_PROXY", True),

    # User-Agent（环境变量：YUNYOU_HTTP_USER_AGENT）
    user_agent=os.getenv(
        "YUNYOU_HTTP_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    ),

    # Connection 头（环境变量：YUNYOU_HTTP_CONNECTION，默认 "close"）
    connection_header=os.getenv("YUNYOU_HTTP_CONNECTION", "close"),

    # 错误预览字符数（环境变量：YUNYOU_HTTP_ERROR_PREVIEW_CHARS，默认 500，范围 100-5000）
    error_preview_chars=_as_int("YUNYOU_HTTP_ERROR_PREVIEW_CHARS", 500, min_value=100, max_value=5000),

    # 重试次数（环境变量：YUNYOU_HTTP_RETRY_ATTEMPTS，默认 1，范围 1-5）
    retry_attempts=_as_int("YUNYOU_HTTP_RETRY_ATTEMPTS", 1, min_value=1, max_value=5),

    # 重试退避（环境变量：YUNYOU_HTTP_RETRY_BACKOFF_MS，默认 300ms，范围 50-5000）
    retry_backoff_ms=_as_int("YUNYOU_HTTP_RETRY_BACKOFF_MS", 300, min_value=50, max_value=5000),

    # 熔断阈值（环境变量：YUNYOU_HTTP_CIRCUIT_BREAKER_THRESHOLD，默认 3，范围 1-20）
    circuit_breaker_threshold=_as_int("YUNYOU_HTTP_CIRCUIT_BREAKER_THRESHOLD", 3, min_value=1, max_value=20),

    # 熔断时长（环境变量：YUNYOU_HTTP_CIRCUIT_OPEN_SECONDS，默认 30 秒，范围 5-600）
    circuit_open_seconds=_as_int("YUNYOU_HTTP_CIRCUIT_OPEN_SECONDS", 30, min_value=5, max_value=600),

    # 正向缓存 TTL（环境变量：YUNYOU_HTTP_CACHE_TTL_SECONDS，默认 20 秒，范围 1-600）
    cache_ttl_seconds=_as_int("YUNYOU_HTTP_CACHE_TTL_SECONDS", 20, min_value=1, max_value=600),

    # 陈旧缓存窗口（环境变量：YUNYOU_HTTP_CACHE_STALE_SECONDS，默认 120 秒，范围 0-3600）
    cache_stale_seconds=_as_int("YUNYOU_HTTP_CACHE_STALE_SECONDS", 120, min_value=0, max_value=3600),
)

# 云柚数据库连接池配置实例
YUNYOU_DB_POOL_CONFIG = YunyouDbPoolConfig(
    # 连接回收时间（环境变量：YUNYOU_DB_POOL_RECYCLE_SECONDS，默认 3600，范围 60-86400）
    pool_recycle_seconds=_as_int("YUNYOU_DB_POOL_RECYCLE_SECONDS", 3600, min_value=60, max_value=86400),

    # 连接池大小（环境变量：YUNYOU_DB_POOL_SIZE，默认 10，范围 1-100）
    pool_size=_as_int("YUNYOU_DB_POOL_SIZE", 10, min_value=1, max_value=100),

    # 最大溢出（环境变量：YUNYOU_DB_MAX_OVERFLOW，默认 10，范围 0-200）
    max_overflow=_as_int("YUNYOU_DB_MAX_OVERFLOW", 10, min_value=0, max_value=200),

    # Holter 表名（环境变量：YUNYOU_HOLTER_TABLE，默认 "t_holter_use_record"）
    holter_table_name=os.getenv("YUNYOU_HOLTER_TABLE", "t_holter_use_record"),

    # 默认查询限制（环境变量：YUNYOU_HOLTER_DEFAULT_LIMIT，默认 5，范围 1-200）
    holter_default_limit=_as_int("YUNYOU_HOLTER_DEFAULT_LIMIT", 5, min_value=1, max_value=200),

    # 最大查询限制（环境变量：YUNYOU_HOLTER_MAX_LIMIT，默认 200，范围 1-1000）
    holter_max_limit=_as_int("YUNYOU_HOLTER_MAX_LIMIT", 200, min_value=1, max_value=1000),
)

# 路由指标配置实例
ROUTE_METRICS_CONFIG = RouteMetricsConfig(
    # 纠正关键词（环境变量：ROUTE_CORRECTION_KEYWORDS，默认列表）
    correction_keywords=_as_csv_tuple(
        "ROUTE_CORRECTION_KEYWORDS",
        ("不对", "错了", "查错", "不是这个", "不是我要的", "还是不对", "你查错了"),
    ),

    # 最大事件数（环境变量：ROUTE_METRICS_MAX_EVENTS，默认 500，范围 50-5000）
    max_events=_as_int("ROUTE_METRICS_MAX_EVENTS", 500, min_value=50, max_value=5000),
)

# 语义缓存配置实例
SEMANTIC_CACHE_CONFIG = SemanticCacheConfig(
    # 默认 TTL（环境变量：SEMANTIC_CACHE_TTL_SECONDS，默认 120，范围 1-86400）
    default_ttl_seconds=_as_int("SEMANTIC_CACHE_TTL_SECONDS", 120, min_value=1, max_value=86400),

    # 最大缓存数（环境变量：SEMANTIC_CACHE_MAX_SIZE，默认 1000，范围 10-500000）
    max_size=_as_int("SEMANTIC_CACHE_MAX_SIZE", 1000, min_value=10, max_value=500000),
)

# Agent 循环配置实例
AGENT_LOOP_CONFIG = AgentLoopConfig(
    # 搜索 Agent 最大循环（环境变量：SEARCH_AGENT_MAX_TOOL_LOOPS，默认 4，范围 1-20）
    search_max_tool_loops=_as_int("SEARCH_AGENT_MAX_TOOL_LOOPS", 4, min_value=1, max_value=20),

    # 天气 Agent 最大循环（环境变量：WEATHER_AGENT_MAX_TOOL_LOOPS，默认 4，范围 1-20）
    weather_max_tool_loops=_as_int("WEATHER_AGENT_MAX_TOOL_LOOPS", 4, min_value=1, max_value=20),

    # 云柚 Agent 最大循环（环境变量：YUNYOU_AGENT_MAX_TOOL_LOOPS，默认 6，范围 1-30）
    yunyou_max_tool_loops=_as_int("YUNYOU_AGENT_MAX_TOOL_LOOPS", 6, min_value=1, max_value=30),

    # 历史消息数（环境变量：AGENT_CONTEXT_HISTORY_MESSAGES，默认 10，范围 2-50）
    context_history_messages=_as_int("AGENT_CONTEXT_HISTORY_MESSAGES", 10, min_value=2, max_value=50),

    # 压缩最大 Token（环境变量：AGENT_CONTEXT_COMPRESS_MAX_TOKENS，默认 1800，范围 200-32000）
    context_compress_max_tokens=_as_int("AGENT_CONTEXT_COMPRESS_MAX_TOKENS", 1800, min_value=200, max_value=32000),

    # 压缩最大字符（环境变量：AGENT_CONTEXT_COMPRESS_MAX_CHARS，默认 12000，范围 1000-200000）
    context_compress_max_chars=_as_int("AGENT_CONTEXT_COMPRESS_MAX_CHARS", 12000, min_value=1000, max_value=200000),

    # 相关性尾部保留（环境变量：AGENT_CONTEXT_RELEVANCE_TAIL_MESSAGES，默认 4，范围 1-20）
    context_relevance_tail_messages=_as_int("AGENT_CONTEXT_RELEVANCE_TAIL_MESSAGES", 4, min_value=1, max_value=20),

    # 相关性最小词长（环境变量：AGENT_CONTEXT_RELEVANCE_MIN_TOKEN_CHARS，默认 2，范围 1-10）
    context_relevance_min_token_chars=_as_int("AGENT_CONTEXT_RELEVANCE_MIN_TOKEN_CHARS", 2, min_value=1, max_value=10),
)

# 模型分层调度配置实例
MODEL_TIERING_CONFIG = ModelTieringConfig(
    # 路由专用小模型（环境变量：ROUTER_MODEL，默认空）
    router_model=os.getenv("ROUTER_MODEL", "").strip(),

    # 简单问答专用小模型（环境变量：SIMPLE_CHAT_MODEL，默认空）
    simple_chat_model=os.getenv("SIMPLE_CHAT_MODEL", "").strip(),
)

# 路由策略配置实例
ROUTER_POLICY_CONFIG = RouterPolicyConfig(
    # Domain 路由 LLM 兜底开关（环境变量：ROUTER_DOMAIN_LLM_FALLBACK_ENABLED，默认 false）
    domain_llm_fallback_enabled=_as_bool("ROUTER_DOMAIN_LLM_FALLBACK_ENABLED", False),

    # Intent 路由 LLM 兜底开关（环境变量：ROUTER_INTENT_LLM_FALLBACK_ENABLED，默认 false）
    intent_llm_fallback_enabled=_as_bool("ROUTER_INTENT_LLM_FALLBACK_ENABLED", False),

    # 分类历史窗口（环境变量：ROUTER_CLASSIFIER_HISTORY_MESSAGES，默认 8，范围 3-20）
    classifier_history_messages=_as_int("ROUTER_CLASSIFIER_HISTORY_MESSAGES", 8, min_value=3, max_value=20),

    # GENERAL 直达 CHAT（环境变量：ROUTER_GENERAL_CHAT_FASTPATH_ENABLED，默认 true）
    general_chat_fastpath_enabled=_as_bool("ROUTER_GENERAL_CHAT_FASTPATH_ENABLED", True),

    # Planner LLM 兜底开关（环境变量：ROUTER_PLANNER_LLM_FALLBACK_ENABLED，默认 false）
    planner_llm_fallback_enabled=_as_bool("ROUTER_PLANNER_LLM_FALLBACK_ENABLED", False),

    # 路由器结构化调用超时（环境变量：ROUTER_LLM_TIMEOUT_SEC，默认 12.0，范围 2-120）
    router_llm_timeout_sec=_as_float("ROUTER_LLM_TIMEOUT_SEC", 12.0, min_value=2.0, max_value=120.0),
)

# 聚合器配置实例
AGGREGATOR_CONFIG = AggregatorConfig(
    # 是否启用大模型聚合（环境变量：AGGREGATOR_USE_LLM，默认 false）
    use_llm_aggregation=_as_bool("AGGREGATOR_USE_LLM", False),

    # 每个任务结果最大字符数（环境变量：AGGREGATOR_MAX_RESULT_CHARS，默认 2000，范围 200-20000）
    max_result_chars=_as_int("AGGREGATOR_MAX_RESULT_CHARS", 2000, min_value=200, max_value=20000),
)

# 任务执行后自动反思配置
WORKFLOW_REFLECTION_CONFIG = WorkflowReflectionConfig(
    # 是否启用（环境变量：WORKFLOW_REFLECTION_ENABLED，默认 true）
    enabled=_as_bool("WORKFLOW_REFLECTION_ENABLED", True),

    # 是否启用 LLM 反思器（环境变量：WORKFLOW_REFLECTION_LLM_ENABLED，默认 true）
    llm_enabled=_as_bool("WORKFLOW_REFLECTION_LLM_ENABLED", True),

    # 最大反思轮次（环境变量：WORKFLOW_REFLECTION_MAX_ROUNDS，默认 2，范围 0-5）
    max_rounds=_as_int("WORKFLOW_REFLECTION_MAX_ROUNDS", 2, min_value=0, max_value=5),

    # 单次反思超时（环境变量：WORKFLOW_REFLECTION_LLM_TIMEOUT_SEC，默认 10.0，范围 1-60）
    llm_timeout_sec=_as_float("WORKFLOW_REFLECTION_LLM_TIMEOUT_SEC", 10.0, min_value=1.0, max_value=60.0),

    # 单任务结果预览最大字符数（环境变量：WORKFLOW_REFLECTION_RESULT_PREVIEW_MAX_CHARS，默认 1200，范围 200-10000）
    result_preview_max_chars=_as_int(
        "WORKFLOW_REFLECTION_RESULT_PREVIEW_MAX_CHARS",
        1200,
        min_value=200,
        max_value=10000,
    ),
)

# Chat 兜底节点流式开关（环境变量：CHAT_NODE_STREAM_ENABLED，默认 true）
CHAT_NODE_STREAM_ENABLED = _as_bool("CHAT_NODE_STREAM_ENABLED", True)

# LLM 传输层请求超时（环境变量：LLM_REQUEST_TIMEOUT_SEC，默认 100.0，范围 3-300）
LLM_REQUEST_TIMEOUT_SEC = _as_float(
    "LLM_REQUEST_TIMEOUT_SEC",
    100.0,
    min_value=3.0,
    max_value=300.0,
)

# LLM 传输层最大重试次数（环境变量：LLM_MAX_RETRIES，默认 1，范围 0-5）
LLM_MAX_RETRIES = _as_int(
    "LLM_MAX_RETRIES",
    1,
    min_value=0,
    max_value=5,
)

# Chat 兜底节点首 token 超时（环境变量：CHAT_NODE_FIRST_TOKEN_TIMEOUT_SEC，默认 20.0，范围 1-120）
CHAT_NODE_FIRST_TOKEN_TIMEOUT_SEC = _as_float(
    "CHAT_NODE_FIRST_TOKEN_TIMEOUT_SEC",
    20.0,
    min_value=1.0,
    max_value=120.0,
)

# Chat 兜底节点总超时（环境变量：CHAT_NODE_TOTAL_TIMEOUT_SEC，默认 90.0，范围 2-300）
CHAT_NODE_TOTAL_TIMEOUT_SEC = _as_float(
    "CHAT_NODE_TOTAL_TIMEOUT_SEC",
    90.0,
    min_value=2.0,
    max_value=300.0,
)

# 流式聊天历史窗口（环境变量：CHAT_STREAM_HISTORY_LIMIT，默认 30，范围 1-200）
CHAT_STREAM_HISTORY_LIMIT = _as_int("CHAT_STREAM_HISTORY_LIMIT", 30, min_value=1, max_value=200)

# 自定义日志异步写入开关（环境变量：LOG_ASYNC_MODE，默认 true）
LOG_ASYNC_MODE = _as_bool("LOG_ASYNC_MODE", True)

# 子 Agent 实时正文流开关（环境变量：AGENT_LIVE_STREAM_ENABLED，默认 true）
AGENT_LIVE_STREAM_ENABLED = _as_bool("AGENT_LIVE_STREAM_ENABLED", True)

# Checkpointer 策略（环境变量：CHECKPOINTER_POLICY，默认 hybrid）
# - hybrid: supervisor/sql/yunyou 使用持久化，其它 Agent 使用内存
# - all_durable: 所有图都使用持久化
# - all_memory: 所有图都使用内存
CHECKPOINTER_POLICY = (os.getenv("CHECKPOINTER_POLICY", "hybrid") or "hybrid").strip().lower()

# SQL schema 缓存 TTL（环境变量：SQL_SCHEMA_CACHE_TTL_SECONDS，默认 120，范围 1-3600）
SQL_SCHEMA_CACHE_TTL_SECONDS = _as_int("SQL_SCHEMA_CACHE_TTL_SECONDS", 120, min_value=1, max_value=3600)

# 云柚 Holter 表发现缓存 TTL（环境变量：YUNYOU_TABLE_DISCOVERY_CACHE_TTL_SECONDS，默认 300，范围 1-3600）
YUNYOU_TABLE_DISCOVERY_CACHE_TTL_SECONDS = _as_int(
    "YUNYOU_TABLE_DISCOVERY_CACHE_TTL_SECONDS", 300, min_value=1, max_value=3600
)

# 搜索工具超时预算（环境变量：TAVILY_TIMEOUT_SEC，默认 6，范围 1-60）
SEARCH_TOOL_TIMEOUT_SECONDS = _as_int("TAVILY_TIMEOUT_SEC", 6, min_value=1, max_value=60)

# 天气工具超时预算（环境变量：WEATHER_TOOL_TIMEOUT_SEC，默认 6，范围 1-60）
WEATHER_TOOL_TIMEOUT_SECONDS = _as_int("WEATHER_TOOL_TIMEOUT_SEC", 6, min_value=1, max_value=60)


@dataclass(frozen=True)
class SessionPoolConfig:
    """Supervisor 图实例预热池配置。"""

    # 是否启用预热池（关闭则退化为按需编译）
    enabled: bool

    # 池中预热的实例数量上限
    pool_size: int

    # 实例最大空闲存活时间（秒），超过后被替换
    max_idle_seconds: float

    # 后台预热任务的检查间隔（秒）
    refill_interval_seconds: float

    # 从池中借取实例时的最长等待时间（秒），超时则按需创建
    borrow_timeout_seconds: float


# Session 预热池配置实例
SESSION_POOL_CONFIG = SessionPoolConfig(
    # 是否启用（环境变量：SESSION_POOL_ENABLED，默认 true）
    enabled=_as_bool("SESSION_POOL_ENABLED", True),

    # 预热实例数（环境变量：SESSION_POOL_SIZE，默认 4，范围 1-32）
    pool_size=_as_int("SESSION_POOL_SIZE", 4, min_value=1, max_value=32),

    # 最大空闲时间（环境变量：SESSION_POOL_MAX_IDLE_SECONDS，默认 300，范围 30-3600）
    max_idle_seconds=_as_float("SESSION_POOL_MAX_IDLE_SECONDS", 300.0, min_value=30.0, max_value=3600.0),

    # 预热检查间隔（环境变量：SESSION_POOL_REFILL_INTERVAL_SECONDS，默认 60，范围 5-600）
    refill_interval_seconds=_as_float("SESSION_POOL_REFILL_INTERVAL_SECONDS", 60.0, min_value=5.0, max_value=600.0),

    # 借取超时（环境变量：SESSION_POOL_BORROW_TIMEOUT_SECONDS，默认 0.1，范围 0.01-5）
    borrow_timeout_seconds=_as_float("SESSION_POOL_BORROW_TIMEOUT_SECONDS", 0.1, min_value=0.01, max_value=5.0),
)
