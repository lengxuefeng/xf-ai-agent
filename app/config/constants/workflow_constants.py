# -*- coding: utf-8 -*-
"""
工作流状态与节点常量。

定义了LangGraph工作流中的各种状态、事件类型和路由策略。
这些常量是整个Agent编排系统的核心状态机定义。

工作流设计理念：
1. 采用DAG（有向无环图）来描述复杂的任务拆解
2. 支持多Agent并发执行
3. 内置审批中断机制，确保敏感操作的安全
4. 提供细粒度的状态跟踪，方便调试和监控
"""
from enum import Enum
from typing import Dict, Tuple


class TaskStatus(str, Enum):
    """
    DAG 子任务状态枚举。

    定义了子任务在整个执行生命周期中可能的状态。
    状态流转：PENDING -> DISPATCHED -> (PENDING_APPROVAL) -> DONE/ERROR/CANCELLED
    """

    # 任务待处理：任务已经创建，但还没有开始执行
    # 场景：Parent Planner刚刚拆分出这个子任务，等待调度
    PENDING = "pending"

    # 任务已派发：任务已经被Dispatcher派发给Worker开始执行
    # 场景：Dispatcher检测到该任务的依赖条件已满足，发送执行请求
    DISPATCHED = "dispatched"

    # 任务待审批：任务执行到了敏感操作，等待人工审批
    # 场景：Agent执行SQL或代码时触发中断，等待用户决策
    PENDING_APPROVAL = "pending_approval"

    # 任务已完成：任务执行成功，返回了结果
    # 场景：Agent成功完成了查询、搜索、分析等任务
    DONE = "done"

    # 任务执行错误：任务执行过程中发生了错误
    # 场景：API调用失败、数据解析错误、超时等
    ERROR = "error"

    # 任务被取消：任务被用户主动取消或系统取消
    # 场景：用户点击停止按钮、超时取消、客户端断开连接
    CANCELLED = "cancelled"


# 待处理任务状态集合
# 用于判断哪些任务还在活跃状态，不能被清理或覆盖
# 场景：在执行过程中，需要知道哪些任务还在进行中，避免重复派发
PENDING_TASK_STATUSES: Tuple[str, ...] = (
    TaskStatus.PENDING.value,           # 待处理
    TaskStatus.DISPATCHED.value,         # 已派发
    TaskStatus.PENDING_APPROVAL.value,   # 待审批
)


# Worker 节点待审批结果标识
# Worker执行完需要审批的操作后，返回这个特殊结果
# 场景：SQL Agent生成了SQL语句，但没有执行，返回pending_approval让前端显示审批按钮
WORKER_PENDING_APPROVAL_RESULT = "pending_approval"

# Worker 节点取消执行结果标识
# Worker被取消时返回的特殊结果，用于区分正常完成和被取消
# 场景：用户点击停止按钮，正在执行的Agent收到取消信号
WORKER_CANCELLED_RESULT = "__cancelled__"

# 工作流中断结果类型标识
# 当工作流触发中断时，返回这个特殊类型
# 场景：审批流程恢复时，需要识别之前是因为中断而挂起的
WORKFLOW_INTERRUPT_RESULT_TYPE = "__interrupt__"


class GraphQueueItemType(str, Enum):
    """
    GraphRunner 线程队列中的事件类型枚举。

    GraphRunner使用生产者-消费者模式，后台线程执行图，主线程通过队列消费事件。
    这里定义了队列中可能的事件类型。
    """

    # 日志事件：记录执行过程中的日志信息
    # 场景：Agent执行时产生的调试信息、警告等
    LOG = "log"

    # 图执行事件：图的执行状态变化
    # 场景：节点开始执行、节点完成、状态更新等
    GRAPH = "graph"

    # 子 Agent 实时正文流事件：Agent输出的文本内容
    # 场景：Chat Agent正在生成回复，逐字推送给前端
    LIVE_STREAM = "live_stream"

    # 完成事件：整个图执行完成
    # 场景：所有子任务完成，Aggregator汇总结果后发送完成事件
    DONE = "done"

    # 错误事件：图执行过程中发生错误
    # 场景：某个子任务失败、异常抛出等
    ERROR = "error"


# 图流式输出模式
# LangGraph支持两种流式模式：updates（状态更新）和messages（消息流）
# 同时监听两种模式可以兼顾状态跟踪和实时内容推送
GRAPH_STREAM_MODES: Tuple[str, str] = ("updates", "messages")  # 状态更新和消息流


class RouteStrategy(str, Enum):
    """
    路由策略类型枚举。

    定义Supervisor在分析用户输入后决定采用的路由策略。
    不同的策略对应不同的执行路径和资源分配。
    """

    # 单一领域单一意图，直接单兵执行
    # 场景：用户问"郑州今天天气如何？"，只需要天气Agent处理
    SINGLE_DOMAIN = "single_domain"

    # 命中多领域/多意图，需要拆分任务
    # 场景：用户问"郑州今天天气如何？还有附近的商场推荐？"，需要天气Agent和搜索Agent并发执行
    MULTI_DOMAIN_SPLIT = "multi_domain_split"

    # 单领域但语义复杂（先后依赖等），走规划拆解
    # 场景：用户问"帮我分析一下这个SQL查询为什么慢，并给出优化建议"，需要分步执行：执行SQL -> 分析结果 -> 提出建议
    COMPLEX_SINGLE_DOMAIN = "complex_single_domain"


# 多域拆分阶段允许编排的 Agent（按默认优先级）
# 当采用多域拆分策略时，这些Agent可以并发执行
# 优先级决定了前端显示的顺序和资源分配的优先级
MULTI_DOMAIN_AGENT_PRIORITY: Tuple[str, ...] = (
    "yunyou_agent",    # 云柚专员：处理医疗设备相关数据，通常最重要
    "sql_agent",       # 账房先生：处理数据库查询，影响数据准确性
    "weather_agent",   # 天象司：处理天气信息，时效性强
    "search_agent",    # 典籍司：处理联网搜索，信息量大
    "medical_agent",   # 医馆参谋：处理医疗咨询，需要专业知识
    "code_agent",      # 工坊司：处理代码执行，复杂度最高
)


# Agent 到数据域的映射（用于统一的前置分析结果）
# 将不同的Agent映射到数据源，方便做数据域分析和路由决策
# 场景：分析用户输入时，判断需要访问哪些数据源，从而决定调用哪些Agent
AGENT_DOMAIN_MAP: Dict[str, str] = {
    "yunyou_agent": "YUNYOU_DB",   # 云柚Agent访问云柚数据库
    "sql_agent": "LOCAL_DB",        # SQL Agent访问本地数据库
    "weather_agent": "WEB_SEARCH",  # 天气Agent通过联网获取实时数据
    "search_agent": "WEB_SEARCH",   # 搜索Agent通过联网获取信息
    "medical_agent": "GENERAL",     # 医疗Agent使用通用知识库
    "code_agent": "GENERAL",        # 代码Agent使用通用知识和工具
    "CHAT": "GENERAL",              # 通用Chat使用通用知识
}
