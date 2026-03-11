# -*- coding: utf-8 -*-
"""工作流状态与节点常量。"""
from enum import Enum
from typing import Dict, Tuple


class TaskStatus(str, Enum):
    """DAG 子任务状态枚举"""

    # 任务待处理
    PENDING = "pending"

    # 任务已派发
    DISPATCHED = "dispatched"

    # 任务待审批
    PENDING_APPROVAL = "pending_approval"

    # 任务已完成
    DONE = "done"

    # 任务执行错误
    ERROR = "error"

    # 任务被取消（例如客户端主动中断）
    CANCELLED = "cancelled"


# 待处理任务状态集合
PENDING_TASK_STATUSES: Tuple[str, ...] = (
    TaskStatus.PENDING.value,           # 待处理
    TaskStatus.DISPATCHED.value,         # 已派发
    TaskStatus.PENDING_APPROVAL.value,   # 待审批
)


# Worker 节点待审批结果标识
WORKER_PENDING_APPROVAL_RESULT = "pending_approval"

# Worker 节点取消执行结果标识
WORKER_CANCELLED_RESULT = "__cancelled__"

# 工作流中断结果类型标识
WORKFLOW_INTERRUPT_RESULT_TYPE = "__interrupt__"


class GraphQueueItemType(str, Enum):
    """GraphRunner 线程队列中的事件类型枚举"""

    # 日志事件
    LOG = "log"

    # 图执行事件
    GRAPH = "graph"

    # 完成事件
    DONE = "done"

    # 错误事件
    ERROR = "error"


# 图流式输出模式
GRAPH_STREAM_MODES: Tuple[str, str] = ("updates", "messages")  # 状态更新和消息流


class RouteStrategy(str, Enum):
    """路由策略类型枚举。"""

    # 单一领域单一意图，直接单兵执行
    SINGLE_DOMAIN = "single_domain"

    # 命中多领域/多意图，需要拆分任务
    MULTI_DOMAIN_SPLIT = "multi_domain_split"

    # 单领域但语义复杂（先后依赖等），走规划拆解
    COMPLEX_SINGLE_DOMAIN = "complex_single_domain"


# 多域拆分阶段允许编排的 Agent（按默认优先级）
MULTI_DOMAIN_AGENT_PRIORITY: Tuple[str, ...] = (
    "yunyou_agent",
    "sql_agent",
    "weather_agent",
    "search_agent",
    "medical_agent",
    "code_agent",
)


# Agent 到数据域的映射（用于统一的前置分析结果）
AGENT_DOMAIN_MAP: Dict[str, str] = {
    "yunyou_agent": "YUNYOU_DB",
    "sql_agent": "LOCAL_DB",
    "weather_agent": "WEB_SEARCH",
    "search_agent": "WEB_SEARCH",
    "medical_agent": "GENERAL",
    "code_agent": "GENERAL",
    "CHAT": "GENERAL",
}
