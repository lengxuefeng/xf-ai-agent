# -*- coding: utf-8 -*-
"""
运行时类型定义（Runtime Types）。

定义了运行时核心数据结构，是整个运行时系统的类型基础。
这些数据结构在各个模块间传递，确保了类型的一致性。

设计要点：
1. 使用dataclass简化数据类定义
2. 使用slots优化内存占用
3. 使用类型注解提高代码可读性
4. 统一的时间戳格式

使用场景：
- RunContext: 贯穿整个图执行过程的上下文对象
- RunStatus: 运行状态枚举
- RunStateSnapshot: 运行状态快照
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict


def utc_now_iso() -> str:
    """
    返回统一 UTC 时间戳。

    设计理由：
    1. 使用UTC时间戳，避免时区问题
    2. ISO 8601格式，标准且易解析
    3. 精确到秒，减少存储空间

    Returns:
        str: UTC时间戳字符串

    场景：
    - 数据模型的时间戳
    - 日志时间戳
    - 工作流事件时间戳
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunStatus(str, Enum):
    """
    运行态状态枚举。

    定义了单次运行在整个生命周期中可能的状态。
    状态流转：PENDING -> RUNNING -> (COMPLETED/FAILED/INTERRUPTED/CANCELLED/WAITING_APPROVAL)

    状态说明：
    - PENDING: 待处理，运行已注册但未开始执行
    - RUNNING: 运行中，正在执行
    - COMPLETED: 已完成，执行成功
    - FAILED: 失败，执行过程中发生错误
    - INTERRUPTED: 已中断，等待审批恢复
    - CANCELLED: 已取消，被用户或系统取消
    - WAITING_APPROVAL: 等待审批，需要人工介入
    """

    # 待处理：运行已注册但未开始执行
    PENDING = "pending"

    # 运行中：正在执行
    RUNNING = "running"

    # 已完成：执行成功
    COMPLETED = "completed"

    # 失败：执行过程中发生错误
    FAILED = "failed"

    # 已中断：等待审批恢复
    INTERRUPTED = "interrupted"

    # 已取消：被用户或系统取消
    CANCELLED = "cancelled"

    # 等待审批：需要人工介入
    WAITING_APPROVAL = "waiting_approval"


@dataclass(slots=True)
class RunContext:
    """
    单次运行的统一上下文。

    核心职责：
    1. 封装运行所需的所有信息
    2. 在各模块间传递上下文
    3. 提供统一的配置接口

    设计理由：
    1. 统一的上下文对象，避免参数传递混乱
    2. 使用dataclass简化定义和使用
    3. 使用slots优化内存占用
    4. 支持运行恢复（is_resume）

    字段说明：
    - session_id: 会话ID，用于关联用户对话历史
    - run_id: 运行ID，用于标识单次运行
    - user_input: 用户输入的文本
    - model_config: 模型配置字典
    - history_size: 历史消息数量
    - is_resume: 是否为恢复运行（如审批后恢复）
    - context_summary: 会话上下文摘要
    - created_at: 创建时间
    - meta: 元数据字典

    使用流程：
    1. 请求进来时创建RunContext
    2. 传递给GraphRunner开始执行
    3. 执行过程中通过graph_config()获取LangGraph配置
    4. 执行结束后用于记录日志和统计
    """

    # 核心标识
    session_id: str  # 会话ID
    run_id: str     # 运行ID

    # 输入和配置
    user_input: str  # 用户输入
    model_config: Dict[str, Any] = field(default_factory=dict)  # 模型配置

    # 上下文信息
    history_size: int = 0        # 历史消息数量
    is_resume: bool = False     # 是否为恢复运行
    context_summary: str = ""   # 会话上下文摘要

    # 时间戳和元数据
    created_at: str = field(default_factory=utc_now_iso)  # 创建时间
    meta: Dict[str, Any] = field(default_factory=dict)    # 元数据

    def graph_config(self) -> Dict[str, Dict[str, str]]:
        """
        构造 LangGraph 使用的 configurable 配置。

        设计理由：
        1. LangGraph需要configurable参数来配置状态持久化
        2. thread_id用于标识会话级别的状态
        3. run_id用于标识单次运行

        Returns:
            Dict[str, Dict[str, str]]: LangGraph配置字典

        场景：
        - 图执行前配置LangGraph
        - 设置checkpoint thread_id
        """
        return {
            "configurable": {
                "thread_id": self.session_id,
                "run_id": self.run_id,
            }
        }


@dataclass(slots=True)
class RunStateSnapshot:
    """
    运行态快照，供健康检查和调试复用。

    核心职责：
    1. 记录运行的实时状态
    2. 支持健康检查接口查询
    3. 提供调试信息
    4. 供前端展示执行状态

    设计理由：
    1. 实时状态跟踪，方便监控
    2. 完整的上下文信息，便于调试
    3. 结构化数据，便于前端展示
    4. 使用dataclass，易于序列化

    字段说明：
    - run_id: 运行ID
    - session_id: 会话ID
    - status: 当前状态
    - current_phase: 当前阶段
    - title: 标题
    - summary: 摘要
    - error: 错误信息
    - agent_name: 当前执行的Agent名称
    - last_workflow_event: 最后一次工作流事件
    - updated_at: 更新时间
    - meta: 元数据字典

    使用场景：
    - 健康检查接口查询运行状态
    - 前端轮询执行进度
    - 调试时查看运行状态
    - 审批恢复时获取之前的状态
    """

    # 标识字段
    run_id: str      # 运行ID
    session_id: str # 会话ID

    # 状态信息
    status: str        # 当前状态
    current_phase: str = ""  # 当前阶段
    title: str = ""          # 标题
    summary: str = ""        # 摘要
    error: str = ""          # 错误信息
    agent_name: str = ""     # 当前执行的Agent名称

    # 工作流信息
    last_workflow_event: Dict[str, Any] = field(default_factory=dict)  # 最后一次工作流事件

    # 时间戳和元数据
    updated_at: str = field(default_factory=utc_now_iso)  # 更新时间
    meta: Dict[str, Any] = field(default_factory=dict)    # 元数据

