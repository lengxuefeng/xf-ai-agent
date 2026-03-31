# -*- coding: utf-8 -*-
"""
运行态快照存储（Run State Store）。

这是一个内存存储，用于记录每次运行的实时状态。
运行态快照提供了对当前执行状态的查询能力，支持健康检查、状态监控等功能。

设计要点：
1. 内存存储，访问速度快
2. 支持按run_id查询和按session_id查询最新运行
3. 自动清理机制，避免内存无限增长
4. 状态标准化，将各种workflow状态映射为统一的RunStatus
5. 线程安全，支持多线程并发访问

使用场景：
- 健康检查接口查询当前运行状态
- 前端轮询获取执行进度
- 调试和排错，查看当前执行到哪一步
- 审批恢复时检查之前的状态

数据结构：
- _runs: run_id -> RunStateSnapshot（运行快照）
- _latest_by_session: session_id -> run_id（每个会话的最新运行）

清理策略：
1. 当运行数量超过max_runs时触发清理
2. 优先清理已完成的运行（terminal status）
3. 如果还不够，按时间顺序清理最老的运行
"""
from __future__ import annotations

import copy
import threading
from typing import Any, Dict, Optional

from harness.types import RunContext, RunStateSnapshot, RunStatus, utc_now_iso


# Workflow状态到RunStatus的映射表
# 将LangGraph等框架返回的各种状态映射为统一的RunStatus
# 这样可以实现状态管理的一致性，不依赖具体框架
_WORKFLOW_STATUS_TO_RUN_STATUS = {
    "completed": RunStatus.COMPLETED.value,       # 已完成
    "failed": RunStatus.FAILED.value,             # 失败
    "error": RunStatus.FAILED.value,              # 错误（也归为失败）
    "interrupted": RunStatus.INTERRUPTED.value,    # 已中断（审批中）
    "cancelled": RunStatus.CANCELLED.value,        # 已取消
    "pending_approval": RunStatus.WAITING_APPROVAL.value,  # 等待审批
    "blocked": RunStatus.WAITING_APPROVAL.value,  # 已阻塞（等待审批）
    "running": RunStatus.RUNNING.value,           # 运行中
    "streaming": RunStatus.RUNNING.value,         # 流式输出中
    "in_progress": RunStatus.RUNNING.value,       # 进行中
    "info": RunStatus.RUNNING.value,              # 信息状态（视为运行中）
}


class RunStateStore:
    """
    运行态快照存储。

    核心职责：
    1. 存储每次运行的实时状态快照
    2. 提供状态查询接口（按run_id和session_id）
    3. 支持状态更新和元数据附加
    4. 自动清理过期快照，避免内存泄漏

    设计要点：
    1. 使用RLock保护所有共享数据的访问
    2. 返回快照时进行深拷贝，避免外部修改影响内部状态
    3. 自动清理机制，当快照数量超过max_runs时触发
    4. 支持按session_id查询最新运行，方便前端展示
    """

    def __init__(self, max_runs: int = 256) -> None:
        """
        初始化运行态存储。

        Args:
            max_runs: 最大保存的运行快照数量，超过这个数量会触发自动清理
        """
        self._lock = threading.RLock()  # 保护共享数据的锁
        self._max_runs = max(1, int(max_runs or 256))  # 至少保存1个
        self._runs: Dict[str, RunStateSnapshot] = {}  # run_id -> 快照
        self._latest_by_session: Dict[str, str] = {}  # session_id -> 最新run_id

    def _remove_locked(self, run_id: str) -> None:
        """
        删除运行快照（需要在持有锁的情况下调用）。

        设计要点：
        1. 删除快照
        2. 如果这个快照是某个session的最新快照，清除引用

        Args:
            run_id: 要删除的运行ID
        """
        snapshot = self._runs.pop(run_id, None)
        if snapshot and snapshot.session_id:
            # 检查是否是这个session的最新快照
            current_run_id = self._latest_by_session.get(snapshot.session_id)
            if current_run_id == run_id:
                self._latest_by_session.pop(snapshot.session_id, None)

    def _prune_locked(self) -> None:
        """
        清理过期的运行快照（需要在持有锁的情况下调用）。

        清理策略：
        1. 先清理已完成的运行（terminal status）
        2. 如果还不够，清理最老的运行
        3. 保留至少_max_runs个快照

        注意事项：
        - 不会清理正在运行中的快照
        - 按插入顺序清理，保留较新的快照
        """
        if len(self._runs) <= self._max_runs:
            return  # 数量未超标，不需要清理

        # 定义终止状态（这些状态的运行可以被清理）
        terminal_statuses = {
            RunStatus.COMPLETED.value,
            RunStatus.FAILED.value,
            RunStatus.CANCELLED.value,
            RunStatus.INTERRUPTED.value,
            RunStatus.WAITING_APPROVAL.value,
        }

        # 第一轮：清理已终止的运行
        for run_id, snapshot in list(self._runs.items()):
            if len(self._runs) <= self._max_runs:
                break
            if snapshot.status in terminal_statuses:
                self._remove_locked(run_id)

        # 第二轮：如果还不够，清理最老的运行
        while len(self._runs) > self._max_runs:
            oldest_run_id = next(iter(self._runs), None)
            if not oldest_run_id:
                break
            self._remove_locked(oldest_run_id)

    def register_run(self, run_context: RunContext) -> RunStateSnapshot:
        """
        注册一个新的运行。

        设计要点：
        1. 创建初始快照，状态为RUNNING
        2. 更新session的最新运行ID
        3. 触发自动清理
        4. 返回快照的深拷贝，避免外部修改

        Args:
            run_context: 运行上下文对象

        Returns:
            RunStateSnapshot: 创建的运行快照（深拷贝）
        """
        snapshot = RunStateSnapshot(
            run_id=run_context.run_id,
            session_id=run_context.session_id,
            status=RunStatus.RUNNING.value,
            current_phase="run_registered",
            title="运行已注册",
            summary=run_context.user_input[:160],  # 最多160个字符
            meta=copy.deepcopy(run_context.meta),
        )
        with self._lock:
            self._runs[run_context.run_id] = snapshot
            if run_context.session_id:
                self._latest_by_session[run_context.session_id] = run_context.run_id
            self._prune_locked()  # 触发自动清理
            return copy.deepcopy(snapshot)

    def record_workflow_event(self, run_id: str, payload: Dict[str, Any]) -> Optional[RunStateSnapshot]:
        """
        记录工作流事件，更新运行状态。

        设计要点：
        1. 解析workflow事件，提取phase、title、summary、status等字段
        2. 将workflow状态映射为RunStatus
        3. 更新快照的各个字段
        4. 保存最后一次workflow事件的完整内容

        Args:
            run_id: 运行ID
            payload: workflow事件的载荷

        Returns:
            Optional[RunStateSnapshot]: 更新后的快照（深拷贝），如果run_id不存在则返回None

        场景：
        - Agent开始执行时更新phase
        - Agent完成时更新status
        - 工作流产生事件时记录
        """
        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                return None

            # 解析payload
            phase = str(payload.get("phase") or "")
            title = str(payload.get("title") or "")
            summary = str(payload.get("summary") or "")
            workflow_status = str(payload.get("status") or "").strip().lower()
            agent_name = str(payload.get("agent_name") or "")

            # 更新快照字段
            if phase:
                snapshot.current_phase = phase
            if title:
                snapshot.title = title
            if summary:
                snapshot.summary = summary
            if agent_name:
                snapshot.agent_name = agent_name

            # 映射workflow状态到RunStatus
            if workflow_status:
                normalized = _WORKFLOW_STATUS_TO_RUN_STATUS.get(workflow_status)
                if normalized:
                    snapshot.status = normalized

            # 保存最后一次workflow事件
            snapshot.last_workflow_event = copy.deepcopy(payload)
            snapshot.updated_at = utc_now_iso()
            return copy.deepcopy(snapshot)

    def mark_status(
        self,
        run_id: str,
        status: str,
        *,
        phase: str = "",
        title: str = "",
        summary: str = "",
        error: str = "",
        agent_name: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[RunStateSnapshot]:
        """
        标记运行状态。

        设计要点：
        1. 更新指定的字段（只更新非空字段）
        2. 合并meta字段，不覆盖已有字段
        3. 更新时间戳

        Args:
            run_id: 运行ID
            status: 新的状态
            phase: 可选的阶段名称
            title: 可选的标题
            summary: 可选的摘要
            error: 可选的错误信息
            agent_name: 可选的Agent名称
            meta: 可选的元数据字典（会合并到现有meta中）

        Returns:
            Optional[RunStateSnapshot]: 更新后的快照（深拷贝），如果run_id不存在则返回None

        场景：
        - 运行完成时标记为COMPLETED
        - 运行失败时标记为FAILED并记录错误信息
        - 运行被取消时标记为CANCELLED
        """
        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                return None

            incoming_status = str(status or snapshot.status)
            terminal_statuses = {
                RunStatus.COMPLETED.value,
                RunStatus.FAILED.value,
                RunStatus.CANCELLED.value,
                RunStatus.INTERRUPTED.value,
                RunStatus.WAITING_APPROVAL.value,
            }
            current_status = str(snapshot.status or "")

            # 终态幂等收口：
            # - 已取消后不允许被其他终态覆盖；
            # - 其他终态之间默认不互相覆盖；
            # - 允许任何状态升级为 cancelled（例如客户端断连晚到）。
            if current_status in terminal_statuses and incoming_status != current_status:
                if current_status == RunStatus.CANCELLED.value:
                    return copy.deepcopy(snapshot)
                if incoming_status != RunStatus.CANCELLED.value:
                    return copy.deepcopy(snapshot)

            # 更新状态（必填）
            snapshot.status = incoming_status

            # 更新可选字段
            if phase:
                snapshot.current_phase = phase
            if title:
                snapshot.title = title
            if summary:
                snapshot.summary = summary
            if error:
                snapshot.error = error
            if agent_name:
                snapshot.agent_name = agent_name

            # 合并meta字段
            if meta:
                snapshot.meta.update(copy.deepcopy(meta))

            snapshot.updated_at = utc_now_iso()
            return copy.deepcopy(snapshot)

    def attach_meta(self, run_id: str, **meta: Any) -> Optional[RunStateSnapshot]:
        """
        为运行态附加结构化元数据。

        设计要点：
        1. 只更新meta字段，不影响其他字段
        2. 新字段会添加，旧字段会被覆盖
        3. 返回更新后的快照

        Args:
            run_id: 运行ID
            **meta: 要附加的元数据（关键字参数）

        Returns:
            Optional[RunStateSnapshot]: 更新后的快照（深拷贝），如果run_id不存在则返回None

        场景：
        - 记录路由决策信息
        - 记录性能指标
        - 记录Agent执行信息
        """
        if not meta:
            return self.get(run_id)  # 没有meta，直接返回当前快照

        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                return None

            # 添加或更新meta字段
            for key, value in meta.items():
                snapshot.meta[key] = copy.deepcopy(value)

            snapshot.updated_at = utc_now_iso()
            return copy.deepcopy(snapshot)

    def get(self, run_id: str) -> Optional[RunStateSnapshot]:
        """
        获取运行快照。

        Args:
            run_id: 运行ID

        Returns:
            Optional[RunStateSnapshot]: 运行快照（深拷贝），如果不存在则返回None
        """
        with self._lock:
            snapshot = self._runs.get(run_id)
            return copy.deepcopy(snapshot) if snapshot else None

    def get_latest_for_session(self, session_id: str) -> Optional[RunStateSnapshot]:
        """
        获取会话的最新运行快照。

        设计要点：
        1. 先查找session的最新run_id
        2. 再根据run_id获取快照

        Args:
            session_id: 会话ID

        Returns:
            Optional[RunStateSnapshot]: 最新运行的快照（深拷贝），如果不存在则返回None

        场景：
        - 前端查询会话状态
        - 健康检查接口
        - 恢复审批时获取之前的状态
        """
        with self._lock:
            run_id = self._latest_by_session.get(session_id)
            if not run_id:
                return None
            snapshot = self._runs.get(run_id)
            return copy.deepcopy(snapshot) if snapshot else None

    def remove(self, run_id: str) -> None:
        """
        删除运行快照。

        Args:
            run_id: 要删除的运行ID

        场景：
        - 手动清理过期的快照
        - 测试和调试
        """
        with self._lock:
            self._remove_locked(run_id)


# 全局唯一的运行态存储实例
run_state_store = RunStateStore()
