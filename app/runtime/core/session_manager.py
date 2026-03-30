# -*- coding: utf-8 -*-
"""
运行会话管理器（Runtime Session Manager）。

这是运行时的核心管理类，统一管理一次运行的生命周期。
它整合了取消管理、状态存储、实时流等多个模块，提供统一的管理接口。

设计要点：
1. 统一的生命周期管理：从注册到清理
2. 集成的状态管理：取消、状态、实时流等
3. 简化的接口：上层模块只需调用这个管理器
4. 自动化的关系建立：session和run的父子关系自动建立

使用场景：
- GraphRunner在开始执行前调用register_run
- Agent执行时调用mark_running/mark_completed等
- 需要实时流时调用register_live_stream_callback
- 执行完成后调用cleanup_run

架构说明：
- 本模块是对runtime/core各模块的整合和封装
- 提供统一的管理接口，简化上层代码
- 确保各模块的协调工作，避免遗漏
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from runtime.core.cancel_manager import runtime_cancel_manager
from runtime.core.live_stream_bus import live_stream_bus
from runtime.core.run_context import build_run_context
from runtime.core.run_state_store import run_state_store
from runtime.core.workflow_event_bus import parse_workflow_event_chunk
from runtime.types import RunContext, RunStatus


class RuntimeSessionManager:
    """
    统一管理一次运行的生命周期。

    核心职责：
    1. 创建运行上下文（RunContext）
    2. 注册运行，建立session和run的父子关系
    3. 管理运行状态（运行中、完成、失败、中断等）
    4. 管理实时流回调
    5. 清理运行资源

    设计要点：
    1. 提供完整的生命周期管理接口
    2. 自动处理模块间的依赖关系
    3. 简化上层代码，提高可维护性
    4. 支持状态标记和元数据附加
    """

    def create_run_context(
        self,
        *,
        session_id: str,
        user_input: str,
        model_config: Optional[Dict[str, Any]] = None,
        history_messages: Optional[List[Dict[str, Any]]] = None,
        session_context: Optional[Dict[str, Any]] = None,
        is_resume: bool = False,
        run_id: str = "",
    ) -> RunContext:
        """
        创建运行上下文对象。

        设计要点：
        1. 委托给build_run_context函数处理
        2. 提供统一的接口，避免直接调用底层函数

        Args:
            session_id: 会话ID
            user_input: 用户输入
            model_config: 模型配置
            history_messages: 历史消息
            session_context: 会话上下文
            is_resume: 是否为恢复运行
            run_id: 可选的run_id

        Returns:
            RunContext: 构建好的运行上下文对象
        """
        return build_run_context(
            session_id=session_id,
            user_input=user_input,
            model_config=model_config,
            history_messages=history_messages,
            session_context=session_context,
            is_resume=is_resume,
            run_id=run_id,
        )

    def register_run(self, run_context: RunContext) -> None:
        """
        注册一个运行，建立必要的上下文和关系。

        设计要点：
        1. 在取消管理器中注册run的取消令牌
        2. 建立session和run的父子关系（级联取消）
        3. 在状态存储中注册运行快照

        Args:
            run_context: 运行上下文对象

        场景：
        - GraphRunner开始执行前调用
        - Agent开始执行前调用
        """
        # 在取消管理器中注册run
        runtime_cancel_manager.register_request(run_context.run_id)

        # 建立session和run的父子关系
        # 这样取消session时会自动取消所有run
        if run_context.session_id:
            runtime_cancel_manager.link_request(run_context.session_id, run_context.run_id)

        # 在状态存储中注册运行
        run_state_store.register_run(run_context)

    def bind_run(self, run_context: RunContext):
        """
        绑定当前请求ID到上下文，用于异步调用链。

        设计要点：
        1. 使用ContextVar在异步调用链中传递request_id
        2. 返回上下文管理器，支持with语句

        Args:
            run_context: 运行上下文对象

        Returns:
            上下文管理器，可以用于with语句

        场景：
        - Agent执行前绑定，所有子工具调用都能获取到request_id
        - 工具调用时绑定，工具内部的异步代码都能获取到request_id
        """
        return runtime_cancel_manager.bind_request(run_context.run_id)

    def register_live_stream_callback(
        self,
        run_context: RunContext,
        callback: Callable[[Dict[str, Any]], None],
        *,
        enabled: bool = True,
    ) -> None:
        """
        注册实时流回调函数。

        设计要点：
        1. 当子Agent产生输出时，回调函数会被调用
        2. 回调函数接收包含run_id、agent_name、content的字典
        3. 支持通过enabled参数控制是否注册

        Args:
            run_context: 运行上下文对象
            callback: 回调函数，签名为callback(dict) -> None
            enabled: 是否启用实时流

        场景：
        - GraphRunner在开始执行前注册回调
        - 回调函数将实时内容推送到SSE流
        """
        if enabled:
            live_stream_bus.register_callback(run_context.run_id, callback)

    def unregister_live_stream_callback(
        self,
        run_context: RunContext,
        *,
        enabled: bool = True,
    ) -> None:
        """
        注销实时流回调函数。

        设计要点：
        1. 运行结束后应该注销回调，避免内存泄漏
        2. 支持通过enabled参数控制是否注销

        Args:
            run_context: 运行上下文对象
            enabled: 是否启用实时流注销

        场景：
        - 运行结束后注销回调
        - 运行被取消时注销回调
        """
        if enabled:
            live_stream_bus.unregister_callback(run_context.run_id)

    def record_workflow_event_chunk(self, run_context: RunContext, chunk: str) -> None:
        """
        记录工作流事件。

        设计要点：
        1. 从SSE chunk中提取workflow_event
        2. 更新运行状态快照

        Args:
            run_context: 运行上下文对象
            chunk: SSE事件chunk

        场景：
        - 处理workflow_event SSE事件
        - 更新运行状态
        """
        payload = parse_workflow_event_chunk(chunk)
        if payload:
            run_state_store.record_workflow_event(run_context.run_id, payload)

    def mark_running(
        self,
        run_context: RunContext,
        *,
        phase: str = "",
        summary: str = "",
        title: str = "",
    ) -> None:
        """
        标记运行为运行中状态。

        设计要点：
        1. 更新状态为RUNNING
        2. 可选更新phase、summary、title

        Args:
            run_context: 运行上下文对象
            phase: 可选的阶段名称
            summary: 可选的摘要
            title: 可选的标题

        场景：
        - Agent开始执行时标记
        - 进入新阶段时更新phase
        """
        run_state_store.mark_status(
            run_context.run_id,
            RunStatus.RUNNING.value,
            phase=phase,
            summary=summary,
            title=title,
        )

    def mark_completed(
        self,
        run_context: RunContext,
        *,
        phase: str = "",
        summary: str = "",
        title: str = "",
    ) -> None:
        """
        标记运行为已完成状态。

        设计要点：
        1. 更新状态为COMPLETED
        2. 可选更新phase、summary、title

        Args:
            run_context: 运行上下文对象
            phase: 可选的阶段名称
            summary: 可选的摘要
            title: 可选的标题

        场景：
        - Agent执行完成时标记
        - 整个运行完成时标记
        """
        run_state_store.mark_status(
            run_context.run_id,
            RunStatus.COMPLETED.value,
            phase=phase,
            summary=summary,
            title=title,
        )

    def mark_interrupted(
        self,
        run_context: RunContext,
        *,
        phase: str = "",
        summary: str = "",
        title: str = "",
    ) -> None:
        """
        标记运行为已中断状态。

        设计要点：
        1. 更新状态为INTERRUPTED
        2. 通常表示等待审批
        3. 可选更新phase、summary、title

        Args:
            run_context: 运行上下文对象
            phase: 可选的阶段名称
            summary: 可选的摘要
            title: 可选的标题

        场景：
        - Agent触发中断，等待审批时标记
        - SQL/代码执行前触发审批时标记
        """
        run_state_store.mark_status(
            run_context.run_id,
            RunStatus.INTERRUPTED.value,
            phase=phase,
            summary=summary,
            title=title,
        )

    def mark_failed(
        self,
        run_context: RunContext,
        *,
        phase: str = "",
        summary: str = "",
        title: str = "",
        error: str = "",
    ) -> None:
        """
        标记运行为失败状态。

        设计要点：
        1. 更新状态为FAILED
        2. 记录错误信息
        3. 可选更新phase、summary、title

        Args:
            run_context: 运行上下文对象
            phase: 可选的阶段名称
            summary: 可选的摘要
            title: 可选的标题
            error: 错误信息

        场景：
        - Agent执行失败时标记
        - 异常抛出时标记
        - 超时时标记
        """
        run_state_store.mark_status(
            run_context.run_id,
            RunStatus.FAILED.value,
            phase=phase,
            summary=summary,
            title=title,
            error=error,
        )

    def cancel_run(self, run_context: RunContext, *, summary: str = "") -> None:
        """
        取消运行。

        设计要点：
        1. 触发取消信号
        2. 更新状态为CANCELLED
        3. 可选更新summary

        Args:
            run_context: 运行上下文对象
            summary: 可选的摘要

        场景：
        - 用户点击停止按钮
        - 客户端断开连接
        - 系统超时
        """
        runtime_cancel_manager.cancel_request(run_context.run_id)
        run_state_store.mark_status(
            run_context.run_id,
            RunStatus.CANCELLED.value,
            phase="cancelled",
            summary=summary or "运行已取消",
        )

    def cleanup_run(self, run_context: RunContext) -> None:
        """
        清理运行资源。

        设计要点：
        1. 清理取消令牌
        2. 释放相关资源
        3. 避免内存泄漏

        Args:
            run_context: 运行上下文对象

        场景：
        - 运行正常结束时清理
        - 运行被取消时清理
        - 异常退出时清理（在finally中）
        """
        runtime_cancel_manager.cleanup_request(run_context.run_id)

    def attach_meta(self, run_context: RunContext, **meta: Any) -> None:
        """
        附加元数据到运行快照。

        设计要点：
        1. 可以记录任意结构化的元数据
        2. 支持添加或更新字段

        Args:
            run_context: 运行上下文对象
            **meta: 要附加的元数据（关键字参数）

        场景：
        - 记录路由决策
        - 记录性能指标
        - 记录Agent执行信息
        """
        run_state_store.attach_meta(run_context.run_id, **meta)


# 全局唯一的运行会话管理器实例
runtime_session_manager = RuntimeSessionManager()
