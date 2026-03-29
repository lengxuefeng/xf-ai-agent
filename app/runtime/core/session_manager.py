# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from runtime.core.cancel_manager import runtime_cancel_manager
from runtime.core.live_stream_bus import live_stream_bus
from runtime.core.run_context import build_run_context
from runtime.core.run_state_store import run_state_store
from runtime.core.workflow_event_bus import parse_workflow_event_chunk
from runtime.types import RunContext, RunStatus


class RuntimeSessionManager:
    """统一管理一次运行的生命周期。"""

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
        runtime_cancel_manager.register_request(run_context.run_id)
        if run_context.session_id:
            runtime_cancel_manager.link_request(run_context.session_id, run_context.run_id)
        run_state_store.register_run(run_context)

    def bind_run(self, run_context: RunContext):
        return runtime_cancel_manager.bind_request(run_context.run_id)

    def register_live_stream_callback(
        self,
        run_context: RunContext,
        callback: Callable[[Dict[str, Any]], None],
        *,
        enabled: bool = True,
    ) -> None:
        if enabled:
            live_stream_bus.register_callback(run_context.run_id, callback)

    def unregister_live_stream_callback(
        self,
        run_context: RunContext,
        *,
        enabled: bool = True,
    ) -> None:
        if enabled:
            live_stream_bus.unregister_callback(run_context.run_id)

    def record_workflow_event_chunk(self, run_context: RunContext, chunk: str) -> None:
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
        run_state_store.mark_status(
            run_context.run_id,
            RunStatus.FAILED.value,
            phase=phase,
            summary=summary,
            title=title,
            error=error,
        )

    def cancel_run(self, run_context: RunContext, *, summary: str = "") -> None:
        runtime_cancel_manager.cancel_request(run_context.run_id)
        run_state_store.mark_status(
            run_context.run_id,
            RunStatus.CANCELLED.value,
            phase="cancelled",
            summary=summary or "运行已取消",
        )

    def cleanup_run(self, run_context: RunContext) -> None:
        runtime_cancel_manager.cleanup_request(run_context.run_id)

    def attach_meta(self, run_context: RunContext, **meta: Any) -> None:
        run_state_store.attach_meta(run_context.run_id, **meta)


runtime_session_manager = RuntimeSessionManager()
