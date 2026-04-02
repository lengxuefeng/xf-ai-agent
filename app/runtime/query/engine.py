# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional

from harness.types import RunContext
from runtime.query.budgets import QueryBudget
from runtime.query.events import build_runtime_event
from runtime.query.models import QueryState, RuntimeEvent


class RuntimeEngine:
    """Runtime-first 查询执行内核骨架。"""

    LEGACY_EXECUTOR_NAME = "legacy_langgraph_supervisor"
    RULE_INTERCEPT_EXECUTOR_NAME = "rule_intercept"
    RESUME_EXECUTOR_NAME = "legacy_langgraph_resume"

    def bootstrap_query(
        self,
        run_context: RunContext,
        *,
        executor_name: str = LEGACY_EXECUTOR_NAME,
        budget: Optional[QueryBudget] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> QueryState:
        query_state = QueryState(
            query_id=str(run_context.run_id or "").strip(),
            session_id=str(run_context.session_id or "").strip(),
            run_id=str(run_context.run_id or "").strip(),
            user_input=str(run_context.user_input or ""),
            executor_name=str(executor_name or self.LEGACY_EXECUTOR_NAME).strip(),
            budget=budget or QueryBudget(),
            meta={
                "request_id": str(run_context.request_id or "").strip(),
                "history_size": int(run_context.history_size or 0),
                "is_resume": bool(run_context.is_resume),
                **dict(meta or {}),
            },
        )
        query_state.touch()
        return query_state

    def mark_running(self, query_state: QueryState) -> QueryState:
        query_state.status = "running"
        query_state.touch()
        return query_state

    def finalize_query(
        self,
        query_state: QueryState,
        *,
        status: str,
        final_text: str = "",
        error: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> QueryState:
        query_state.status = str(status or "").strip() or "completed"
        query_state.final_text = str(final_text or "")
        query_state.error = str(error or "")
        if meta:
            query_state.meta.update(dict(meta))
        query_state.finished_at = query_state.finished_at or query_state.updated_at
        query_state.touch()
        query_state.finished_at = query_state.updated_at
        return query_state

    def build_runtime_event(
        self,
        query_state: QueryState,
        *,
        phase: str,
        title: str,
        summary: str = "",
        status: str = "info",
        meta: Optional[Dict[str, Any]] = None,
    ) -> RuntimeEvent:
        merged_meta: Dict[str, Any] = {
            "query_id": query_state.query_id,
            "run_id": query_state.run_id,
            "executor_name": query_state.executor_name,
        }
        if meta:
            merged_meta.update(dict(meta))
        return build_runtime_event(
            event_type="query_runtime",
            phase=phase,
            title=title,
            summary=summary,
            status=status,
            meta=merged_meta,
        )


runtime_engine = RuntimeEngine()
