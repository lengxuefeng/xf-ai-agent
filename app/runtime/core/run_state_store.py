# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import threading
from typing import Any, Dict, Optional

from runtime.types import RunContext, RunStateSnapshot, RunStatus, utc_now_iso


_WORKFLOW_STATUS_TO_RUN_STATUS = {
    "completed": RunStatus.COMPLETED.value,
    "failed": RunStatus.FAILED.value,
    "error": RunStatus.FAILED.value,
    "interrupted": RunStatus.INTERRUPTED.value,
    "cancelled": RunStatus.CANCELLED.value,
    "pending_approval": RunStatus.WAITING_APPROVAL.value,
    "blocked": RunStatus.WAITING_APPROVAL.value,
    "running": RunStatus.RUNNING.value,
    "streaming": RunStatus.RUNNING.value,
    "in_progress": RunStatus.RUNNING.value,
    "info": RunStatus.RUNNING.value,
}


class RunStateStore:
    """运行态快照存储。"""

    def __init__(self, max_runs: int = 256) -> None:
        self._lock = threading.RLock()
        self._max_runs = max(1, int(max_runs or 256))
        self._runs: Dict[str, RunStateSnapshot] = {}
        self._latest_by_session: Dict[str, str] = {}

    def _remove_locked(self, run_id: str) -> None:
        snapshot = self._runs.pop(run_id, None)
        if snapshot and snapshot.session_id:
            current_run_id = self._latest_by_session.get(snapshot.session_id)
            if current_run_id == run_id:
                self._latest_by_session.pop(snapshot.session_id, None)

    def _prune_locked(self) -> None:
        if len(self._runs) <= self._max_runs:
            return

        terminal_statuses = {
            RunStatus.COMPLETED.value,
            RunStatus.FAILED.value,
            RunStatus.CANCELLED.value,
            RunStatus.INTERRUPTED.value,
            RunStatus.WAITING_APPROVAL.value,
        }

        for run_id, snapshot in list(self._runs.items()):
            if len(self._runs) <= self._max_runs:
                break
            if snapshot.status in terminal_statuses:
                self._remove_locked(run_id)

        while len(self._runs) > self._max_runs:
            oldest_run_id = next(iter(self._runs), None)
            if not oldest_run_id:
                break
            self._remove_locked(oldest_run_id)

    def register_run(self, run_context: RunContext) -> RunStateSnapshot:
        snapshot = RunStateSnapshot(
            run_id=run_context.run_id,
            session_id=run_context.session_id,
            status=RunStatus.RUNNING.value,
            current_phase="run_registered",
            title="运行已注册",
            summary=run_context.user_input[:160],
            meta=copy.deepcopy(run_context.meta),
        )
        with self._lock:
            self._runs[run_context.run_id] = snapshot
            if run_context.session_id:
                self._latest_by_session[run_context.session_id] = run_context.run_id
            self._prune_locked()
            return copy.deepcopy(snapshot)

    def record_workflow_event(self, run_id: str, payload: Dict[str, Any]) -> Optional[RunStateSnapshot]:
        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                return None

            phase = str(payload.get("phase") or "")
            title = str(payload.get("title") or "")
            summary = str(payload.get("summary") or "")
            workflow_status = str(payload.get("status") or "").strip().lower()
            agent_name = str(payload.get("agent_name") or "")

            if phase:
                snapshot.current_phase = phase
            if title:
                snapshot.title = title
            if summary:
                snapshot.summary = summary
            if agent_name:
                snapshot.agent_name = agent_name
            if workflow_status:
                normalized = _WORKFLOW_STATUS_TO_RUN_STATUS.get(workflow_status)
                if normalized:
                    snapshot.status = normalized
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
        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                return None
            snapshot.status = str(status or snapshot.status)
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
            if meta:
                snapshot.meta.update(copy.deepcopy(meta))
            snapshot.updated_at = utc_now_iso()
            return copy.deepcopy(snapshot)

    def attach_meta(self, run_id: str, **meta: Any) -> Optional[RunStateSnapshot]:
        """为运行态附加结构化元数据。"""
        if not meta:
            return self.get(run_id)
        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                return None
            for key, value in meta.items():
                snapshot.meta[key] = copy.deepcopy(value)
            snapshot.updated_at = utc_now_iso()
            return copy.deepcopy(snapshot)

    def get(self, run_id: str) -> Optional[RunStateSnapshot]:
        with self._lock:
            snapshot = self._runs.get(run_id)
            return copy.deepcopy(snapshot) if snapshot else None

    def get_latest_for_session(self, session_id: str) -> Optional[RunStateSnapshot]:
        with self._lock:
            run_id = self._latest_by_session.get(session_id)
            if not run_id:
                return None
            snapshot = self._runs.get(run_id)
            return copy.deepcopy(snapshot) if snapshot else None

    def remove(self, run_id: str) -> None:
        with self._lock:
            self._remove_locked(run_id)


run_state_store = RunStateStore()
