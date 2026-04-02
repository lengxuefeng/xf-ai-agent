# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

from harness.types import utc_now_iso
from runtime.query.budgets import QueryBudget


@dataclass(slots=True)
class ToolCallEnvelope:
    """运行时统一工具调用封装。"""

    call_id: str
    tool_name: str
    args: Dict[str, Any] = field(default_factory=dict)
    source_agent: str = ""
    status: str = "pending"
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ToolResultEnvelope:
    """运行时统一工具结果封装。"""

    call_id: str
    tool_name: str
    ok: bool
    result: Any = None
    error: str = ""
    duration_ms: int = 0
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class WorkerResultEnvelope:
    """运行时 Worker 结果封装。"""

    worker_name: str
    task_id: str = ""
    status: str = "completed"
    output: str = ""
    error: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class RuntimeEvent:
    """运行时统一事件结构。"""

    event_type: str
    phase: str
    title: str
    summary: str = ""
    status: str = "info"
    timestamp: str = field(default_factory=utc_now_iso)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class QueryState:
    """单次 Query Runtime 的统一状态快照。"""

    query_id: str
    session_id: str
    run_id: str
    user_input: str
    executor_name: str
    status: str = "initialized"
    budget: QueryBudget = field(default_factory=QueryBudget)
    started_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    finished_at: str = ""
    final_text: str = ""
    error: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = utc_now_iso()

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["budget"] = self.budget.to_dict()
        return payload

