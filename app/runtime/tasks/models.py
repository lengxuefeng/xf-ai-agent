# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from harness.types import utc_now_iso


@dataclass(slots=True)
class TaskSpec:
    """Runtime Task 规格。"""

    task_id: str
    input: str
    agent: str = ""
    depends_on: List[str] = field(default_factory=list)
    status: str = "pending"
    result: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["id"] = payload.pop("task_id")
        return payload


@dataclass(slots=True)
class TaskPlan:
    """Runtime 任务计划。"""

    source: str
    steps: List[str] = field(default_factory=list)
    tasks: List[TaskSpec] = field(default_factory=list)
    replan_reason: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "steps": list(self.steps),
            "tasks": [task.to_dict() for task in self.tasks],
            "replan_reason": self.replan_reason,
            "created_at": self.created_at,
            "meta": dict(self.meta),
        }

