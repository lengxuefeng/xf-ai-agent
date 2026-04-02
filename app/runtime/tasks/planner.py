# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Iterable, List

from runtime.tasks.models import TaskPlan, TaskSpec


class RuntimeTaskPlanner:
    """把现有 planner 输出投影成 Runtime TaskPlan。"""

    def build_plan(
        self,
        *,
        steps: Iterable[str] | None,
        task_list: Iterable[Dict[str, Any]] | None,
        source: str,
        replan_reason: str = "",
        planner_usage: Dict[str, int] | None = None,
    ) -> TaskPlan:
        normalized_steps = [str(step or "").strip() for step in (steps or []) if str(step or "").strip()]
        tasks: list[TaskSpec] = []
        for task in task_list or []:
            if not isinstance(task, dict):
                continue
            tasks.append(
                TaskSpec(
                    task_id=str(task.get("id") or "").strip(),
                    input=str(task.get("input") or "").strip(),
                    agent=str(task.get("agent") or "").strip(),
                    depends_on=[str(item or "").strip() for item in (task.get("depends_on") or []) if str(item or "").strip()],
                    status=str(task.get("status") or "pending").strip() or "pending",
                    result=str(task.get("result") or "").strip(),
                )
            )
        return TaskPlan(
            source=str(source or "").strip(),
            steps=normalized_steps,
            tasks=tasks,
            replan_reason=str(replan_reason or "").strip(),
            meta={
                "task_count": len(tasks),
                "step_count": len(normalized_steps),
                "planner_usage": dict(planner_usage or {}),
            },
        )


runtime_task_planner = RuntimeTaskPlanner()

