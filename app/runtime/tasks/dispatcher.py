# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List

from config.constants.workflow_constants import TaskStatus


class RuntimeTaskDispatcher:
    """统一的下一步任务选择器。"""

    def dispatch_next(
        self,
        *,
        plan: List[str] | None,
        task_list: List[Dict[str, Any]] | None,
        resolve_agent,
    ) -> Dict[str, Any]:
        remaining_plan = [str(item or "").strip() for item in list(plan or []) if str(item or "").strip()]
        updated_task_list = [dict(task) for task in list(task_list or [])]

        next_task = ""
        next_task_id = ""
        next_task_agent = ""

        if remaining_plan:
            next_task = remaining_plan.pop(0)
            for task in updated_task_list:
                if str(task.get("status") or "").strip() != TaskStatus.PENDING.value:
                    continue
                candidate_input = str(task.get("input") or "").strip()
                next_task_id = str(task.get("id") or "").strip()
                task["status"] = TaskStatus.DISPATCHED.value
                next_task_agent = str(task.get("agent") or "").strip() or resolve_agent(candidate_input or next_task)
                if next_task_agent:
                    task["agent"] = next_task_agent
                break

        if not next_task:
            next_task = "END_TASK"

        return {
            "plan": remaining_plan,
            "task_list": updated_task_list,
            "next_task": next_task,
            "next_task_id": next_task_id,
            "next_task_agent": next_task_agent,
            "dispatch_completed": bool(next_task_id),
        }


runtime_task_dispatcher = RuntimeTaskDispatcher()

