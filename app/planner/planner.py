import time
from typing import Any, Dict, List
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from prompts.prompt_loader import load_prompt_template
from supervisor.state import GraphState
from supervisor.supervisor_rule_support import split_query_clauses
from config.constants.workflow_constants import TaskStatus
from config.runtime_settings import WORKFLOW_MAX_REPLANS
from common.utils.custom_logger import get_logger

log = get_logger(__name__)


class Plan(BaseModel):
    """Planner 输出结构。"""

    steps: List[str] = Field(default_factory=list, description="原子的、可独立执行的子任务步骤列表")


class PlannerNode:
    SIMPLE_REQUEST_CONNECTOR_RE = re.compile(
        r"(?:然后|接着|并且|以及|同时|顺便|后面|随后|再|\band\b|\bthen\b|(?<![A-Za-z0-9])和(?![A-Za-z0-9]))",
        flags=re.IGNORECASE,
    )

    @staticmethod
    def _system_prompt() -> str:
        return load_prompt_template("agent_prompts/templates/planner_steps_system.txt")

    @staticmethod
    def _replan_system_prompt() -> str:
        return load_prompt_template("agent_prompts/templates/planner_replan_system.txt")

    @staticmethod
    def _latest_user_text(state: GraphState) -> str:
        from supervisor.supervisor import _latest_human_message

        return _latest_human_message(state.get("messages", []) or [])

    @staticmethod
    def _normalize_steps(steps: List[str], fallback_text: str) -> List[str]:
        normalized: List[str] = []
        for step in steps or []:
            text = str(step or "").strip()
            if text and text not in normalized:
                normalized.append(text)
        if normalized:
            return normalized
        fallback = str(fallback_text or "").strip()
        return [fallback] if fallback else []

    @staticmethod
    def _fallback_steps(user_text: str) -> List[str]:
        clauses = split_query_clauses(user_text)
        steps = [clause.strip() for clause in clauses if str(clause or "").strip()]
        return PlannerNode._normalize_steps(steps, user_text)

    @staticmethod
    def _task_sequence(tasks: List[Dict[str, Any]]) -> int:
        max_seq = 0
        for task in tasks or []:
            task_id = str(task.get("id") or "").strip()
            matched = re.search(r"(\d+)", task_id)
            if matched:
                max_seq = max(max_seq, int(matched.group(1)))
        return max_seq + 1 if max_seq else 1

    @staticmethod
    def _collect_completed_step_results(state: GraphState) -> List[Dict[str, Any]]:
        memory = dict(state.get("memory") or {})
        step_results = list(memory.get("step_results") or [])
        if step_results:
            return [dict(item) for item in step_results if isinstance(item, dict)]

        completed: List[Dict[str, Any]] = []
        for task in list(state.get("task_list") or []):
            if str(task.get("status") or "") != TaskStatus.DONE.value:
                continue
            completed.append(
                {
                    "step": str(task.get("input") or "").strip(),
                    "agent": str(task.get("agent") or "").strip(),
                    "result": str(task.get("result") or "").strip(),
                    "status": TaskStatus.DONE.value,
                }
            )
        return completed

    @staticmethod
    def _build_replan_payload(state: GraphState, user_text: str) -> Dict[str, Any]:
        task_list = [dict(task) for task in list(state.get("task_list") or [])]
        completed_results = PlannerNode._collect_completed_step_results(state)
        remaining_goals = [
            str(task.get("input") or "").strip()
            for task in task_list
            if str(task.get("status") or "").strip() not in {TaskStatus.DONE.value, TaskStatus.CANCELLED.value}
            and str(task.get("input") or "").strip()
        ]
        return {
            "original_user_request": str(
                (state.get("memory") or {}).get("original_user_request") or user_text
            ).strip(),
            "completed_results": completed_results,
            "remaining_goals": remaining_goals,
            "replan_reason": str(state.get("replan_reason") or "").strip(),
            "task_list": task_list,
        }

    @staticmethod
    def _append_new_tasks(
        *,
        existing_tasks: List[Dict[str, Any]],
        steps: List[str],
    ) -> List[Dict[str, Any]]:
        next_sequence = PlannerNode._task_sequence(existing_tasks)
        known_inputs = {
            str(task.get("input") or "").strip()
            for task in existing_tasks
            if str(task.get("input") or "").strip()
        }
        appended: List[Dict[str, Any]] = []
        for step in steps:
            normalized_step = str(step or "").strip()
            if (not normalized_step) or normalized_step in known_inputs:
                continue
            appended.append(
                {
                    "id": f"t{next_sequence}",
                    "agent": "",
                    "input": normalized_step,
                    "depends_on": [],
                    "status": TaskStatus.PENDING.value,
                    "result": None,
                }
            )
            known_inputs.add(normalized_step)
            next_sequence += 1
        return appended

    @staticmethod
    def _is_simple_request(user_text: str) -> bool:
        text = str(user_text or "").strip()
        if not text:
            return True
        return not PlannerNode.SIMPLE_REQUEST_CONNECTOR_RE.search(text)

    @staticmethod
    async def planner_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
        """Planner 节点：使用 fast model 异步拆解原子步骤列表。"""
        user_text = PlannerNode._latest_user_text(state)
        started_at = time.perf_counter()
        replan_reason = str(state.get("replan_reason") or "").strip()
        existing_tasks = [dict(task) for task in list(state.get("task_list") or [])]
        preserved_tasks = [
            task for task in existing_tasks
            if str(task.get("status") or "").strip() in {
                TaskStatus.DONE.value,
                TaskStatus.ERROR.value,
                TaskStatus.CANCELLED.value,
                TaskStatus.PENDING_APPROVAL.value,
            }
        ]

        if replan_reason:
            replan_payload = PlannerNode._build_replan_payload(state, user_text)
            planner_source = "llm_replan"
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", PlannerNode._replan_system_prompt()),
                    (
                        "user",
                        "用户原始目标：\n{original_user_request}\n\n"
                        "已完成步骤结果：\n{completed_results}\n\n"
                        "尚未完成目标：\n{remaining_goals}\n\n"
                        "当前失败原因：\n{replan_reason}\n\n"
                        "只返回剩余步骤 JSON。",
                    ),
                ]
            )
            try:
                chain = prompt | model.with_structured_output(Plan)
                plan_result: Plan = await chain.ainvoke(
                    {
                        "original_user_request": replan_payload["original_user_request"],
                        "completed_results": replan_payload["completed_results"],
                        "remaining_goals": replan_payload["remaining_goals"],
                        "replan_reason": replan_payload["replan_reason"],
                    },
                    config=config,
                )
                steps = PlannerNode._normalize_steps(
                    plan_result.steps,
                    "\n".join(replan_payload["remaining_goals"]),
                )
                if not steps:
                    raise ValueError("planner replan returned empty steps")
            except Exception as exc:
                planner_source = "fallback_replan"
                log.warning(f"Planner replan failed, fallback to remaining goals: {exc}")
                steps = PlannerNode._normalize_steps(
                    list(replan_payload["remaining_goals"]),
                    replan_payload["original_user_request"],
                )
            task_list = preserved_tasks + PlannerNode._append_new_tasks(
                existing_tasks=preserved_tasks,
                steps=steps,
            )
        elif PlannerNode._is_simple_request(user_text):
            steps = PlannerNode._normalize_steps([user_text], user_text)
            planner_source = "passthrough_single"
            task_list = [
                {
                    "id": f"t{index + 1}",
                    "agent": "",
                    "input": step,
                    "depends_on": [],
                    "status": TaskStatus.PENDING.value,
                    "result": None,
                }
                for index, step in enumerate(steps)
            ]
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", PlannerNode._system_prompt()),
                    ("user", "用户请求：\n{user_input}\n\n只返回 JSON。"),
                ]
            )

            planner_source = "llm"
            try:
                chain = prompt | model.with_structured_output(Plan)
                plan_result: Plan = await chain.ainvoke({"user_input": user_text}, config=config)
                steps = PlannerNode._normalize_steps(plan_result.steps, user_text)
                if not steps:
                    raise ValueError("planner returned empty steps")
            except Exception as exc:
                planner_source = "fallback_split"
                log.warning(f"Planner structured output failed, fallback to rule split: {exc}")
                steps = PlannerNode._fallback_steps(user_text)
            task_list = [
                {
                    "id": f"t{index + 1}",
                    "agent": "",
                    "input": step,
                    "depends_on": [],
                    "status": TaskStatus.PENDING.value,
                    "result": None,
                }
                for index, step in enumerate(steps)
            ]

        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        memory = dict(state.get("memory") or {})
        memory["original_user_request"] = str(memory.get("original_user_request") or user_text).strip()
        memory["planned_steps"] = list(steps)
        memory["plan_source"] = planner_source
        memory["completed_step_results"] = PlannerNode._collect_completed_step_results(state)

        log.info(f"⏱️ Planner [{planner_source}] 耗时: {elapsed_ms}ms, steps={steps}")
        return {
            "plan": list(steps),
            "task_list": task_list,
            "planner_source": planner_source,
            "planner_elapsed_ms": elapsed_ms,
            "memory": memory,
            "active_tasks": task_list,
            "current_task": "",
            "current_task_id": "",
            "current_step_input": "",
            "current_step_agent": "",
            "executor_active": False,
            "interrupt_payload": None,
            "error_message": None,
            "error_detail": None,
            "last_step_result": "",
            "last_step_status": "",
            "last_step_error": "",
            "replan_reason": "",
            "max_replans": int(state.get("max_replans") or WORKFLOW_MAX_REPLANS),
            "next": "dispatch_node",
        }

    @staticmethod
    async def parent_planner_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
        """兼容旧引用，统一转到新的 planner_node。"""
        return await PlannerNode.planner_node(state, model=model, config=config)
