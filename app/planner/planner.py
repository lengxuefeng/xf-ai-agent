import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_serializer, model_validator

from prompts.prompt_loader import load_prompt_template
from runtime.tasks import runtime_task_planner
from supervisor.state import GraphState
from supervisor.supervisor_rule_support import split_query_clauses
from config.constants.workflow_constants import RouteStrategy, TaskStatus
from config.runtime_settings import ROUTER_POLICY_CONFIG, WORKFLOW_MAX_REPLANS
from common.utils.custom_logger import get_logger

log = get_logger(__name__)


class Plan(BaseModel):
    """Planner 输出结构。"""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    plan: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("plan", "steps"),
        description="原子的、可独立执行的子任务步骤列表",
    )

    @property
    def steps(self) -> List[str]:
        return list(self.plan)

    def to_state_plan(self) -> List[str]:
        return [str(step or "").strip() for step in self.plan if str(step or "").strip()]

    @model_validator(mode="before")
    @classmethod
    def _coerce_plan_payload(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return {"plan": value.to_state_plan()}
        if isinstance(value, dict):
            raw_steps = value.get("plan", value.get("steps", []))
        elif isinstance(value, (list, tuple)):
            raw_steps = value
        elif isinstance(value, str):
            raw_steps = [value]
        else:
            return value

        if isinstance(raw_steps, str):
            raw_steps = [raw_steps]

        if not isinstance(raw_steps, (list, tuple)):
            raw_steps = []

        normalized_steps = [
            str(step or "").strip()
            for step in raw_steps
            if str(step or "").strip()
        ]
        return {"plan": normalized_steps}

    @model_serializer(mode="plain")
    def _serialize_to_state(self) -> Dict[str, List[str]]:
        return {"plan": self.to_state_plan()}


class PlannerNode:
    SIMPLE_REQUEST_CONNECTOR_RE = re.compile(
        r"(?:然后|接着|并且|以及|同时|顺便|后面|随后|再|\band\b|\bthen\b)",
        flags=re.IGNORECASE,
    )

    @staticmethod
    def _system_prompt() -> str:
        return load_prompt_template("agent_prompts/templates/planner_steps_system.txt")

    @staticmethod
    def _replan_system_prompt() -> str:
        return load_prompt_template("agent_prompts/templates/planner_replan_system.txt")

    @staticmethod
    def _normalize_usage(usage_payload: Any) -> Optional[Dict[str, int]]:
        if not isinstance(usage_payload, dict):
            return None

        def _to_int(value: Any) -> int:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return 0

        input_tokens = _to_int(
            usage_payload.get("input_tokens")
            or usage_payload.get("prompt_tokens")
        )
        output_tokens = _to_int(
            usage_payload.get("output_tokens")
            or usage_payload.get("completion_tokens")
        )
        total_tokens = _to_int(
            usage_payload.get("total_tokens")
            or usage_payload.get("total")
        )
        if total_tokens <= 0:
            total_tokens = input_tokens + output_tokens

        if input_tokens <= 0 and output_tokens <= 0 and total_tokens <= 0:
            return None

        return {
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
        }

    @staticmethod
    def _extract_usage_from_response_metadata(response_metadata: Any) -> Optional[Dict[str, int]]:
        if not isinstance(response_metadata, dict):
            return None

        usage_candidates = (
            response_metadata.get("token_usage"),
            response_metadata.get("usage"),
            response_metadata.get("usage_metadata"),
        )
        for candidate in usage_candidates:
            normalized = PlannerNode._normalize_usage(candidate)
            if normalized:
                return normalized
        return None

    @staticmethod
    async def _invoke_plan(
        *,
        prompt: ChatPromptTemplate,
        model: BaseChatModel,
        inputs: Dict[str, Any],
        config: RunnableConfig,
    ) -> tuple[Plan, Optional[Dict[str, int]]]:
        usage: Optional[Dict[str, int]] = None

        for kwargs in ({"include_raw": True}, {}):
            try:
                chain = prompt | model.with_structured_output(Plan, **kwargs)
                result = await chain.ainvoke(inputs, config=config)
            except TypeError:
                if kwargs:
                    continue
                raise

            parsed_result = result
            if kwargs.get("include_raw") and isinstance(result, dict):
                parsed_result = result.get("parsed")
                raw_message = result.get("raw")
                usage = PlannerNode._extract_usage_from_response_metadata(
                    getattr(raw_message, "response_metadata", {}) or {},
                )

            if isinstance(parsed_result, Plan):
                return parsed_result, usage

            return Plan.model_validate(parsed_result), usage

        raise RuntimeError("planner structured output returned no result")

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
    def _serialize_plan_payload(plan: Any) -> Any:
        if hasattr(plan, "model_dump"):
            try:
                return plan.model_dump()
            except Exception:
                return plan
        return plan

    @staticmethod
    def _plan_steps_from_payload(plan: Any) -> List[str]:
        serialized_plan = PlannerNode._serialize_plan_payload(plan)
        if isinstance(serialized_plan, dict):
            return list(
                serialized_plan.get("plan")
                or serialized_plan.get("steps")
                or []
            )
        if isinstance(serialized_plan, (list, tuple)):
            return list(serialized_plan)
        if isinstance(serialized_plan, str):
            return [serialized_plan]
        return []

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
    def _infer_step_agent(step_text: str) -> str:
        normalized_step = str(step_text or "").strip()
        if not normalized_step:
            return ""

        try:
            from supervisor.supervisor import (
                _looks_like_holter_request,
                _looks_like_search_request,
                _looks_like_sql_request,
                _looks_like_weather_request,
            )
        except Exception:
            return ""

        if _looks_like_holter_request(normalized_step):
            return "yunyou_agent"
        if _looks_like_sql_request(normalized_step):
            return "sql_agent"
        if _looks_like_weather_request(normalized_step):
            return "weather_agent"
        if _looks_like_search_request(normalized_step):
            return "search_agent"
        return ""

    @staticmethod
    def _build_task_record(task_id: str, step: str, *, agent: str = "") -> Dict[str, Any]:
        normalized_step = str(step or "").strip()
        resolved_agent = str(agent or "").strip() or PlannerNode._infer_step_agent(normalized_step)
        return {
            "id": str(task_id or "").strip(),
            "agent": resolved_agent,
            "input": normalized_step,
            "depends_on": [],
            "status": TaskStatus.PENDING.value,
            "result": None,
        }

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
            appended.append(PlannerNode._build_task_record(f"t{next_sequence}", normalized_step))
            known_inputs.add(normalized_step)
            next_sequence += 1
        return appended

    @staticmethod
    def _is_simple_request(user_text: str) -> bool:
        text = str(user_text or "").strip()
        if not text:
            return True
        if len(split_query_clauses(text)) > 1:
            return False
        return not PlannerNode.SIMPLE_REQUEST_CONNECTOR_RE.search(text)

    @staticmethod
    def _planner_llm_enabled() -> bool:
        return bool(getattr(ROUTER_POLICY_CONFIG, "planner_llm_fallback_enabled", False))

    @staticmethod
    async def planner_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
        """Planner 节点：使用 fast model 异步拆解原子步骤列表。"""
        user_text = PlannerNode._latest_user_text(state)
        started_at = time.perf_counter()
        replan_reason = str(state.get("replan_reason") or "").strip()
        route_strategy = str(state.get("route_strategy") or "").strip()
        existing_tasks = [dict(task) for task in list(state.get("task_list") or [])]
        planner_usage: Optional[Dict[str, int]] = None
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
            if PlannerNode._planner_llm_enabled():
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
                    plan_result, planner_usage = await PlannerNode._invoke_plan(
                        prompt=prompt,
                        model=model,
                        inputs={
                            "original_user_request": replan_payload["original_user_request"],
                            "completed_results": replan_payload["completed_results"],
                            "remaining_goals": replan_payload["remaining_goals"],
                            "replan_reason": replan_payload["replan_reason"],
                        },
                        config=config,
                    )
                    steps = PlannerNode._normalize_steps(
                        PlannerNode._plan_steps_from_payload(plan_result),
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
            else:
                planner_source = "fallback_replan_disabled_llm"
                steps = PlannerNode._normalize_steps(
                    list(replan_payload["remaining_goals"]),
                    replan_payload["original_user_request"],
                )
            task_list = preserved_tasks + PlannerNode._append_new_tasks(
                existing_tasks=preserved_tasks,
                steps=steps,
            )
        elif route_strategy == RouteStrategy.MULTI_DOMAIN_SPLIT.value:
            steps = PlannerNode._fallback_steps(user_text)
            planner_source = "rule_split"
            task_list = [
                PlannerNode._build_task_record(f"t{index + 1}", step)
                for index, step in enumerate(steps)
            ]
        elif (
            route_strategy != RouteStrategy.COMPLEX_SINGLE_DOMAIN.value
            and PlannerNode._is_simple_request(user_text)
        ):
            steps = PlannerNode._normalize_steps([user_text], user_text)
            planner_source = "passthrough_single"
            task_list = [
                PlannerNode._build_task_record(f"t{index + 1}", step)
                for index, step in enumerate(steps)
            ]
        else:
            if PlannerNode._planner_llm_enabled():
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", PlannerNode._system_prompt()),
                        ("user", "用户请求：\n{user_input}\n\n只返回 JSON。"),
                    ]
                )

                planner_source = "llm"
                try:
                    plan_result, planner_usage = await PlannerNode._invoke_plan(
                        prompt=prompt,
                        model=model,
                        inputs={
                            "user_input": user_text,
                        },
                        config=config,
                    )
                    steps = PlannerNode._normalize_steps(
                        PlannerNode._plan_steps_from_payload(plan_result),
                        user_text,
                    )
                    if not steps:
                        raise ValueError("planner returned empty steps")
                except Exception as exc:
                    planner_source = "fallback_split"
                    log.warning(f"Planner structured output failed, fallback to rule split: {exc}")
                    steps = PlannerNode._fallback_steps(user_text)
            else:
                planner_source = "rule_split_disabled_llm"
                steps = PlannerNode._fallback_steps(user_text)
            task_list = [
                PlannerNode._build_task_record(f"t{index + 1}", step)
                for index, step in enumerate(steps)
            ]

        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        memory = dict(state.get("memory") or {})
        memory["original_user_request"] = str(memory.get("original_user_request") or user_text).strip()
        memory["planned_steps"] = list(steps)
        memory["plan_source"] = planner_source
        memory["completed_step_results"] = PlannerNode._collect_completed_step_results(state)
        if planner_usage:
            memory["planner_usage"] = planner_usage
        memory["runtime_task_plan"] = runtime_task_planner.build_plan(
            steps=steps,
            task_list=task_list,
            source=planner_source,
            replan_reason=replan_reason,
            planner_usage=planner_usage or {},
        ).to_dict()

        log.info(f"⏱️ Planner [{planner_source}] 耗时: {elapsed_ms}ms, steps={steps}")
        return {
            "plan": list(steps),
            "task_list": task_list,
            "planner_source": planner_source,
            "planner_elapsed_ms": elapsed_ms,
            "planner_usage": planner_usage,
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
