import time
from typing import List
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from supervisor.state import GraphState
from supervisor.supervisor_rule_support import split_query_clauses
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

    SYSTEM_PROMPT = (
        "你是一个任务拆解专家，请将用户的复杂请求拆分为多个原子的、可独立执行的子任务步骤。\n"
        "要求：\n"
        "1. 只输出 JSON 对象，格式必须是 {\"steps\": [\"步骤1\", \"步骤2\"]}。\n"
        "2. 每个步骤都必须是单一动作，不能把多个动作塞进同一步。\n"
        "3. 必须保留原请求中的顺序关系、地点实体、对象实体和限定条件。\n"
        "4. 如果请求里包含“然后、接着、再、并且、and、then”等连接词，必须拆成多步。\n"
        "5. 如果同一类任务涉及多个地点或多个目标对象，也要拆成多步。\n"
        "6. 不要解释，不要补充说明，不要输出 Markdown。"
    )

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

        if PlannerNode._is_simple_request(user_text):
            steps = PlannerNode._normalize_steps([user_text], user_text)
            planner_source = "passthrough_single"
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", PlannerNode.SYSTEM_PROMPT),
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

        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        memory = dict(state.get("memory") or {})
        memory["original_user_request"] = user_text
        memory["planned_steps"] = list(steps)
        memory["plan_source"] = planner_source

        log.info(f"⏱️ Planner [{planner_source}] 耗时: {elapsed_ms}ms, steps={steps}")
        task_list = [
            {
                "id": f"t{index + 1}",
                "agent": "",
                "input": step,
                "depends_on": [],
                "status": "pending",
                "result": None,
            }
            for index, step in enumerate(steps)
        ]
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
            "next": "dispatch_node",
        }

    @staticmethod
    async def parent_planner_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
        """兼容旧引用，统一转到新的 planner_node。"""
        return await PlannerNode.planner_node(state, model=model, config=config)
