# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig

from models.schemas.agent_runtime_schemas import AgentRequest
from prompts.agent_prompts.code_prompt import CodePrompt
from runtime.workers.base import RuntimeWorker
from supervisor.agents.code_agent import (
    _detect_requested_language,
    _format_generated_code_reply,
    _latest_human_text,
    _normalize_model_content,
    _should_execute_request,
    _strip_markdown_fences,
)


class CodeWorker(RuntimeWorker):
    """代码生成 Worker。

    Python 执行链路仍交给 legacy CodeAgent，以复用现有 interrupt/resume 语义。
    """

    worker_name = "code_worker"
    agent_name = "code_agent"

    @staticmethod
    def _latest_request_text(req: AgentRequest) -> str:
        state_messages = list((req.state or {}).get("messages") or [])
        current_task = str((req.state or {}).get("current_task") or "").strip()
        if current_task and current_task != "END_TASK":
            return current_task
        latest = _latest_human_text(state_messages)
        return latest or str(req.user_input or "").strip()

    def supports(self, req: AgentRequest) -> bool:
        latest_text = self._latest_request_text(req)
        requested_language = _detect_requested_language(latest_text)
        execution_requested = _should_execute_request(latest_text)
        return not (execution_requested and requested_language in {"python"})

    @staticmethod
    def _normalize_messages(req: AgentRequest) -> List[BaseMessage]:
        messages = list((req.state or {}).get("messages") or [])
        if messages:
            return messages
        user_input = str(req.user_input or "").strip()
        return [HumanMessage(content=user_input)] if user_input else []

    @staticmethod
    def _build_streaming_preface(*, language: str, execution_requested: bool, execution_supported: bool) -> str:
        display_name = language.capitalize() if language else "代码"
        if execution_requested and not execution_supported:
            return (
                f"按你的要求，我先给你整理一个 {display_name} 示例。"
                f"当前内置自动执行链路主要支持 Python，所以这段 {display_name} 代码先不直接运行。\n\n"
            )
        return f"按你的要求，我先给你整理一个 {display_name} 示例：\n\n"

    async def run(self, req: AgentRequest, *, config: RunnableConfig) -> Dict[str, object]:
        llm = req.model
        if llm is None:
            raise ValueError("CodeWorker 缺少可用模型。")

        latest_text = self._latest_request_text(req)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CodePrompt.SYSTEM),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        requested_language = _detect_requested_language(latest_text)
        execution_requested = _should_execute_request(latest_text)
        execution_supported = requested_language in {"python"}
        prompt_value = prompt.invoke({"messages": self._normalize_messages(req)})

        streamed_code_parts: list[str] = []
        streamed_response_metadata: dict = {}
        did_stream = False
        try:
            preface = self._build_streaming_preface(
                language=requested_language,
                execution_requested=execution_requested,
                execution_supported=execution_supported,
            )
            await self._publish_stream_text(req=req, config=config, text=preface, chunk_size=20)
            async for chunk in llm.astream(prompt_value.messages, config=config):
                chunk_text = _normalize_model_content(getattr(chunk, "content", chunk))
                if not chunk_text:
                    continue
                did_stream = True
                streamed_code_parts.append(chunk_text)
                await self._publish_stream_text(req=req, config=config, text=chunk_text, chunk_size=20)
                streamed_response_metadata.update(dict(getattr(chunk, "response_metadata", {}) or {}))
        except Exception:
            did_stream = False
            streamed_code_parts = []
            streamed_response_metadata = {}

        if did_stream:
            code = _strip_markdown_fences("".join(streamed_code_parts))
            response_metadata = streamed_response_metadata
        else:
            response = await (prompt | llm).ainvoke({"messages": self._normalize_messages(req)}, config=config)
            code = _strip_markdown_fences(_normalize_model_content(getattr(response, "content", response)))
            response_metadata = dict(getattr(response, "response_metadata", {}) or {})

        content = _format_generated_code_reply(
            code,
            language=requested_language,
            execution_requested=execution_requested,
            execution_supported=execution_supported,
        )
        return {
            "content": content,
            "response_metadata": {
                **response_metadata,
                "runtime_worker": self.worker_name,
                "requested_language": requested_language,
                "execution_requested": execution_requested,
                "legacy_execution_required": False,
                "live_streamed": did_stream,
            },
        }


code_worker = CodeWorker()
