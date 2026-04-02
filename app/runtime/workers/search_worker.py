# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig

from harness.core.run_context import build_run_context
from models.schemas.agent_runtime_schemas import AgentRequest
from prompts.agent_prompts.tools_prompt import ToolsPrompt
from runtime.workers.base import RuntimeWorker
from supervisor.base import BaseAgent
from tools.runtime_tools.search_gateway import search_gateway


class SearchWorker(RuntimeWorker):
    """单轮查询式搜索 Worker。"""

    worker_name = "search_worker"
    agent_name = "search_agent"

    def supports(self, req: AgentRequest) -> bool:
        return True

    @staticmethod
    def _normalize_messages(req: AgentRequest) -> List[BaseMessage]:
        messages = list((req.state or {}).get("messages") or [])
        if messages:
            return messages
        user_input = str(req.user_input or "").strip()
        return [HumanMessage(content=user_input)] if user_input else []

    @staticmethod
    def _render_search_results(results: Any) -> str:
        if not isinstance(results, list):
            return str(results or "")
        lines: list[str] = []
        for index, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                lines.append(f"[{index}] {item}")
                continue
            title = str(item.get("title") or "搜索结果").strip()
            url = str(item.get("url") or "").strip()
            content = str(item.get("content") or "").strip()
            lines.append(
                f"[{index}] 标题: {title}\n"
                f"链接: {url or 'N/A'}\n"
                f"摘要: {content}"
            )
        return "\n\n".join(lines).strip()

    async def run(self, req: AgentRequest, *, config: RunnableConfig) -> Dict[str, Any]:
        llm = req.model
        if llm is None:
            raise ValueError("SearchWorker 缺少可用模型。")

        run_context = build_run_context(
            session_id=str(req.session_id or ""),
            user_input=str(req.user_input or ""),
            model_config=dict(req.llm_config or {}),
            history_messages=[],
            session_context={"context_summary": str((req.state or {}).get("context_summary") or "")},
            request_id=str(config.get("configurable", {}).get("request_id") or ""),
        )
        search_report = search_gateway.search_once(
            query=str(req.user_input or ""),
            topic="general",
            run_context=run_context,
            source_agent=self.agent_name,
        )
        if not search_report.ok:
            return {
                "content": search_report.error or "联网检索暂时失败，请稍后重试。",
                "response_metadata": {
                    "runtime_worker": self.worker_name,
                    "tool_report": search_report.to_dict(),
                },
            }

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ToolsPrompt.SEARCH_SYSTEM),
                (
                    "system",
                    "你已经拿到统一搜索网关返回的单轮结果。"
                    "现在只能基于这些结果总结，不得继续调用任何工具或发起第二轮搜索。",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        messages = self._normalize_messages(req)
        messages.append(
            SystemMessage(
                content=(
                    "统一搜索结果如下，请综合判断后直接回答用户：\n"
                    f"{self._render_search_results(search_report.result)}"
                )
            )
        )
        response = await (prompt | llm).ainvoke({"messages": messages}, config=config)
        normalized_content = BaseAgent._message_text(response).strip()
        if not normalized_content:
            normalized_content = self._render_search_results(search_report.result)
        return {
            "content": normalized_content,
            "response_metadata": {
                **dict(getattr(response, "response_metadata", {}) or {}),
                "runtime_worker": self.worker_name,
                "tool_report": search_report.to_dict(),
                "search_results_preview": self._render_search_results(search_report.result)[:500],
            },
        }


search_worker = SearchWorker()

