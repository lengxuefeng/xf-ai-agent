# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.tools import tool

from models.schemas.tool_input_schemas import TavilySearchToolInput
from runtime.tools import ToolExecutionReport, runtime_tool_orchestrator
from tools.agent_tools.search_tools import tavily_search_tool
from tools.runtime_tools.tool_registry import runtime_tool_registry


@tool(
    "web_search_proxy",
    args_schema=TavilySearchToolInput,
    description="统一的外部检索能力入口。",
)
def web_search_proxy(query: str, topic: str = "general") -> List[Dict[str, str]]:
    return tavily_search_tool.invoke({"query": query, "topic": topic})


class SearchGateway:
    """
    统一搜索网关。

    统一从 Runtime Tool 层调用搜索工具，兼容前端能力观测。
    """

    @staticmethod
    def capability_snapshot() -> Dict[str, Any]:
        descriptor = runtime_tool_registry.get_tool("web_search_proxy")
        enabled = bool(
            descriptor is not None
            and runtime_tool_registry.is_tool_allowed_in_current_mode(descriptor)
        )
        return {
            "enabled": enabled,
            "provider": "tavily_proxy" if enabled else "not_configured",
            "tool_name": "web_search_proxy",
            "reason": (
                "统一搜索网关已接入 Tool Runtime。"
                if enabled
                else "当前运行模式未启用统一搜索工具。"
            ),
        }

    @staticmethod
    def search_once(
        *,
        query: str,
        topic: str = "general",
        run_context=None,
        source_agent: str = "search_agent",
        approval_granted: bool = False,
    ) -> ToolExecutionReport:
        return runtime_tool_orchestrator.execute_tool(
            "web_search_proxy",
            args={"query": query, "topic": topic},
            run_context=run_context,
            source_agent=source_agent,
            approval_granted=approval_granted,
            meta={"query": query, "topic": topic},
        )

    @classmethod
    def search(
        cls,
        *,
        query: str,
        topic: str = "general",
        run_context=None,
        source_agent: str = "search_agent",
        approval_granted: bool = False,
    ) -> List[Dict[str, str]]:
        report = cls.search_once(
            query=query,
            topic=topic,
            run_context=run_context,
            source_agent=source_agent,
            approval_granted=approval_granted,
        )
        if report.ok and isinstance(report.result, list):
            return report.result
        return [{
            "title": "搜索失败",
            "url": "",
            "content": report.error or "统一搜索网关未返回可用结果。",
        }]


search_gateway = SearchGateway()
runtime_tool_registry.register_native_tool(
    web_search_proxy,
    category="search",
    source="runtime.tools.search_gateway",
    description="统一的外部检索能力入口。",
)
