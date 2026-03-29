# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(slots=True)
class ToolDescriptor:
    name: str
    category: str
    description: str
    source: str
    requires_approval: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class RuntimeToolRegistry:
    """统一工具注册表。"""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDescriptor] = {
            "execute_python_code": ToolDescriptor(
                name="execute_python_code",
                category="exec",
                description="执行 Python 代码并返回标准输出",
                source="runtime.exec",
                requires_approval=True,
            ),
            "execute_sql": ToolDescriptor(
                name="execute_sql",
                category="database",
                description="执行只读 SQL 查询并返回结构化结果",
                source="agent.tools.sql_tools",
            ),
            "federated_query_gateway": ToolDescriptor(
                name="federated_query_gateway",
                category="data_gateway",
                description="统一访问本地 SQL 与云柚数据域",
                source="agent.gateway.federated_query_gateway",
            ),
            "user_mcp": ToolDescriptor(
                name="user_mcp",
                category="mcp",
                description="用户 MCP 连接配置与访问入口",
                source="services.user_mcp_service",
            ),
            "web_search_proxy": ToolDescriptor(
                name="web_search_proxy",
                category="search",
                description="统一的外部检索能力入口",
                source="runtime.tools.search_gateway",
            ),
        }

    def list_tools(self) -> List[dict]:
        return [tool.to_dict() for tool in self._tools.values()]

    def get_tool(self, name: str) -> ToolDescriptor | None:
        return self._tools.get(str(name or "").strip())

    def stats(self) -> Dict[str, int]:
        categories: Dict[str, int] = {}
        for tool in self._tools.values():
            categories[tool.category] = categories.get(tool.category, 0) + 1
        return {"total": len(self._tools), **categories}


runtime_tool_registry = RuntimeToolRegistry()

