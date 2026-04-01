# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
from enum import Enum


class ToolType(str, Enum):
    """定义工具的来源与执行环境类型"""
    NATIVE = "native"  # 系统原生的物理沙箱执行工具（如 Python exec, SQL query）
    MCP = "mcp"        # 通过 Model Context Protocol 动态加载的网关工具
    SKILL = "skill"    # 声明式的业务技能编排


@dataclass(slots=True)
class ToolDescriptor:
    name: str
    category: str
    description: str
    source: str
    tool_type: ToolType = ToolType.NATIVE
    requires_approval: bool = False
    tool_ref: Optional[Any] = None
    # 针对 MCP 和 Skill 的额外配置上下文
    mcp_config: Optional[Dict[str, Any]] = None
    skill_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "source": self.source,
            "tool_type": self.tool_type.value,
            "requires_approval": self.requires_approval,
        }


class RuntimeToolRegistry:
    """
    统一工具执行注册表 (Runtime Execution Layer).
    
    【核心目的】彻底明确 Agent Brain (智能决策层) 与 Runtime (底层物理执⾏层) 的边界。
    集中管理 Native/MCP/Skills 三类工具的注册、路由和生命周期。
    """

    def __init__(self) -> None:
        self._builtin_aliases: Dict[str, set[str]] = {
            "web_search": {"web_search", "web_search_proxy", "search", "search_tool", "tavily_search_tool"},
            "sql_tools": {"sql_tools", "execute_sql", "execute_sql_tool", "get_schema", "get_schema_tool"},
            "weather_tools": {"weather_tools", "weather", "get_weathers"},
            "python_exec": {"python_exec", "execute_python_code"},
        }
        self._agent_builtin_requirements: Dict[str, set[str]] = {
            "search_agent": {"web_search"},
            "sql_agent": {"sql_tools"},
            "weather_agent": {"weather_tools"},
        }
        self._disabled_agent_messages: Dict[str, str] = {
            "search_agent": "当前会话未启用联网搜索工具，无法执行实时检索。请在 Agent Skills 中启用搜索能力后重试。",
            "sql_agent": "当前会话未启用 SQL 工具，无法执行数据库查询。请在 Agent Skills 中启用 SQL 能力后重试。",
            "weather_agent": "当前会话未启用天气工具，无法执行天气查询。请在 Agent Skills 中启用天气能力后重试。",
        }
        self._tools: Dict[str, ToolDescriptor] = {
            "execute_python_code": ToolDescriptor(
                name="execute_python_code",
                category="exec",
                description="执行 Python 代码并返回标准输出",
                source="runtime.exec",
                tool_type=ToolType.NATIVE,
                requires_approval=True,
            ),
            "execute_sql": ToolDescriptor(
                name="execute_sql",
                category="database",
                description="执行只读 SQL 查询并返回结构化结果",
                source="agent.tools.sql_tools",
                tool_type=ToolType.NATIVE,
            ),
            "federated_query_gateway": ToolDescriptor(
                name="federated_query_gateway",
                category="data_gateway",
                description="统一访问本地 SQL 与云柚数据域",
                source="agent.gateway.federated_query_gateway",
                tool_type=ToolType.NATIVE,
            ),
            "user_mcp": ToolDescriptor(
                name="user_mcp",
                category="mcp",
                description="用户 MCP 连接配置与访问入口",
                source="services.user_mcp_service",
                tool_type=ToolType.NATIVE,
            ),
            "web_search_proxy": ToolDescriptor(
                name="web_search_proxy",
                category="search",
                description="统一的外部检索能力入口",
                source="runtime.tools.search_gateway",
                tool_type=ToolType.NATIVE,
            ),
        }

    def register_native_tool(
        self,
        tool: Any,
        *,
        category: str,
        source: str,
        requires_approval: bool = False,
        description: str = "",
    ) -> None:
        self._register(
            tool=tool,
            tool_type=ToolType.NATIVE,
            category=category,
            source=source,
            requires_approval=requires_approval,
            description=description,
        )

    # 兼容老入口
    register_langchain_tool = register_native_tool

    def register_mcp_tool(
        self,
        tool: Any,
        *,
        category: str,
        source: str = "mcp_gateway",
        requires_approval: bool = True,
        mcp_config: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> None:
        self._register(
            tool=tool,
            tool_type=ToolType.MCP,
            category=category,
            source=source,
            requires_approval=requires_approval,
            description=description,
            mcp_config=mcp_config,
        )

    def register_skill_tool(
        self,
        tool: Any,
        *,
        category: str,
        source: str = "skills_engine",
        requires_approval: bool = False,
        skill_config: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> None:
        self._register(
            tool=tool,
            tool_type=ToolType.SKILL,
            category=category,
            source=source,
            requires_approval=requires_approval,
            description=description,
            skill_config=skill_config,
        )

    def _register(
        self,
        tool: Any,
        tool_type: ToolType,
        category: str,
        source: str,
        requires_approval: bool,
        description: str,
        mcp_config: Optional[Dict[str, Any]] = None,
        skill_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        tool_name = str(getattr(tool, "name", "") or "").strip()
        if not tool_name:
            return
        tool_description = str(description or getattr(tool, "description", "") or "").strip()
        self._tools[tool_name] = ToolDescriptor(
            name=tool_name,
            category=category,
            description=tool_description,
            source=source,
            tool_type=tool_type,
            requires_approval=requires_approval,
            tool_ref=tool,
            mcp_config=mcp_config,
            skill_config=skill_config,
        )

    def list_tools(self) -> List[dict]:
        return [tool.to_dict() for tool in self._tools.values()]

    def build_tool_catalog(self, dynamic_tools: Optional[List[dict]] = None) -> List[dict]:
        catalog = self.list_tools()
        if dynamic_tools:
            catalog.extend(dynamic_tools)
        return catalog

    def list_tools_by_type(self, tool_type: ToolType) -> List[dict]:
        return [tool.to_dict() for tool in self._tools.values() if tool.tool_type == tool_type]

    def get_tool(self, name: str) -> ToolDescriptor | None:
        return self._tools.get(str(name or "").strip())

    def get_langchain_tool(self, name: str) -> Any | None:
        descriptor = self.get_tool(name)
        return descriptor.tool_ref if descriptor else None

    def stats(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"total": len(self._tools), "by_type": {}, "by_category": {}}
        for tool in self._tools.values():
            cat = tool.category
            t_type = tool.tool_type.value
            result["by_category"][cat] = result["by_category"].get(cat, 0) + 1
            result["by_type"][t_type] = result["by_type"].get(t_type, 0) + 1
        return result

    def build_tool_stats(self, dynamic_tools: Optional[List[dict]] = None) -> Dict[str, Any]:
        result = self.stats()
        if not dynamic_tools:
            return result

        merged = {
            "total": int(result.get("total", 0)),
            "by_type": dict(result.get("by_type", {})),
            "by_category": dict(result.get("by_category", {})),
        }
        for tool in dynamic_tools:
            merged["total"] += 1
            tool_type = str(tool.get("tool_type") or ToolType.MCP.value)
            category = str(tool.get("category") or "mcp")
            merged["by_type"][tool_type] = merged["by_type"].get(tool_type, 0) + 1
            merged["by_category"][category] = merged["by_category"].get(category, 0) + 1
        return merged

    def normalize_bound_tools(self, tool_names: Iterable[str] | None) -> List[str]:
        normalized: list[str] = []
        for raw_name in tool_names or []:
            candidate = str(raw_name or "").strip().lower()
            if not candidate:
                continue
            canonical = self._canonical_builtin_name(candidate)
            final_name = canonical or candidate
            if final_name not in normalized:
                normalized.append(final_name)
        return normalized

    def is_tool_restriction_enabled(self, llm_config: Optional[Dict[str, Any]]) -> bool:
        return bool((llm_config or {}).get("tool_restriction_enabled"))

    def get_allowed_builtin_tools(self, llm_config: Optional[Dict[str, Any]]) -> set[str]:
        tool_names = (llm_config or {}).get("allowed_builtin_tools") or []
        return set(self.normalize_bound_tools(tool_names))

    def is_agent_enabled(self, agent_name: str, llm_config: Optional[Dict[str, Any]]) -> bool:
        if not self.is_tool_restriction_enabled(llm_config):
            return True
        required = self._agent_builtin_requirements.get(str(agent_name or "").strip())
        if not required:
            return True
        allowed = self.get_allowed_builtin_tools(llm_config)
        return bool(required & allowed)

    def get_disabled_agent_message(self, agent_name: str) -> str:
        return self._disabled_agent_messages.get(
            str(agent_name or "").strip(),
            "当前会话未启用所需工具能力，请调整 Skill 配置后重试。",
        )

    def _canonical_builtin_name(self, tool_name: str) -> Optional[str]:
        candidate = str(tool_name or "").strip().lower()
        if not candidate:
            return None
        for canonical, aliases in self._builtin_aliases.items():
            if candidate == canonical or candidate in aliases:
                return canonical
        return None


runtime_tool_registry = RuntimeToolRegistry()
