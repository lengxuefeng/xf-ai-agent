# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
from enum import Enum

from config.runtime_settings import (
    RunMode,
    build_run_mode_denied_message,
    get_run_mode,
    is_local_mode,
)


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
    allowed_modes: tuple[str, ...] = (RunMode.CLOUD.value, RunMode.LOCAL.value)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "source": self.source,
            "tool_type": self.tool_type.value,
            "requires_approval": self.requires_approval,
            "allowed_modes": list(self.allowed_modes),
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
        local_only: bool = False,
    ) -> None:
        self._register(
            tool=tool,
            tool_type=ToolType.NATIVE,
            category=category,
            source=source,
            requires_approval=requires_approval,
            description=description,
            local_only=local_only,
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
        local_only: bool = False,
    ) -> None:
        self._register(
            tool=tool,
            tool_type=ToolType.MCP,
            category=category,
            source=source,
            requires_approval=requires_approval,
            description=description,
            mcp_config=mcp_config,
            local_only=local_only,
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
        local_only: bool = False,
    ) -> None:
        self._register(
            tool=tool,
            tool_type=ToolType.SKILL,
            category=category,
            source=source,
            requires_approval=requires_approval,
            description=description,
            skill_config=skill_config,
            local_only=local_only,
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
        local_only: bool = False,
    ) -> None:
        tool_name = str(getattr(tool, "name", "") or "").strip()
        if not tool_name:
            return
        if local_only and not is_local_mode():
            raise ValueError(build_run_mode_denied_message(f"仅限 local 模式注册的工具 `{tool_name}`"))
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
            allowed_modes=(RunMode.LOCAL.value,) if local_only else (RunMode.CLOUD.value, RunMode.LOCAL.value),
        )

    def list_tools(self) -> List[dict]:
        return [tool.to_dict() for tool in self._iter_visible_tools()]

    def build_tool_catalog(self, dynamic_tools: Optional[List[dict]] = None) -> List[dict]:
        catalog = self.list_tools()
        if dynamic_tools:
            catalog.extend(dynamic_tools)
        return self.sort_tool_catalog(catalog)

    def list_tools_by_type(self, tool_type: ToolType) -> List[dict]:
        return [tool.to_dict() for tool in self._iter_visible_tools() if tool.tool_type == tool_type]

    def get_tool(self, name: str) -> ToolDescriptor | None:
        return self._tools.get(str(name or "").strip())

    def get_langchain_tool(self, name: str) -> Any | None:
        descriptor = self.get_tool(name)
        return descriptor.tool_ref if descriptor else None

    def is_tool_allowed_in_current_mode(self, tool: str | ToolDescriptor | None) -> bool:
        if tool is None:
            return False
        descriptor = tool if isinstance(tool, ToolDescriptor) else self.get_tool(str(tool or ""))
        if descriptor is None:
            return False
        return get_run_mode().value in descriptor.allowed_modes

    def stats(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"total": 0, "by_type": {}, "by_category": {}}
        for tool in self._iter_visible_tools():
            result["total"] += 1
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

    def filter_bound_tools(self, tool_names: Iterable[str] | None) -> List[str]:
        requested = set(self.normalize_bound_tools(tool_names))
        if not requested:
            return []
        filtered: list[str] = []
        for item in self.sort_tool_catalog(self.list_tools()):
            visible_name = str(item.get("name") or "").strip()
            if not visible_name:
                continue
            canonical_visible = self._canonical_builtin_name(visible_name) or visible_name
            if visible_name in requested or canonical_visible in requested:
                filtered.append(visible_name)
        return filtered

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

    @staticmethod
    def sort_tool_catalog(catalog: Iterable[dict] | None) -> List[dict]:
        tools = [dict(item) for item in (catalog or []) if isinstance(item, dict)]
        return sorted(tools, key=lambda item: str(item.get("name") or "").lower())

    @staticmethod
    def sort_langchain_tools(tools: Iterable[Any] | None) -> List[Any]:
        return sorted(
            [tool for tool in (tools or []) if tool is not None],
            key=lambda item: str(getattr(item, "name", "") or "").lower(),
        )

    def bind_tools(self, llm: Any, tools: Iterable[Any] | None, **kwargs: Any) -> Any:
        return llm.bind_tools(self.sort_langchain_tools(tools), **kwargs)

    def _canonical_builtin_name(self, tool_name: str) -> Optional[str]:
        candidate = str(tool_name or "").strip().lower()
        if not candidate:
            return None
        for canonical, aliases in self._builtin_aliases.items():
            if candidate == canonical or candidate in aliases:
                return canonical
        return None

    def _iter_visible_tools(self) -> List[ToolDescriptor]:
        visible = [
            descriptor
            for descriptor in self._tools.values()
            if self.is_tool_allowed_in_current_mode(descriptor)
        ]
        return self.sort_langchain_tools(visible)


runtime_tool_registry = RuntimeToolRegistry()

if is_local_mode():
    import tools.runtime_tools.local_workspace_tools as _local_workspace_tools  # noqa: F401
