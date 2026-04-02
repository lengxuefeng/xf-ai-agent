# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Iterable

from config.runtime_settings import get_run_mode


class RuntimeModeProfileResolver:
    """把 RUN_MODE 投影成显式运行时能力边界。"""

    def resolve(self, *, tool_catalog: Iterable[dict] | None = None) -> Dict[str, Any]:
        catalog = [dict(item) for item in (tool_catalog or []) if isinstance(item, dict)]
        categories = {str(item.get("category") or "").strip() for item in catalog}
        tool_types = {str(item.get("tool_type") or "").strip() for item in catalog}
        run_mode = get_run_mode().value
        supports_workspace = "workspace" in categories
        supports_shell = "exec" in categories
        supports_search = "search" in categories
        supports_mcp = "mcp" in categories or "mcp" in tool_types
        supports_skills = "skill" in tool_types
        return {
            "run_mode": run_mode,
            "transport": "hybrid" if run_mode == "local" else "remote",
            "supports_workspace": supports_workspace,
            "supports_shell": supports_shell,
            "supports_search": supports_search,
            "supports_mcp": supports_mcp,
            "supports_skills": supports_skills,
            "tool_count": len(catalog),
        }


runtime_mode_profile_resolver = RuntimeModeProfileResolver()

