# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional

from common.utils.custom_logger import get_logger
from db import get_db_context
from services.runtime_skill_service import runtime_skill_service
from tools.runtime_tools.mcp_gateway import mcp_gateway
from tools.runtime_tools.tool_registry import runtime_tool_registry

log = get_logger(__name__)


class RuntimeUserConfigService:
    """按用户与 Skill 选择构造运行时配置。"""

    def build_runtime_profile(
        self,
        *,
        user_id: int,
        skill_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        with get_db_context() as db:
            requested_skill_ids = self._normalize_skill_ids(skill_ids)
            runtime_skills = runtime_skill_service.load_runtime_skills(
                db=db,
                user_id=user_id,
                skill_ids=requested_skill_ids,
            )
            active_skills = [skill for skill in runtime_skills if skill.is_active]

            skill_prompt_blocks = [
                str(item.system_prompt or "").strip()
                for item in active_skills
                if str(item.system_prompt or "").strip()
            ]
            skill_names = [str(item.name or "").strip() for item in active_skills if str(item.name or "").strip()]

            declared_bound_tools = [
                tool_name
                for skill in active_skills
                for tool_name in (skill.bound_tools or [])
            ]
            normalized_bound_tools = runtime_tool_registry.normalize_bound_tools(declared_bound_tools)
            filtered_bound_tools = runtime_tool_registry.filter_bound_tools(declared_bound_tools)
            tool_restriction_enabled = bool(normalized_bound_tools)

            mcp_servers = mcp_gateway.list_user_connectors(db, user_id, active_only=True)
            dynamic_tool_catalog = [
                {
                    "name": connector["name"],
                    "category": "mcp",
                    "description": f"MCP Server({connector['transport']})",
                    "source": "runtime.user_mcp",
                    "tool_type": "mcp",
                    "requires_approval": False,
                }
                for connector in mcp_servers
            ]

            profile = {
                "selected_skill_ids": [
                    int(item.source_id)
                    for item in active_skills
                    if str(item.source).strip() == "db" and str(item.source_id or "").isdigit()
                ],
                "selected_skill_names": skill_names,
                "skill_prompt_blocks": skill_prompt_blocks,
                "tool_restriction_enabled": tool_restriction_enabled,
                "allowed_builtin_tools": normalized_bound_tools,
                "resolved_bound_tools": filtered_bound_tools,
                "runtime_skills": [
                    {
                        "name": item.name,
                        "description": item.description,
                        "source": item.source,
                        "source_id": item.source_id,
                        "source_path": item.source_path,
                        "is_active": item.is_active,
                        "bound_tools": list(item.bound_tools),
                    }
                    for item in active_skills
                ],
                "mcp_servers": mcp_servers,
                "dynamic_tool_catalog": dynamic_tool_catalog,
            }
            log.info(
                f"运行时用户配置已装载: user_id={user_id}, "
                f"skills={profile['selected_skill_ids']}, "
                f"mcp_servers={len(mcp_servers)}, "
                f"tool_restriction={tool_restriction_enabled}"
            )
            return profile

    @staticmethod
    def _normalize_skill_ids(skill_ids: Optional[List[int]]) -> List[int]:
        normalized: list[int] = []
        for raw_id in skill_ids or []:
            try:
                skill_id = int(raw_id)
            except Exception:
                continue
            if skill_id > 0 and skill_id not in normalized:
                normalized.append(skill_id)
        return normalized


runtime_user_config_service = RuntimeUserConfigService()
