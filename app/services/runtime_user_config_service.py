# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional

from db import get_db_context
from services.user_skill_service import user_skill_service
from tools.runtime_tools.mcp_gateway import mcp_gateway
from tools.runtime_tools.tool_registry import runtime_tool_registry
from common.utils.custom_logger import get_logger

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
            explicit_selection = bool(requested_skill_ids)

            if explicit_selection:
                active_skills = self._load_selected_skills(db, user_id, requested_skill_ids)
            else:
                active_skills = user_skill_service.get_user_skills(db, user_id=user_id, active_only=True)

            skill_prompt_blocks = [str(item.system_prompt or "").strip() for item in active_skills if str(item.system_prompt or "").strip()]
            skill_names = [str(item.name or "").strip() for item in active_skills if str(item.name or "").strip()]
            normalized_bound_tools = runtime_tool_registry.normalize_bound_tools(
                tool_name
                for skill in active_skills
                for tool_name in (skill.bound_tools or [])
            )

            tool_restriction_enabled = explicit_selection or bool(active_skills)
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
                "selected_skill_ids": [int(item.id) for item in active_skills],
                "selected_skill_names": skill_names,
                "skill_prompt_blocks": skill_prompt_blocks,
                "tool_restriction_enabled": tool_restriction_enabled,
                "allowed_builtin_tools": normalized_bound_tools,
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

    @staticmethod
    def _load_selected_skills(db, user_id: int, skill_ids: List[int]):
        skills = []
        for skill_id in skill_ids:
            skill = user_skill_service.get_user_skill_by_id(db, skill_id, user_id)
            if skill and bool(skill.is_active):
                skills.append(skill)
        return skills


runtime_user_config_service = RuntimeUserConfigService()
