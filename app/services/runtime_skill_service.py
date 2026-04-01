# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Optional

from services.user_skill_service import RuntimeSkill, user_skill_service


class RuntimeSkillService:
    """兼容旧调用方的运行时 Skill 装配入口。"""

    def load_runtime_skills(
        self,
        *,
        db,
        user_id: int,
        skill_ids: Optional[List[int]] = None,
    ) -> List[RuntimeSkill]:
        return user_skill_service.load_runtime_skills(
            db=db,
            user_id=user_id,
            skill_ids=skill_ids,
        )


runtime_skill_service = RuntimeSkillService()


__all__ = ["RuntimeSkill", "RuntimeSkillService", "runtime_skill_service"]
