# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import yaml
from sqlalchemy.orm import Session

from common.utils.custom_logger import get_logger
from config.runtime_settings import is_local_mode
from db.crud import user_skill_db
from models.schemas.user_skill_schemas import UserSkillCreate, UserSkillUpdate
from tools.runtime_tools.tool_registry import runtime_tool_registry

log = get_logger(__name__)

_FRONTMATTER_PATTERN = re.compile(
    r"^\ufeff?---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n)?(.*)$",
    re.DOTALL,
)


@dataclass(slots=True)
class RuntimeSkill:
    name: str
    description: str
    system_prompt: str
    bound_tools: List[str]
    is_active: bool
    source: str
    source_id: str = ""
    source_path: str = ""


class UserSkillService:
    """用户自定义 Skill 服务层。"""

    def __init__(self, skills_root: Path | None = None) -> None:
        self._skills_root = Path(skills_root or Path(__file__).resolve().parents[2] / "skills")

    def get_user_skills(self, db: Session, user_id: int, active_only: bool = False):
        if active_only:
            return user_skill_db.get_active_by_user_id(db, user_id=user_id)
        return user_skill_db.get_by_user_id(db, user_id=user_id)

    def get_user_skill_by_id(self, db: Session, skill_id: int, user_id: Optional[int] = None):
        user_skill = user_skill_db.get(db, skill_id)
        if not user_skill:
            return None
        if user_id is not None and user_skill.user_id != user_id:
            log.warning(f"越权访问拦截：用户 {user_id} 试图访问技能配置 {skill_id}")
            return None
        return user_skill

    def create_user_skill(self, db: Session, user_skill: UserSkillCreate, user_id: int):
        create_data = user_skill.model_dump()
        create_data["user_id"] = user_id
        created_skill = user_skill_db.create(db, obj_in=create_data)
        log.info(f"用户 {user_id} 创建技能配置成功: skill_id={created_skill.id}")
        return created_skill

    def update_user_skill(self, db: Session, skill_id: int, user_skill: UserSkillUpdate, user_id: int):
        existing_skill = self.get_user_skill_by_id(db, skill_id, user_id)
        if not existing_skill:
            raise ValueError(f"技能配置不存在或无权操作: ID={skill_id}")

        update_data = user_skill.model_dump(exclude_unset=True)
        updated_skill = user_skill_db.update(db, db_obj=existing_skill, obj_in=update_data)
        log.info(f"用户 {user_id} 更新技能配置成功: skill_id={skill_id}")
        return updated_skill

    def remove_user_skill(self, db: Session, skill_id: int, user_id: int):
        existing_skill = self.get_user_skill_by_id(db, skill_id, user_id)
        if not existing_skill:
            raise ValueError(f"技能配置不存在或无权操作: ID={skill_id}")
        removed_skill = user_skill_db.remove(db, id=skill_id)
        log.info(f"用户 {user_id} 删除技能配置成功: skill_id={skill_id}")
        return removed_skill

    def load_runtime_skills(
        self,
        *,
        db: Session | None,
        user_id: int,
        skill_ids: Optional[List[int]] = None,
    ) -> List[RuntimeSkill]:
        if is_local_mode():
            local_skills = self.load_local_markdown_skills()
            if local_skills:
                log.info(f"local 模式已装载本地 Markdown Skills: count={len(local_skills)}")
                return local_skills
            log.info("local 模式未发现有效 Markdown Skills，已回退到 DB Skill 配置。")

        return self._load_db_runtime_skills(
            db=db,
            user_id=user_id,
            skill_ids=skill_ids,
        )

    def load_local_markdown_skills(self, skills_root: Path | None = None) -> List[RuntimeSkill]:
        root = Path(skills_root or self._skills_root)
        if not root.exists():
            return []

        skills: list[RuntimeSkill] = []
        for file_path in sorted(root.rglob("*.md")):
            parsed = self._parse_local_markdown_skill(file_path=file_path, skills_root=root)
            if parsed is not None:
                skills.append(parsed)
        return skills

    def _load_db_runtime_skills(
        self,
        *,
        db: Session | None,
        user_id: int,
        skill_ids: Optional[List[int]] = None,
    ) -> List[RuntimeSkill]:
        if db is None:
            log.warning("运行时 DB Skill 装载失败：缺少数据库会话，已降级为空 Skill 列表。")
            return []

        requested_skill_ids = self._normalize_skill_ids(skill_ids)
        if requested_skill_ids:
            source_skills = []
            for skill_id in requested_skill_ids:
                skill = self.get_user_skill_by_id(db, skill_id, user_id)
                if skill:
                    source_skills.append(skill)
        else:
            source_skills = self.get_user_skills(db, user_id=user_id, active_only=False)

        runtime_skills: list[RuntimeSkill] = []
        for skill in source_skills or []:
            name = str(getattr(skill, "name", "") or "").strip()
            description = str(getattr(skill, "description", "") or "").strip()
            if not name or not description:
                continue
            runtime_skills.append(
                RuntimeSkill(
                    name=name,
                    description=description,
                    system_prompt=str(getattr(skill, "system_prompt", "") or "").strip(),
                    bound_tools=self._normalize_bound_tools(getattr(skill, "bound_tools", []) or []),
                    is_active=bool(getattr(skill, "is_active", True)),
                    source="db",
                    source_id=str(getattr(skill, "id", "") or "").strip(),
                )
            )
        return runtime_skills

    def _parse_local_markdown_skill(self, *, file_path: Path, skills_root: Path) -> RuntimeSkill | None:
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            log.warning(f"本地 Skill 读取失败，已跳过: path={file_path}, error={exc}")
            return None

        matched = _FRONTMATTER_PATTERN.match(content)
        if not matched:
            log.warning(f"本地 Skill 缺少合法 frontmatter，已跳过: {file_path}")
            return None

        frontmatter_text = matched.group(1)
        markdown_body = matched.group(2)

        try:
            metadata = yaml.safe_load(frontmatter_text) or {}
        except Exception as exc:
            log.warning(f"本地 Skill frontmatter 解析失败，已跳过: path={file_path}, error={exc}")
            return None

        if not isinstance(metadata, dict):
            log.warning(f"本地 Skill frontmatter 非对象结构，已跳过: {file_path}")
            return None

        name = str(metadata.get("name") or "").strip()
        description = str(metadata.get("description") or "").strip()
        if not name or not description:
            log.warning(f"本地 Skill 缺少 name/description，已跳过: {file_path}")
            return None

        relative_path = str(file_path.relative_to(skills_root.parent))
        return RuntimeSkill(
            name=name,
            description=description,
            system_prompt=str(markdown_body or "").strip(),
            bound_tools=self._normalize_bound_tools(metadata.get("tools")),
            is_active=bool(metadata.get("is_active", True)),
            source="markdown",
            source_id=relative_path,
            source_path=str(file_path),
        )

    @staticmethod
    def _normalize_bound_tools(raw_tools) -> List[str]:
        if raw_tools in (None, ""):
            return []

        items: list[str] = []
        if isinstance(raw_tools, str):
            items = [part.strip() for part in re.split(r"[,\n]", raw_tools) if part.strip()]
        elif isinstance(raw_tools, Iterable) and not isinstance(raw_tools, (dict, bytes, bytearray)):
            for item in raw_tools:
                normalized = str(item or "").strip()
                if normalized:
                    items.append(normalized)
        else:
            normalized = str(raw_tools or "").strip()
            if normalized:
                items = [normalized]

        return runtime_tool_registry.normalize_bound_tools(items)

    @staticmethod
    def _normalize_skill_ids(skill_ids: Optional[Iterable[int]]) -> List[int]:
        normalized: list[int] = []
        for raw_id in skill_ids or []:
            try:
                skill_id = int(raw_id)
            except Exception:
                continue
            if skill_id > 0 and skill_id not in normalized:
                normalized.append(skill_id)
        return normalized


user_skill_service = UserSkillService()


__all__ = ["RuntimeSkill", "UserSkillService", "user_skill_service"]
