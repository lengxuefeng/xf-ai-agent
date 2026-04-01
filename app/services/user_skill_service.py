# -*- coding: utf-8 -*-
from typing import Optional

from sqlalchemy.orm import Session

from db.crud import user_skill_db
from models.schemas.user_skill_schemas import UserSkillCreate, UserSkillUpdate
from common.utils.custom_logger import get_logger

log = get_logger(__name__)


class UserSkillService:
    """用户自定义 Skill 服务层。"""

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


user_skill_service = UserSkillService()
