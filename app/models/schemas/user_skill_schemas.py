# -*- coding: utf-8 -*-
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from models.schemas.base import BaseSchema


class UserSkillBase(BaseModel):
    """用户 Skill 基础模型。"""

    name: str
    description: str
    system_prompt: str
    bound_tools: list[str] = Field(default_factory=list)
    is_active: bool = True

    @field_validator("name", "description", "system_prompt", mode="before")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("字段不能为空")
        return normalized

    @field_validator("bound_tools", mode="before")
    @classmethod
    def normalize_tools(cls, value):
        if value in (None, ""):
            return []
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            result: list[str] = []
            for item in value:
                normalized = str(item).strip()
                if normalized and normalized not in result:
                    result.append(normalized)
            return result
        return value


class UserSkillCreate(UserSkillBase):
    """创建用户 Skill。"""


class UserSkillUpdate(BaseModel):
    """更新用户 Skill。"""

    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    bound_tools: Optional[list[str]] = None
    is_active: Optional[bool] = None

    @field_validator("name", "description", "system_prompt", mode="before")
    @classmethod
    def normalize_optional_text(cls, value):
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("bound_tools", mode="before")
    @classmethod
    def normalize_tools(cls, value):
        if value in (None, ""):
            return None
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            result: list[str] = []
            for item in value:
                normalized = str(item).strip()
                if normalized and normalized not in result:
                    result.append(normalized)
            return result
        return value


class UserSkill(BaseSchema):
    """数据库模型对应的 Pydantic 模型。"""

    id: int
    user_id: int
    name: str
    description: str
    system_prompt: str
    bound_tools: list[str]
    is_active: bool
    create_time: datetime
    update_time: datetime


class UserSkillOut(UserSkill):
    """用户 Skill 输出模型。"""
