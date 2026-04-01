# -*- coding: utf-8 -*-
"""
用户自定义 Skill 配置模型。

Skill 负责持久化：
1. 面向当前用户的专属系统提示词片段
2. 当前 Skill 允许绑定的工具白名单
3. 启用状态，供运行期按需注入
"""
from __future__ import annotations

from datetime import datetime
from typing import List

from sqlalchemy import BigInteger, Boolean, DateTime, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from db import Base


class UserSkill(Base):
    """用户自定义 Skill 配置。"""

    __tablename__ = "t_user_skill"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, doc="主键")
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True, doc="用户 ID")

    name: Mapped[str] = mapped_column(String(100), nullable=False, doc="技能名称")
    description: Mapped[str] = mapped_column(String(500), nullable=False, doc="技能描述")
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False, doc="技能专属系统提示词")
    bound_tools: Mapped[List[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
        doc="绑定的工具名称列表",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
        doc="是否启用",
    )

    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), doc="创建时间")
    update_time: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        doc="更新时间",
    )

    def __repr__(self) -> str:
        return (
            f"<UserSkill(id={self.id}, user_id={self.user_id}, name='{self.name}', "
            f"is_active={self.is_active})>"
        )
