# -*- coding: utf-8 -*-
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import String, DateTime, func, BigInteger, Boolean, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from db import Base

"""
会话状态持久化表（Session State）。

设计目标：
1. 把“城市/用户画像槽位”等结构化上下文落在数据库，避免仅靠历史文本猜测。
2. 让路由层、Agent 层都能复用统一状态，减少重复追问和误路由。
"""


class SessionStateModel(Base):
    """会话状态表（每个 session_id 一条）"""
    __tablename__ = "t_session_state"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)

    # 槽位字典：如 city/name/age/gender/height_cm/weight_kg 等
    slots: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    # 人类可读摘要，便于快速注入系统上下文
    summary_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # 保存最近一次路由快照，用于排障和策略分析
    last_route: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    # 累积轮次，便于后续策略（如摘要刷新阈值）扩展
    turn_count: Mapped[int] = mapped_column(BigInteger, default=0)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    update_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

