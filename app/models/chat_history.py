# -*- coding: utf-8 -*-
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import String, DateTime, func, BigInteger, Boolean, Text
from sqlalchemy.dialects.postgresql import JSONB  # 【核心】导入 PgSQL 的 JSONB
from sqlalchemy.orm import Mapped, mapped_column

from db import Base

# from db import Base

"""
【学习笔记】PostgreSQL 聊天历史数据模型
通过关系型表结构保证 user_id 和 session_id 的强一致性；
通过 JSONB 字段 (extra_data) 保证极高的动态扩展性，完美替代 MongoDB。
"""


class ChatSessionModel(Base):
    """聊天会话表"""
    __tablename__ = 't_chat_session'

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="新对话")
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    update_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class ChatMessageModel(Base):
    """聊天消息表"""
    __tablename__ = 't_chat_message'

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    # 存超长对话内容，Text 在 PgSQL 中性能极佳
    user_content: Mapped[str] = mapped_column(Text, nullable=False)
    model_content: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=True)

    tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    latency_ms: Mapped[int] = mapped_column(BigInteger, default=0)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)

    # 【高阶设计】可以把大模型返回的 tool_calls、复杂的引用来源存入 JSONB 字段
    # PgSQL 支持直接用 SQL 语句查询 JSONB 内部的字段！
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())