# -*- coding: utf-8 -*-
from sqlalchemy import (
    Column,
    String,
    DateTime,
    func,
    BigInteger,
    Text,
    Boolean,
    UniqueConstraint,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB

from db import Base


class InterruptApprovalModel(Base):
    """
    人工审批持久化表。

    说明：
    1. 用 PostgreSQL 持久化审批状态，替代进程内存，避免重启丢失。
    2. `is_consumed` 用于标记该审批结果是否已被 [RESUME] 消费，避免重复恢复。
    """

    __tablename__ = "t_interrupt_approval"
    __table_args__ = (
        UniqueConstraint("session_id", "message_id", name="uq_interrupt_session_message"),
        Index("idx_interrupt_session_status_consumed", "session_id", "status", "is_consumed"),
        Index("idx_interrupt_create_time", "create_time"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True, doc="主键")
    session_id = Column(String(120), nullable=False, index=True, doc="会话ID")
    message_id = Column(String(200), nullable=False, index=True, doc="审批消息ID")

    action_name = Column(String(120), nullable=False, doc="工具名称")
    action_args = Column(JSONB, nullable=True, doc="工具参数")
    description = Column(Text, nullable=True, doc="审批描述")

    status = Column(String(20), nullable=False, default="pending", index=True, doc="pending/approve/reject")
    user_id = Column(BigInteger, nullable=True, index=True, doc="审批用户ID")
    decision_time = Column(DateTime, nullable=True, doc="审批时间")

    agent_name = Column(String(120), nullable=True, doc="触发审批的Agent名称")
    subgraph_thread_id = Column(String(180), nullable=True, doc="子图线程ID")
    checkpoint_id = Column(String(200), nullable=True, doc="checkpoint_id")
    checkpoint_ns = Column(String(200), nullable=True, doc="checkpoint_ns")

    is_consumed = Column(Boolean, nullable=False, default=False, index=True, doc="是否已被恢复流程消费")

    create_time = Column(DateTime, server_default=func.now(), doc="创建时间")
    update_time = Column(DateTime, server_default=func.now(), onupdate=func.now(), doc="更新时间")

