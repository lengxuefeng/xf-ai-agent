# -*- coding: utf-8 -*-
"""
人工审批持久化模型（Interrupt Approval Model）。

这个模型用于持久化存储Agent执行过程中的人工审批请求。
使用数据库而不是内存存储，可以避免进程重启导致审批信息丢失。

设计要点：
1. 持久化存储：审批信息保存在PostgreSQL中
2. 状态管理：pending -> approve/reject
3. 消费标记：is_consumed标记审批是否已被恢复流程使用
4. 完整上下文：保存Agent信息、checkpoint信息，支持恢复执行

使用场景：
- SQL执行前的审批
- 代码执行前的审批
- 其他敏感操作的审批
- 审批后恢复执行

状态流转：
pending（待审批）-> approve（已批准）/ reject（已拒绝）
"""
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

    核心职责：
    1. 存储审批请求的基本信息
    2. 记录审批状态和决策
    3. 保存恢复执行所需的上下文信息
    4. 标记审批是否已被消费

    设计理由：
    1. 使用数据库而不是内存，避免进程重启丢失
    2. 完整的上下文信息，支持精确的恢复执行
    3. 消费标记，避免重复恢复
    4. 索引优化，支持高效查询

    字段说明：
    - id: 主键，自增
    - session_id: 会话ID，关联用户对话
    - message_id: 审批消息ID，唯一标识一次审批
    - action_name: 工具名称，如"execute_sql"、"execute_code"
    - action_args: 工具参数，JSONB格式
    - description: 审批描述，用户友好的提示
    - status: 审批状态，pending/approve/reject
    - user_id: 审批用户ID
    - decision_time: 审批时间
    - agent_name: 触发审批的Agent名称
    - subgraph_thread_id: 子图线程ID，用于恢复
    - checkpoint_id: checkpoint ID，用于恢复
    - checkpoint_ns: checkpoint namespace，用于恢复
    - is_consumed: 是否已被消费，避免重复恢复
    - create_time: 创建时间
    - update_time: 更新时间

    索引说明：
    - uq_interrupt_session_message: session_id + message_id 唯一约束
    - idx_interrupt_session_status_consumed: 查询会话的待审批记录
    - idx_interrupt_create_time: 按创建时间排序
    """

    __tablename__ = "t_interrupt_approval"
    __table_args__ = (
        # 唯一约束：同一会话的同一消息只能有一条审批记录
        UniqueConstraint("session_id", "message_id", name="uq_interrupt_session_message"),
        # 复合索引：查询会话的待审批记录
        Index("idx_interrupt_session_status_consumed", "session_id", "status", "is_consumed"),
        # 时间索引：按创建时间排序查询
        Index("idx_interrupt_create_time", "create_time"),
    )

    # 主键
    id = Column(BigInteger, primary_key=True, autoincrement=True, doc="主键")

    # 标识字段
    session_id = Column(String(120), nullable=False, index=True, doc="会话ID")
    message_id = Column(String(200), nullable=False, index=True, doc="审批消息ID")

    # 审批内容
    action_name = Column(String(120), nullable=False, doc="工具名称")
    action_args = Column(JSONB, nullable=True, doc="工具参数")
    description = Column(Text, nullable=True, doc="审批描述")

    # 审批状态
    status = Column(String(20), nullable=False, default="pending", index=True, doc="pending/approve/reject")
    user_id = Column(BigInteger, nullable=True, index=True, doc="审批用户ID")
    decision_time = Column(DateTime, nullable=True, doc="审批时间")

    # 恢复上下文
    agent_name = Column(String(120), nullable=True, doc="触发审批的Agent名称")
    subgraph_thread_id = Column(String(180), nullable=True, doc="子图线程ID")
    checkpoint_id = Column(String(200), nullable=True, doc="checkpoint_id")
    checkpoint_ns = Column(String(200), nullable=True, doc="checkpoint_ns")

    # 消费标记
    is_consumed = Column(Boolean, nullable=False, default=False, index=True, doc="是否已被恢复流程消费")

    # 时间戳
    create_time = Column(DateTime, server_default=func.now(), doc="创建时间")
    update_time = Column(DateTime, server_default=func.now(), onupdate=func.now(), doc="更新时间")

