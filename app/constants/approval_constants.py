# -*- coding: utf-8 -*-
"""人工审批链路常量。"""
from enum import Enum
from typing import Tuple


class ApprovalDecision(str, Enum):
    """用户审批动作枚举"""

    # 批准操作
    APPROVE = "approve"

    # 拒绝操作
    REJECT = "reject"


class ApprovalStatus(str, Enum):
    """审批记录状态枚举"""

    # 待审批
    PENDING = "pending"

    # 已批准
    APPROVE = "approve"

    # 已拒绝
    REJECT = "reject"


# 允许的审批动作列表
DEFAULT_ALLOWED_DECISIONS: Tuple[str, ...] = (
    ApprovalDecision.APPROVE.value,  # 允许批准
    ApprovalDecision.REJECT.value,   # 允许拒绝
)


# 默认中断提示消息
DEFAULT_INTERRUPT_MESSAGE = "需要人工审核"

# SQL 审批动作名称
SQL_APPROVAL_ACTION_NAME = "execute_sql"

# 代码执行审批动作名称
CODE_APPROVAL_ACTION_NAME = "execute_code"

# 代码执行审批提示消息
CODE_APPROVAL_MESSAGE = "需要审批代码执行"

# 审批已处理状态标识
APPROVAL_HANDLE_STATUS_PROCESSED = "processed"
