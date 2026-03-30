# -*- coding: utf-8 -*-
"""
人工审批链路常量。

定义了系统中所有需要人工介入的审批流程相关的常量。
审批流程是系统安全的重要保障，确保敏感操作（如执行SQL、运行代码）必须经过用户确认。

审批流程设计：
1. Agent执行到敏感操作时抛出中断
2. 中断信息被持久化到数据库（避免重启丢失）
3. 前端显示审批界面给用户
4. 用户选择批准或拒绝
5. 后端根据用户决定恢复执行或拒绝操作
"""
from enum import Enum
from typing import Tuple


class ApprovalDecision(str, Enum):
    """
    用户审批动作枚举。

    定义用户可以对审批请求采取的动作。
    注意：这里只定义了最基本的两个动作，未来可以根据业务需要扩展（如"修改参数后批准"）。
    """

    # 批准操作：用户同意执行这个敏感操作
    APPROVE = "approve"

    # 拒绝操作：用户不同意执行这个敏感操作
    REJECT = "reject"


class ApprovalStatus(str, Enum):
    """
    审批记录状态枚举。

    定义审批记录在数据库中可能存在的状态。
    状态流转：PENDING -> APPROVE/REJECT
    """

    # 待审批：Agent刚触发中断，等待用户决策
    PENDING = "pending"

    # 已批准：用户同意执行，等待恢复流程
    APPROVE = "approve"

    # 已拒绝：用户拒绝执行，操作将被取消
    REJECT = "reject"


# 允许的审批动作列表
# 用于在前端渲染审批按钮时判断显示哪些选项
# 场景：将来可能增加"延期审批"、"转交他人审批"等选项
DEFAULT_ALLOWED_DECISIONS: Tuple[str, ...] = (
    ApprovalDecision.APPROVE.value,  # 允许批准
    ApprovalDecision.REJECT.value,   # 允许拒绝
)


# 默认中断提示消息
# 当具体的Agent没有提供中断描述时的兜底文案
# 场景：某个Agent的提示文案缺失，确保用户至少能看到一个友好的提示
DEFAULT_INTERRUPT_MESSAGE = "需要人工审核"

# SQL 审批动作名称
# 用于标识这个审批请求是关于SQL执行的
# 场景：Agent判断用户意图是查询数据库，生成SQL后需要审批
SQL_APPROVAL_ACTION_NAME = "execute_sql"

# 代码执行审批动作名称
# 用于标识这个审批请求是关于代码执行的
# 场景：Agent生成了一段Python代码，需要用户确认才能执行
CODE_APPROVAL_ACTION_NAME = "execute_code"

# 代码执行审批提示消息
# 具体的代码执行审批提示，比默认消息更明确
# 场景：前端显示"需要审批代码执行"，用户知道自己在审批什么
CODE_APPROVAL_MESSAGE = "需要审批代码执行"

# 审批已处理状态标识
# 用于标记一个审批请求已经被流程消费（恢复执行后不再重复处理）
# 场景：用户批准后，流程恢复并执行操作，完成后需要标记这个审批已被消费，避免重复
APPROVAL_HANDLE_STATUS_PROCESSED = "processed"
