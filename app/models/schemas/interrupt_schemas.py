from typing import Optional, Dict, Any

from models.schemas.base import BaseSchema


class InterruptApprovalRequest(BaseSchema):
    """人工审核批准请求"""
    session_id: str
    message_id: str
    decision: str  # 'approve' 或 'reject'
    action_name: Optional[str] = None
    action_args: Optional[Dict[str, Any]] = None


class InterruptApprovalResponse(BaseSchema):
    """人工审核批准响应"""
    success: bool
    message: str
