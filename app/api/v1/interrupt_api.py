from fastapi import APIRouter, Depends, HTTPException
from core.security import verify_token

from schemas.interrupt_schemas import InterruptApprovalRequest, InterruptApprovalResponse
from services.interrupt_service import interrupt_service

"""
人工审核接口
用于处理工具调用的人工审核
"""

interrupt_router = APIRouter()


@interrupt_router.post("/interrupt/approve", summary="批准工具调用")
async def approve_interrupt(
    req: InterruptApprovalRequest,
    user_id: int = Depends(verify_token)
) -> InterruptApprovalResponse:
    """
    批准工具调用

    Args:
        req: 审核批准请求
        user_id: 用户ID

    Returns:
        审核结果
    """
    try:
        result = interrupt_service.handle_approval(
            session_id=req.session_id,
            message_id=req.message_id,
            decision=req.decision,
            user_id=user_id,
            action_name=req.action_name,
            action_args=req.action_args
        )
        return InterruptApprovalResponse(
            success=True,
            message=f"已{req.decision == 'approve' and '批准' or '拒绝'}工具调用"
        )
    except Exception as e:
        return InterruptApprovalResponse(
            success=False,
            message=f"处理失败: {str(e)}"
        )
