from fastapi import APIRouter, Depends, Query

from core.security import verify_token
from schemas.chat_history_schemas import ChatMessageCreate, ChatSessionUpdate, ChatSessionCreate, ChatSessionIn
from schemas.response_model import ResponseModel
from services.chat_history_service import chat_history_service

chat_history_router = APIRouter(prefix="/chat-history", tags=["聊天历史"])


@chat_history_router.post("/save-session", response_model=ResponseModel)
def create_chat_session(req: ChatSessionCreate):
    """
    创建或获取聊天会话
    """
    result = chat_history_service.get_or_create_session(req.user_id,req.session_id, req.title)
    return ResponseModel.success(data=result.model_dump(), message="会话创建成功")


@chat_history_router.post("/sessions", response_model=ResponseModel)
def create_chat_session(
        req: ChatSessionIn,
        user_id: int = Depends(verify_token)
):
    """
    创建新的聊天会话
    """
    result = chat_history_service.create_session(user_id, req)
    return ResponseModel.success(data=result.model_dump(), message="会话创建成功")


@chat_history_router.get("/sessions", response_model=ResponseModel)
def get_chat_sessions(
        page: int = Query(1, ge=1, description="页码"),
        size: int = Query(20, ge=1, le=100, description="每页数量"),
        user_id: int = Depends(verify_token)
):
    """
    获取会话列表
    """
    result = chat_history_service.get_user_sessions(user_id, page, size)
    return ResponseModel.success(data=result, message="获取会话列表成功")


@chat_history_router.put("/sessions/{session_id}", response_model=ResponseModel)
def update_chat_session_title(
        session_id: str,
        req: ChatSessionUpdate,
        user_id: int = Depends(verify_token)
):
    """
    更新会话标题
    """
    result = chat_history_service.update_session_title(user_id, session_id, req.title)
    return ResponseModel.success(data=result, message="会话标题更新成功")


@chat_history_router.delete("/sessions/{session_id}", response_model=ResponseModel)
def delete_chat_session(
        session_id: str,
        hard_delete: bool = Query(False, description="是否硬删除"),
        user_id: int = Depends(verify_token)
):
    """
    删除整个会话
    """
    result = chat_history_service.delete_session(user_id, session_id, hard_delete)
    message = "会话及相关消息已永久删除" if hard_delete else "会话及相关消息已移至回收站"
    return ResponseModel.success(data=result, message=message)


@chat_history_router.get("/sessions/{session_id}/messages", response_model=ResponseModel)
def get_session_messages(
        session_id: str,
        page: int = Query(1, ge=1, description="页码"),
        size: int = Query(50, ge=1, le=100, description="每页数量"),
        user_id: int = Depends(verify_token)
):
    """

    获取会话的聊天记录详情
    """
    result = chat_history_service.get_session_messages(user_id, session_id, page, size)
    return ResponseModel.success(data=result, message="获取会话详情成功")


@chat_history_router.post("/messages", response_model=ResponseModel)
def create_chat_message(
        req: ChatMessageCreate,
        user_id: int = Depends(verify_token)
):
    """
    创建聊天记录 (发送消息)
    """
    result = chat_history_service.create_chat_message(user_id, req)
    return ResponseModel.success(data=result.model_dump(), message="消息发送成功")


@chat_history_router.delete("/messages/{message_id}", response_model=ResponseModel)
def delete_chat_message(
        message_id: str,
        hard_delete: bool = Query(False, description="是否硬删除"),
        user_id: int = Depends(verify_token)
):
    """
    删除单条聊天消息
    """
    result = chat_history_service.delete_single_message(user_id, message_id, hard_delete)
    message = "消息已永久删除" if hard_delete else "消息已移至回收站"
    return ResponseModel.success(data=result, message=message)
