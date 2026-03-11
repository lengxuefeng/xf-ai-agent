# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from core.security import verify_token
from db import get_db
from schemas.chat_history_schemas import (
    ChatMessageCreate, ChatSessionCreate,
    ChatSessionIn, ChatSession, ChatMessage
)
from schemas.response_model import ResponseModel
from services.chat_history_service import chat_history_service

chat_history_router = APIRouter(prefix="/chat-history", tags=["聊天历史"])


@chat_history_router.post("/save-session", response_model=ResponseModel)
def create_chat_session_base(req: ChatSessionCreate, db: Session = Depends(get_db)):
    """保存/获取会话"""
    result = chat_history_service.get_or_create_session(db, req.user_id, req.session_id, req.title)
    # 将 ORM 对象转为 Pydantic 字典返回
    data = ChatSession.model_validate(result).model_dump()
    return ResponseModel.success(data=data, message="会话创建成功")


@chat_history_router.post("/sessions", response_model=ResponseModel)
def create_chat_session(
        req: ChatSessionIn,
        db: Session = Depends(get_db),
        user_id: int = Depends(verify_token)
):
    """创建新会话"""
    result = chat_history_service.create_session(db, user_id, req)
    data = ChatSession.model_validate(result).model_dump()
    return ResponseModel.success(data=data, message="会话创建成功")


@chat_history_router.get("/sessions", response_model=ResponseModel)
def get_chat_sessions(
        page: int = Query(1, ge=1),
        size: int = Query(20, ge=1, le=100),
        db: Session = Depends(get_db),
        user_id: int = Depends(verify_token)
):
    """获取会话列表"""
    sessions = chat_history_service.get_user_sessions(db, user_id, page, size)
    # 列表推导式进行批量验证
    data = [ChatSession.model_validate(s).model_dump() for s in sessions]
    return ResponseModel.success(data=data)


@chat_history_router.get("/sessions/{session_id}/messages", response_model=ResponseModel)
def get_session_messages(
        session_id: str,
        page: int = Query(1, ge=1),
        size: int = Query(50, ge=1, le=100),
        order: str = Query("desc", pattern="^(asc|desc)$"),
        db: Session = Depends(get_db),
        user_id: int = Depends(verify_token)
):
    """获取消息历史"""
    result = chat_history_service.get_session_messages(db, user_id, session_id, page, size, order)
    messages = result.get("messages", [])
    data = {
        "messages": [ChatMessage.model_validate(m).model_dump() for m in messages],
        "total": result.get("total", 0),
        "page": result.get("page", page),
        "size": result.get("size", size),
        "has_more": result.get("has_more", False),
        "order": result.get("order", order),
    }
    return ResponseModel.success(data=data)


@chat_history_router.post("/messages", response_model=ResponseModel)
def create_chat_message(
        req: ChatMessageCreate,
        db: Session = Depends(get_db),
        user_id: int = Depends(verify_token)
):
    """保存消息"""
    result = chat_history_service.create_chat_message(db, user_id, req)
    data = ChatMessage.model_validate(result).model_dump()
    return ResponseModel.success(data=data)
