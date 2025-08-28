from typing import List, Optional
from fastapi import APIRouter, Depends, Query

from core.security import verify_token
from schemas.response_model import ResponseModel
from schemas.chat_history_schemas import ChatHistoryCreate, ChatHistoryUpdate
from services.chat_history_service import chat_history_service

chat_history_router = APIRouter(prefix="/chat-history", tags=["聊天历史"])

"""
聊天历史接口
"""


@chat_history_router.post("/create", response_model=ResponseModel)
def create_chat_history(
    req: ChatHistoryCreate,
    user_id: int = Depends(verify_token)
):
    """
    创建聊天记录
    """
    result = chat_history_service.create_chat(user_id, req)
    return ResponseModel.success(data=result, message="聊天记录创建成功")


@chat_history_router.get("/list", response_model=ResponseModel)
def get_chat_history_list(
    session_id: Optional[str] = Query(None, description="会话ID"),
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(10, ge=1, le=100, description="每页数量"),
    user_id: int = Depends(verify_token)
):
    """
    获取聊天历史列表
    """
    result = chat_history_service.get_chat_list(
        user_id=user_id,
        session_id=session_id,
        page=page,
        size=size
    )
    return ResponseModel.success(data=result, message="获取聊天历史成功")


@chat_history_router.get("/sessions", response_model=ResponseModel)
def get_chat_sessions(
    limit: int = Query(20, ge=1, le=100, description="最大返回数量"),
    user_id: int = Depends(verify_token)
):
    """
    获取会话列表
    """
    result = chat_history_service.get_user_sessions(user_id, limit)
    return ResponseModel.success(data=result, message="获取会话列表成功")


@chat_history_router.get("/{chat_id}", response_model=ResponseModel)
def get_chat_history_detail(
    chat_id: str,
    user_id: int = Depends(verify_token)
):
    """
    获取单条聊天记录详情
    """
    result = chat_history_service.get_chat_by_id(user_id, chat_id)
    return ResponseModel.success(data=result, message="获取聊天记录成功")


@chat_history_router.put("/{chat_id}", response_model=ResponseModel)
def update_chat_history(
    chat_id: str,
    req: ChatHistoryUpdate,
    user_id: int = Depends(verify_token)
):
    """
    更新聊天记录
    """
    result = chat_history_service.update_chat(user_id, chat_id, req)
    return ResponseModel.success(data=result, message="更新聊天记录成功")


@chat_history_router.delete("/{chat_id}", response_model=ResponseModel)
def delete_chat_history(
    chat_id: str,
    hard_delete: bool = Query(False, description="是否硬删除"),
    user_id: int = Depends(verify_token)
):
    """
    删除聊天记录
    """
    result = chat_history_service.delete_chat(user_id, chat_id, hard_delete)
    
    message = "永久删除成功" if hard_delete else "移至回收站成功"
    return ResponseModel.success(data=result, message=message)


@chat_history_router.delete("/session/{session_id}", response_model=ResponseModel)
def delete_chat_session(
    session_id: str,
    hard_delete: bool = Query(False, description="是否硬删除"),
    user_id: int = Depends(verify_token)
):
    """
    删除整个会话
    """
    result = chat_history_service.delete_session(user_id, session_id, hard_delete)
    
    deleted_count = result["deleted_count"]
    message = f"已永久删除 {deleted_count} 条聊天记录" if hard_delete else f"已移至回收站 {deleted_count} 条聊天记录"
    return ResponseModel.success(data=result, message=message)


@chat_history_router.get("/search/keyword", response_model=ResponseModel)
def search_chat_history(
    keyword: str = Query(..., description="搜索关键词"),
    session_id: Optional[str] = Query(None, description="限定会话ID"),
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(10, ge=1, le=100, description="每页数量"),
    user_id: int = Depends(verify_token)
):
    """
    搜索聊天历史
    """
    result = chat_history_service.search_chats(
        user_id=user_id,
        keyword=keyword,
        session_id=session_id,
        page=page,
        size=size
    )
    return ResponseModel.success(data=result, message=f"搜索到 {result['total']} 条相关记录")


@chat_history_router.get("/statistics/overview", response_model=ResponseModel)
def get_chat_statistics(
    days: int = Query(30, ge=1, le=365, description="统计天数"),
    user_id: int = Depends(verify_token)
):
    """
    获取聊天统计信息
    """
    result = chat_history_service.get_user_statistics(user_id, days)
    return ResponseModel.success(data=result, message="获取统计信息成功")
