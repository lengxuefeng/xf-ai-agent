# -*- coding: utf-8 -*-

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from core.security import verify_token
from db import get_db
from schemas.chat_schemas import StreamChatRequest
from services.chat_service import chat_service
from utils.custom_logger import get_logger

log = get_logger(__name__)

"""
定义流式聊天 API 接口
"""

# 创建一个独立的 API 路由，并命名为 chat_router 以便 main.py 导入
chat_router = APIRouter()


@chat_router.post("/chat/stream", summary="流式聊天接口")
async def stream_chat(
    req: StreamChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    user_id: int = Depends(verify_token),
) -> StreamingResponse:
    """
    处理流式聊天请求。

    本接口会实时返回 Agent 的思考过程和最终结果，采用 Server-Sent Events (SSE) 格式。
    支持用户认证，认证用户的聊天记录会自动保存到聊天历史中。
    """
    return await chat_service.process_stream_chat_async(
        req=req,
        request=request,
        user_id=user_id,
        db=db
    )


@chat_router.post("/chat/stream/anonymous", summary="匿名流式聊天接口")
async def stream_chat_anonymous(
    req: StreamChatRequest,
    request: Request,
) -> StreamingResponse:
    """
    匿名用户的流式聊天接口，不需要认证，不保存聊天历史。
    """
    if req.user_model_id is not None:
        raise HTTPException(status_code=400, detail="匿名接口不支持 user_model_id")
        
    return await chat_service.process_stream_chat_async(
        req=req,
        request=request,
        user_id=None,
        db=None
    )
