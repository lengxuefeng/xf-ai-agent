# -*- coding: utf-8 -*-

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from core.security import verify_token
from db import get_db
from schemas.chat_schemas import StreamChatRequest
from services.chat_service import chat_service
from services.request_cancellation_service import request_cancellation_service
from utils.custom_logger import get_logger

log = get_logger(__name__)

"""
定义流式聊天 API 接口
"""

# 创建一个独立的 API 路由，并命名为 chat_router 以便 main.py 导入
chat_router = APIRouter()


def _attach_disconnect_cancellation(
    response: StreamingResponse,
    request: Request,
    session_id: str,
) -> StreamingResponse:
    """
    为 SSE 响应附加断连取消感知。

    说明：
    - 路由层先以 `session_id` 注册取消令牌；
    - GraphRunner 内部再把真实 `run_id` 绑定到该 session；
    - 客户端断连时，取消会沿 session -> run 级联传播。
    """
    original_body = response.body_iterator
    request_cancellation_service.register_request(session_id)

    async def _disconnect_aware_generator():
        try:
            async with request_cancellation_service.cancel_on_disconnect(session_id, request):
                async for chunk in original_body:
                    if await request.is_disconnected():
                        log.info(f"客户端断连，停止 SSE 推送。session_id={session_id[:16]}")
                        request_cancellation_service.cancel_request(session_id)
                        break
                    yield chunk
        finally:
            request_cancellation_service.cleanup_request(session_id)

    response.body_iterator = _disconnect_aware_generator()
    return response


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

    **返回事件流:**
    - `{"type": "thinking", "content": "..."}`: Agent 的思考过程
    - `{"type": "interrupt", "content": "..."}`: 需要用户输入的暂停事件
    - `{"type": "response_start", "content": ""}`: 响应开始
    - `{"type": "stream", "content": "..."}`: 逐字符流式内容
    - `{"type": "response_end", "content": ""}`: 响应结束
    - `{"type": "error", "content": "..."}`: 执行过程中发生的错误
    """
    # Middleware 会在这里挂载动态模型配置
    dynamic_model_config = getattr(request.state, "model_config", None)
    response = await chat_service.process_stream_chat_async(req, user_id, db, dynamic_model_config)
    return _attach_disconnect_cancellation(response, request, req.session_id)


@chat_router.post("/chat/stream/anonymous", summary="匿名流式聊天接口")
async def stream_chat_anonymous(
    req: StreamChatRequest,
    request: Request,
) -> StreamingResponse:
    """
    匿名用户的流式聊天接口，不需要认证，不保存聊天历史。

    接口参数和返回格式与认证接口相同，但不会保存任何聊天记录。
    适用于游客用户或不需要保存历史的场景。
    """
    if req.user_model_id is not None:
        raise HTTPException(status_code=400, detail="匿名接口不支持 user_model_id")
    response = await chat_service.process_stream_chat_async(req, user_id=None, db=None)
    return _attach_disconnect_cancellation(response, request, req.session_id)
