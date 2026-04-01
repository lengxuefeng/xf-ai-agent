# -*- coding: utf-8 -*-
import asyncio
import contextlib
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from common.core.security import verify_token
from common.utils.custom_logger import get_logger
from common.utils.pwd_utils import TokenError, encryption_utils
from db import get_db
from harness.graph_runner_events import WsEventFormatter, ws_event_formatter
from models.schemas.chat_schemas import StreamChatRequest
from services.chat_service import chat_service
from services.request_cancellation_service import request_cancellation_service

log = get_logger(__name__)

"""
定义流式聊天 API 接口
"""

chat_router = APIRouter()
_WS_QUEUE_CLOSE = object()


def _resolve_ws_user_id(token: str) -> int:
    normalized_token = str(token or "").strip()
    if not normalized_token:
        raise HTTPException(status_code=401, detail="缺少认证 token")
    try:
        return encryption_utils.get_user_id_from_token(normalized_token)
    except TokenError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def _build_ws_request(
    payload: dict[str, Any],
    *,
    session_id: str,
    anonymous: bool,
) -> StreamChatRequest:
    request_payload = dict(payload)
    request_payload["session_id"] = session_id
    req = StreamChatRequest(**request_payload)
    if anonymous and req.user_model_id is not None:
        raise ValueError("匿名 WebSocket 不支持 user_model_id。")
    return req


async def _enqueue_stream_events(
    event_queue: asyncio.Queue,
    *,
    req: StreamChatRequest,
    session_id: str,
    user_id: int | None,
) -> None:
    formatter = WsEventFormatter()
    done_sent = False
    request_cancellation_service.register_request(session_id)

    try:
        stream_gen = chat_service.build_stream_generator(
            req=req,
            user_id=user_id,
            request_id=uuid.uuid4().hex,
        )
        async for chunk in stream_gen:
            for event in ws_event_formatter(chunk, formatter=formatter):
                if str(event.get("type") or "") == "done":
                    done_sent = True
                await event_queue.put(event)
    except asyncio.CancelledError:
        request_cancellation_service.cancel_request(session_id)
        raise
    except Exception as exc:
        log.exception(f"WebSocket 后台流任务异常: session_id={session_id}, error={exc}")
        await event_queue.put({"type": "error", "message": str(exc)})
    finally:
        if not done_sent:
            await event_queue.put({"type": "done"})


async def _websocket_sender_loop(
    websocket: WebSocket,
    event_queue: asyncio.Queue,
) -> None:
    while True:
        event = await event_queue.get()
        if event is _WS_QUEUE_CLOSE:
            return
        await websocket.send_json(event)


async def _websocket_receiver_loop(
    websocket: WebSocket,
    event_queue: asyncio.Queue,
    *,
    session_id: str,
    user_id: int | None,
    anonymous: bool,
) -> None:
    active_stream_task: asyncio.Task | None = None

    try:
        while True:
            try:
                payload = await websocket.receive_json()
            except WebSocketDisconnect:
                request_cancellation_service.cancel_request(session_id)
                return
            except Exception as exc:
                await event_queue.put({"type": "error", "message": f"无效的 WebSocket JSON 消息: {exc}"})
                continue

            if not isinstance(payload, dict):
                await event_queue.put({"type": "error", "message": "WebSocket 消息必须是 JSON 对象。"})
                continue

            control_type = str(payload.get("type") or payload.get("action") or "").strip().lower()
            if control_type in {"cancel", "stop"}:
                request_cancellation_service.cancel_request(session_id)
                if active_stream_task is not None and not active_stream_task.done():
                    active_stream_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await active_stream_task
                continue

            if active_stream_task is not None and not active_stream_task.done():
                await event_queue.put({
                    "type": "error",
                    "message": "当前会话已有任务正在执行，请等待完成后再发送新请求。",
                })
                continue

            try:
                req = _build_ws_request(
                    payload,
                    session_id=session_id,
                    anonymous=anonymous,
                )
            except Exception as exc:
                await event_queue.put({"type": "error", "message": str(exc)})
                continue

            active_stream_task = asyncio.create_task(
                _enqueue_stream_events(
                    event_queue,
                    req=req,
                    session_id=session_id,
                    user_id=user_id,
                )
            )
    finally:
        if active_stream_task is not None and not active_stream_task.done():
            active_stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await active_stream_task


async def _serve_websocket_chat(
    websocket: WebSocket,
    *,
    session_id: str,
    user_id: int | None,
    anonymous: bool,
) -> None:
    event_queue: asyncio.Queue = asyncio.Queue()
    sender_task: asyncio.Task | None = None
    receiver_task: asyncio.Task | None = None

    await websocket.accept()

    try:
        sender_task = asyncio.create_task(_websocket_sender_loop(websocket, event_queue))
        receiver_task = asyncio.create_task(
            _websocket_receiver_loop(
                websocket,
                event_queue,
                session_id=session_id,
                user_id=user_id,
                anonymous=anonymous,
            )
        )

        done, pending = await asyncio.wait(
            {sender_task, receiver_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            if task.cancelled():
                continue
            exc = task.exception()
            if exc is not None and not isinstance(exc, WebSocketDisconnect):
                raise exc

        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    except WebSocketDisconnect:
        request_cancellation_service.cancel_request(session_id)
    except Exception as exc:
        log.exception(f"WebSocket 聊天异常: session_id={session_id}, error={exc}")
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": str(exc)})
    finally:
        request_cancellation_service.cancel_request(session_id)
        request_cancellation_service.cleanup_request(session_id)

        await event_queue.put(_WS_QUEUE_CLOSE)
        for task in (sender_task, receiver_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        with contextlib.suppress(Exception):
            await websocket.close()


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
    log.info(
        f"流式聊天请求进入路由: session_id={req.session_id}, "
        f"request_id={getattr(request.state, 'request_id', '') or getattr(request.state, 'req_id', '')}"
    )
    return await chat_service.process_stream_chat_async(
        req=req,
        request=request,
        user_id=user_id,
        db=db,
    )


@chat_router.post("/chat/stream/anonymous", summary="匿名流式聊天接口")
async def stream_chat_anonymous(
    req: StreamChatRequest,
    request: Request,
) -> StreamingResponse:
    """
    匿名用户的流式聊天接口，不需要认证，不保存聊天历史。
    """
    log.info(
        f"匿名流式聊天请求进入路由: session_id={req.session_id}, "
        f"request_id={getattr(request.state, 'request_id', '') or getattr(request.state, 'req_id', '')}"
    )
    if req.user_model_id is not None:
        raise HTTPException(status_code=400, detail="匿名接口不支持 user_model_id")

    return await chat_service.process_stream_chat_async(
        req=req,
        request=request,
        user_id=None,
        db=None,
    )


@chat_router.websocket("/ws/v1/chat/{session_id}")
async def websocket_stream_chat(
    websocket: WebSocket,
    session_id: str,
):
    token = str(websocket.query_params.get("token") or "").strip()
    try:
        user_id = _resolve_ws_user_id(token)
    except HTTPException as exc:
        await websocket.close(code=4401, reason=str(exc.detail))
        return

    await _serve_websocket_chat(
        websocket,
        session_id=session_id,
        user_id=user_id,
        anonymous=False,
    )


@chat_router.websocket("/ws/v1/chat/anonymous/{session_id}")
async def websocket_stream_chat_anonymous(
    websocket: WebSocket,
    session_id: str,
):
    await _serve_websocket_chat(
        websocket,
        session_id=session_id,
        user_id=None,
        anonymous=True,
    )
