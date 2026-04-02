# -*- coding: utf-8 -*-
import asyncio
import contextlib
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse
from starlette.websockets import WebSocketState

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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _enrich_ws_event(
    event: dict[str, Any],
    *,
    session_id: str,
    sequence: int,
) -> dict[str, Any]:
    enriched = dict(event or {})
    event_type = str(enriched.get("type") or "").strip()
    payload = enriched.get("payload") if isinstance(enriched.get("payload"), dict) else {}
    payload_meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    run_id = str(
        enriched.get("run_id")
        or payload.get("run_id")
        or payload_meta.get("run_id")
        or ""
    ).strip()
    request_id = str(
        enriched.get("request_id")
        or payload.get("request_id")
        or payload_meta.get("request_id")
        or ""
    ).strip()

    enriched["session_id"] = session_id
    enriched.setdefault("sequence", sequence)
    enriched.setdefault("timestamp", enriched.get("timestamp") or payload.get("timestamp") or _utc_now_iso())
    if run_id:
        enriched["run_id"] = run_id
    if request_id:
        enriched["request_id"] = request_id

    if event_type == "message":
        enriched.setdefault("status", "streaming")
    elif event_type == "response_start":
        enriched.setdefault("status", "thinking")
    elif event_type == "response_delta":
        enriched.setdefault("status", "streaming")
    elif event_type == "response_end":
        enriched.setdefault("status", "completed")
    elif event_type == "thinking":
        enriched.setdefault("status", "thinking")
    elif event_type == "tool_call":
        enriched.setdefault("status", "running")
    elif event_type == "tool_result":
        enriched.setdefault("status", "completed")
    elif event_type == "workflow_event":
        if payload:
            enriched.setdefault("node", payload.get("node") or payload.get("node_name") or payload.get("agent_name") or "")
            enriched.setdefault("workflow_status", payload.get("status") or "info")
        enriched.setdefault("status", "running")
    elif event_type == "approval_required":
        enriched.setdefault("status", "waiting_approval")
    elif event_type == "error":
        enriched.setdefault("status", "error")
    elif event_type == "done":
        enriched.setdefault("status", "completed")

    return enriched


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
    payload_session_id = str(request_payload.get("session_id") or "").strip()
    if payload_session_id and payload_session_id != session_id:
        raise ValueError("WebSocket 会话ID与请求体中的 session_id 不一致。")
    request_payload["session_id"] = session_id
    req = StreamChatRequest(**request_payload)
    if anonymous and req.user_model_id is not None:
        raise ValueError("匿名 WebSocket 不支持 user_model_id。")
    return req


def _build_safe_ws_error_message(exc: Exception) -> str:
    text = str(exc or "").strip()
    lowered = text.lower()
    if any(marker in text for marker in ("当前未绑定可用模型", "当前选择的模型配置", "模型配置缺少", "当前模型配置缺少", "请先在设置中")):
        return text
    if "timeout" in lowered or "超时" in text:
        return "请求超时，请稍后重试。"
    if any(marker in lowered for marker in ("connectionerror", "operationalerror", "psycopg2", "connection to server at", "host is down", "network")):
        return "服务暂时不可用，请稍后再试。"
    return "系统内部服务暂时不可用，请稍后再试。"


def _build_safe_ws_request_error_message(exc: Exception) -> str:
    text = str(exc or "").strip()
    if "匿名 websocket 不支持 user_model_id" in text.lower():
        return "匿名 WebSocket 不支持 user_model_id。"
    if "session_id" in text.lower() and "不一致" in text:
        return "当前会话状态已过期，请刷新会话后重试。"
    return "请求参数有误，请检查输入后重试。"


async def _enqueue_stream_events(
    event_queue: asyncio.Queue,
    *,
    req: StreamChatRequest,
    session_id: str,
    user_id: int | None,
) -> None:
    formatter = WsEventFormatter()
    done_sent = False
    event_sequence = 0
    request_cancellation_service.register_request(session_id)

    try:
        stream_gen = chat_service.build_stream_generator(
            req=req,
            user_id=user_id,
            request_id=uuid.uuid4().hex,
        )
        async for chunk in stream_gen:
            for event in ws_event_formatter(chunk, formatter=formatter):
                event_sequence += 1
                normalized_event = _enrich_ws_event(
                    event,
                    session_id=session_id,
                    sequence=event_sequence,
                )
                if str(normalized_event.get("type") or "") in {"response_end", "done"}:
                    done_sent = True
                await event_queue.put(normalized_event)
    except asyncio.CancelledError:
        request_cancellation_service.cancel_request(session_id)
        raise
    except Exception as exc:
        log.exception(f"WebSocket 后台流任务异常: session_id={session_id}, error={exc}")
        event_sequence += 1
        await event_queue.put(_enrich_ws_event(
            {"type": "error", "message": _build_safe_ws_error_message(exc), "status": "error"},
            session_id=session_id,
            sequence=event_sequence,
        ))
    finally:
        if not done_sent:
            event_sequence += 1
            await event_queue.put(_enrich_ws_event(
                {"type": "response_end", "status": "completed"},
                session_id=session_id,
                sequence=event_sequence,
            ))


def _is_websocket_connected(websocket: WebSocket) -> bool:
    return (
        websocket.application_state == WebSocketState.CONNECTED
        and websocket.client_state == WebSocketState.CONNECTED
    )


async def _safe_send_ws_json(websocket: WebSocket, event: dict[str, Any]) -> bool:
    if not _is_websocket_connected(websocket):
        return False
    if websocket.client_state != WebSocketState.CONNECTED:
        return False
    await websocket.send_json(event)
    return True


async def _cancel_stream_task(
    stream_task: asyncio.Task | None,
    *,
    session_id: str,
) -> None:
    if stream_task is None:
        return
    if stream_task.done():
        return

    request_cancellation_service.cancel_request(session_id)
    stream_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await stream_task


async def _websocket_sender_loop(
    websocket: WebSocket,
    event_queue: asyncio.Queue,
    *,
    session_id: str,
    stream_task_holder: dict[str, asyncio.Task | None],
) -> None:
    while True:
        event = await event_queue.get()
        if event is _WS_QUEUE_CLOSE:
            return

        if not _is_websocket_connected(websocket):
            await _cancel_stream_task(stream_task_holder.get("stream_task"), session_id=session_id)
            stream_task_holder["stream_task"] = None
            return

        try:
            if await _safe_send_ws_json(websocket, event):
                continue
            await _cancel_stream_task(stream_task_holder.get("stream_task"), session_id=session_id)
            stream_task_holder["stream_task"] = None
            return
        except WebSocketDisconnect:
            await _cancel_stream_task(stream_task_holder.get("stream_task"), session_id=session_id)
            stream_task_holder["stream_task"] = None
            return
        except Exception as exc:
            log.warning(f"WebSocket 发送事件失败，停止发送循环: session_id={session_id}, error={exc}")
            await _cancel_stream_task(stream_task_holder.get("stream_task"), session_id=session_id)
            stream_task_holder["stream_task"] = None
            return


async def _websocket_receiver_loop(
    websocket: WebSocket,
    event_queue: asyncio.Queue,
    *,
    session_id: str,
    user_id: int | None,
    anonymous: bool,
    stream_task_holder: dict[str, asyncio.Task | None],
) -> None:
    try:
        while True:
            try:
                payload = await websocket.receive_json()
            except WebSocketDisconnect:
                await _cancel_stream_task(stream_task_holder.get("stream_task"), session_id=session_id)
                stream_task_holder["stream_task"] = None
                return
            except Exception as exc:
                if not _is_websocket_connected(websocket):
                    await _cancel_stream_task(stream_task_holder.get("stream_task"), session_id=session_id)
                    stream_task_holder["stream_task"] = None
                    return
                await event_queue.put({"type": "error", "message": f"无效的 WebSocket JSON 消息: {exc}"})
                continue

            if not isinstance(payload, dict):
                await event_queue.put({"type": "error", "message": "WebSocket 消息必须是 JSON 对象。"})
                continue

            control_type = str(payload.get("type") or payload.get("action") or "").strip().lower()
            if control_type in {"cancel", "stop"}:
                await _cancel_stream_task(stream_task_holder.get("stream_task"), session_id=session_id)
                stream_task_holder["stream_task"] = None
                continue

            active_stream_task = stream_task_holder.get("stream_task")
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
                await event_queue.put(
                    {"type": "error", "message": _build_safe_ws_request_error_message(exc)}
                )
                continue

            stream_task_holder["stream_task"] = asyncio.create_task(
                _enqueue_stream_events(
                    event_queue,
                    req=req,
                    session_id=session_id,
                    user_id=user_id,
                )
            )
    finally:
        await _cancel_stream_task(stream_task_holder.get("stream_task"), session_id=session_id)
        stream_task_holder["stream_task"] = None


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
    stream_task_holder: dict[str, asyncio.Task | None] = {"stream_task": None}

    await websocket.accept()

    try:
        sender_task = asyncio.create_task(
            _websocket_sender_loop(
                websocket,
                event_queue,
                session_id=session_id,
                stream_task_holder=stream_task_holder,
            )
        )
        receiver_task = asyncio.create_task(
            _websocket_receiver_loop(
                websocket,
                event_queue,
                session_id=session_id,
                user_id=user_id,
                anonymous=anonymous,
                stream_task_holder=stream_task_holder,
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
        await _cancel_stream_task(stream_task_holder.get("stream_task"), session_id=session_id)
        stream_task_holder["stream_task"] = None
    except Exception as exc:
        log.exception(f"WebSocket 聊天异常: session_id={session_id}, error={exc}")
        if _is_websocket_connected(websocket):
            with contextlib.suppress(Exception):
                await _safe_send_ws_json(
                    websocket,
                    {"type": "error", "message": _build_safe_ws_error_message(exc)},
                )
    finally:
        await _cancel_stream_task(stream_task_holder.get("stream_task"), session_id=session_id)
        stream_task_holder["stream_task"] = None
        request_cancellation_service.cancel_request(session_id)
        request_cancellation_service.cleanup_request(session_id)

        await event_queue.put(_WS_QUEUE_CLOSE)
        for task in (sender_task, receiver_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        if websocket.application_state != WebSocketState.DISCONNECTED:
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
