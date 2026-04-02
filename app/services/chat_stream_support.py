# -*- coding: utf-8 -*-
import asyncio
import json
from typing import Any, Dict, Generator, Optional

from common.utils.stream_delta import resolve_stream_delta
from config.constants.chat_service_constants import (
    CHAT_AI_CONTENT_TYPES,
    CHAT_SERVICE_INTERRUPTED_APPEND_TEMPLATE,
    CHAT_SERVICE_INTERRUPTED_TEMPLATE,
)
from config.constants.sse_constants import SseEventType, SsePayloadField
from harness.core.run_context import build_run_context
from harness.core.run_state_store import run_state_store
from harness.workspace.manager import workspace_manager
from common.utils.assistant_text_sanitizer import strip_internal_execution_noise
from common.utils.chat_utils import ChatUtils


def is_async_stream(stream_obj: Any) -> bool:
    """判断对象是否为 async 可迭代流。"""
    return hasattr(stream_obj, "__aiter__")


def iterate_stream_sync(stream_obj: Any) -> Generator[str, None, None]:
    """
    在同步上下文里兼容遍历同步流和异步流。

    这个桥接层只负责把 chunk 一个个取出来，不夹带业务判断，
    方便 ChatService 只关心“拿到什么”和“怎么落库”。
    """
    if not is_async_stream(stream_obj):
        yield from stream_obj
        return

    loop = asyncio.new_event_loop()
    async_iter = stream_obj.__aiter__()
    try:
        while True:
            try:
                chunk = loop.run_until_complete(async_iter.__anext__())
            except StopAsyncIteration:
                break
            yield chunk
    finally:
        try:
            aclose = getattr(stream_obj, "aclose", None)
            if callable(aclose):
                loop.run_until_complete(aclose())
        except Exception:
            pass
        loop.close()


def close_stream_safely(stream_obj: Any) -> None:
    """统一关闭同步流和异步流，避免前端中断后遗留后台生成器。"""
    if stream_obj is None:
        return

    close_fn = getattr(stream_obj, "close", None)
    if callable(close_fn):
        try:
            close_fn()
            return
        except Exception:
            pass

    aclose_fn = getattr(stream_obj, "aclose", None)
    if callable(aclose_fn):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(aclose_fn())
        except Exception:
            pass
        finally:
            loop.close()


def parse_sse_payload(chunk: str) -> Optional[Dict[str, Any]]:
    """从 SSE chunk 中提取 JSON 负载，兼容 event/data 与纯 data 两种格式。"""
    try:
        if chunk.startswith("event: "):
            lines = chunk.split("\n")
            data_line = next((line for line in lines if line.startswith("data: ")), None)
            if data_line:
                return json.loads(data_line[6:])
        elif chunk.startswith("data: "):
            return json.loads(chunk[6:])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return None


def _append_workflow_trace(
    workflow_trace: list[dict],
    payload: Dict[str, Any],
) -> None:
    """把结构化流程事件写进轨迹里，并压缩高频 streaming 事件。"""
    workflow_payload = payload.get(SsePayloadField.PAYLOAD.value)
    if not isinstance(workflow_payload, dict):
        return

    normalized = dict(workflow_payload)
    phase = str(normalized.get("phase") or "")
    agent_name = str(normalized.get("agent_name") or "")
    task_id = str(normalized.get("task_id") or "")

    if phase in {"worker_streaming", "direct_response_streaming"}:
        for index in range(len(workflow_trace) - 1, -1, -1):
            existing = workflow_trace[index]
            if not isinstance(existing, dict):
                continue
            if (
                str(existing.get("phase") or "") == phase
                and str(existing.get("agent_name") or "") == agent_name
                and str(existing.get("task_id") or "") == task_id
            ):
                workflow_trace[index] = normalized
                return

    workflow_trace.append(normalized)


def _append_thinking_entry(thinking_entries: list[str], entry: str) -> None:
    """累积思考过程文本，并避开连续重复项。"""
    normalized = str(entry or "").strip()
    if not normalized:
        return
    if thinking_entries and thinking_entries[-1] == normalized:
        return
    thinking_entries.append(normalized)


def collect_trace_from_chunk(
    chunk: str,
    thinking_entries: list[str],
    workflow_trace: list[dict],
) -> None:
    """从单个 chunk 中提取可持久化的思考过程和流程轨迹。"""
    data = parse_sse_payload(chunk)
    if not data:
        return

    event_type = str(data.get(SsePayloadField.TYPE.value) or "")
    if event_type == SseEventType.WORKFLOW_EVENT.value:
        _append_workflow_trace(workflow_trace, data)
        return

    if event_type == SseEventType.THINKING.value:
        _append_thinking_entry(thinking_entries, str(data.get(SsePayloadField.CONTENT.value) or ""))
        return

    if event_type == SseEventType.LOG.value:
        _append_thinking_entry(thinking_entries, str(data.get("message") or ""))
        return

    if event_type == SseEventType.INTERRUPT.value:
        interrupt_content = data.get(SsePayloadField.CONTENT.value)
        if isinstance(interrupt_content, str):
            _append_thinking_entry(thinking_entries, interrupt_content)
        elif isinstance(interrupt_content, dict):
            _append_thinking_entry(
                thinking_entries,
                str(interrupt_content.get("message") or ""),
            )


def _build_thinking_trace_data(extra_data: Dict[str, Any], thinking_entries: list[str]) -> None:
    if thinking_entries:
        extra_data["thinking_trace"] = "\n\n".join(thinking_entries)

def _build_workflow_trace_data(extra_data: Dict[str, Any], workflow_trace: list[dict]) -> None:
    if workflow_trace:
        extra_data["workflow_trace"] = workflow_trace
        extra_data["workflow_version"] = 1

def _build_runtime_snapshot_data(extra_data: Dict[str, Any], session_id: str) -> None:
    if not session_id:
        return
    snapshot = run_state_store.get_latest_for_session(session_id)
    if not snapshot:
        return
    extra_data["runtime_snapshot"] = {
        "run_id": snapshot.run_id,
        "session_id": snapshot.session_id,
        "status": snapshot.status,
        "current_phase": snapshot.current_phase,
        "title": snapshot.title,
        "summary": snapshot.summary,
        "error": snapshot.error,
        "agent_name": snapshot.agent_name,
        "updated_at": snapshot.updated_at,
    }
    if snapshot.meta:
        extra_data["runtime_meta"] = snapshot.meta

def _write_artifacts_and_collect(
    extra_data: Dict[str, Any],
    session_id: str,
    final_response: str,
    workflow_trace: list[dict]
) -> None:
    if not session_id:
        return
    snapshot = run_state_store.get_latest_for_session(session_id)
    if not snapshot:
        return
    run_context = build_run_context(
        session_id=snapshot.session_id,
        user_input="",
        run_id=snapshot.run_id,
    )
    if final_response:
        workspace_manager.write_text_artifact(
            run_context,
            name="final_response.md",
            content=final_response,
            category="response",
            media_type="text/markdown",
        )
    if workflow_trace:
        workspace_manager.write_json_artifact(
            run_context,
            name="workflow_trace",
            payload=workflow_trace,
            category="trace",
        )
    artifacts = workspace_manager.list_artifacts(run_context)
    if artifacts:
        extra_data["runtime_artifacts"] = artifacts

def build_chat_extra_data(
    thinking_entries: list[str],
    workflow_trace: list[dict],
    *,
    session_id: str = "",
    final_response: str = "",
) -> Optional[Dict[str, Any]]:
    """构造消息落库需要的扩展数据，并同步写入运行产物。"""
    extra_data: Dict[str, Any] = {}
    _build_thinking_trace_data(extra_data, thinking_entries)
    _build_workflow_trace_data(extra_data, workflow_trace)
    _build_runtime_snapshot_data(extra_data, session_id)
    _write_artifacts_and_collect(extra_data, session_id, final_response, workflow_trace)
    return extra_data or None


def build_final_response(ai_response: str, error_occurred: bool, error_message: str) -> str:
    """把正常输出和异常补充语拼成最终落库正文。"""
    final_content = strip_internal_execution_noise(ai_response)
    if error_occurred:
        if not ai_response:
            final_content = CHAT_SERVICE_INTERRUPTED_TEMPLATE.format(error=error_message)
        else:
            final_content += CHAT_SERVICE_INTERRUPTED_APPEND_TEMPLATE.format(error=error_message)
    return final_content


def extract_ai_content_from_chunk(chunk: str) -> Optional[str]:
    """从 SSE chunk 中提取真正的模型正文，过滤 thinking 和 interrupt 事件。"""
    try:
        payload = parse_sse_payload(chunk)
        if not payload:
            return None
        if payload.get(SsePayloadField.TYPE.value) in CHAT_AI_CONTENT_TYPES:
            return strip_internal_execution_noise(
                payload.get(SsePayloadField.CONTENT.value, ""),
                trim=False,
                collapse_blank_lines=False,
            )
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return None


def merge_ai_response_from_chunk(accumulated_response: str, chunk: str) -> tuple[str, str]:
    """把单个 SSE chunk 归一化成正文增量，并合并进累计正文。"""
    ai_content = extract_ai_content_from_chunk(chunk)
    if not ai_content:
        return "", str(accumulated_response or "")
    return resolve_stream_delta(str(accumulated_response or ""), ai_content)


def normalize_history_messages(history_data: Any) -> list[dict]:
    """把不同来源的历史消息统一成 graph_runner 可消费的结构。"""
    if not history_data:
        return []

    if isinstance(history_data, dict):
        source_items = history_data.get("messages", []) or []
    elif isinstance(history_data, list):
        source_items = history_data
    else:
        source_items = []

    normalized = []
    for item in source_items:
        if isinstance(item, dict):
            normalized.append(item)
            continue

        normalized.append({
            "user_content": getattr(item, "user_content", ""),
            "model_content": getattr(item, "model_content", ""),
            "name": getattr(item, "model_name", None),
        })
    return normalized


def format_error_event(message: str) -> str:
    """构造统一错误事件，避免服务层散落字符串拼接。"""
    return ChatUtils.format_sse_data(
        event_type=SseEventType.ERROR.value,
        data={SsePayloadField.CONTENT.value: message},
    )
