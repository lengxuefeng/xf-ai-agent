# -*- coding: utf-8 -*-
"""
GraphRunner 流事件桥接层。

职责：
1. 解析现有 SSE chunk；
2. 复用现有 workflow/tool instrumentation；
3. 输出标准化 WebSocket JSON 事件。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from config.constants.sse_constants import SseEventType, SsePayloadField


def parse_sse_chunk(chunk: str) -> Optional[Dict[str, Any]]:
    """从 SSE chunk 中提取 JSON 负载。"""
    raw_chunk = str(chunk or "")
    if not raw_chunk.strip():
        return None

    event_name = ""
    data_lines: list[str] = []
    for line in raw_chunk.splitlines():
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if not data_lines:
        return None

    data_text = "\n".join(data_lines).strip()
    if not data_text:
        payload: Dict[str, Any] = {}
    else:
        try:
            parsed = json.loads(data_text)
        except (json.JSONDecodeError, TypeError):
            payload = {SsePayloadField.CONTENT.value: data_text}
        else:
            if isinstance(parsed, dict):
                payload = dict(parsed)
            else:
                payload = {SsePayloadField.CONTENT.value: parsed}

    if event_name and not payload.get(SsePayloadField.TYPE.value):
        payload[SsePayloadField.TYPE.value] = event_name
    return payload or None


def _coerce_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        if isinstance(parsed, dict):
            return dict(parsed)
    return {}


def _stringify_data(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _summary_from_workflow_event(workflow_payload: Dict[str, Any]) -> str:
    return str(
        workflow_payload.get("summary")
        or workflow_payload.get("title")
        or workflow_payload.get("phase")
        or ""
    ).strip()


class WsEventFormatter:
    """把 GraphRunner 的 SSE 事件归一成前端可直接消费的 WS JSON。"""

    def __init__(self) -> None:
        self._seen_tool_call_keys: set[str] = set()
        self._seen_tool_result_keys: set[str] = set()
        self._done_sent = False

    def consume(self, chunk: str) -> List[Dict[str, Any]]:
        parsed = parse_sse_chunk(chunk)
        if not parsed:
            return []

        event_type = str(parsed.get(SsePayloadField.TYPE.value) or "").strip()
        if not event_type:
            return []

        if event_type == SseEventType.RESPONSE_START.value:
            return []

        if event_type == SseEventType.RESPONSE_END.value:
            return self._emit_done()

        if event_type in {SseEventType.STREAM.value, "message"}:
            content = self._extract_content(parsed)
            return [{"type": "message", "content": content}] if content else []

        if event_type == SseEventType.THINKING.value:
            content = self._extract_content(parsed)
            return [{"type": "thinking", "content": content}] if content else []

        if event_type == SseEventType.LOG.value:
            content = str(parsed.get("message") or self._extract_content(parsed) or "").strip()
            return [{"type": "thinking", "content": content}] if content else []

        if event_type == SseEventType.INTERRUPT.value:
            content = self._extract_interrupt_content(parsed)
            return [{"type": "thinking", "content": content}] if content else []

        if event_type == SseEventType.TOOL_CALL.value:
            return self._normalize_tool_call_event(parsed.get(SsePayloadField.PAYLOAD.value) or parsed)

        if event_type == SseEventType.WORKFLOW_EVENT.value:
            workflow_payload = parsed.get(SsePayloadField.PAYLOAD.value)
            if isinstance(workflow_payload, dict):
                return self._normalize_workflow_event(workflow_payload)
            return []

        if event_type == SseEventType.ERROR.value:
            message = str(
                parsed.get("message")
                or parsed.get(SsePayloadField.CONTENT.value)
                or ""
            ).strip()
            return [{"type": "error", "message": message}] if message else []

        return []

    @staticmethod
    def _extract_content(payload: Dict[str, Any]) -> str:
        return str(
            payload.get(SsePayloadField.CONTENT.value)
            or payload.get("message")
            or ""
        ).strip()

    def _extract_interrupt_content(self, payload: Dict[str, Any]) -> str:
        raw_content = self._extract_content(payload)
        interrupt_payload = _coerce_dict(raw_content)
        if interrupt_payload:
            return str(
                interrupt_payload.get("message")
                or interrupt_payload.get("summary")
                or interrupt_payload.get("title")
                or raw_content
            ).strip()
        return raw_content

    def _normalize_workflow_event(self, workflow_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        phase = str(workflow_payload.get("phase") or "").strip()
        if phase == "tool_called":
            return self._normalize_tool_call_event(workflow_payload)
        if phase == "tool_completed":
            return self._normalize_tool_result_event(workflow_payload)

        content = _summary_from_workflow_event(workflow_payload)
        return [{"type": "thinking", "content": content}] if content else []

    def _normalize_tool_call_event(self, payload: Dict[str, Any] | Any) -> List[Dict[str, Any]]:
        tool_payload = _coerce_dict(payload)
        if not tool_payload:
            return []

        meta = _coerce_dict(tool_payload.get("meta"))
        tool_name = str(
            tool_payload.get("name")
            or tool_payload.get("tool_name")
            or meta.get("tool_name")
            or tool_payload.get("title")
            or ""
        ).strip()
        if tool_name.startswith("调用工具:"):
            tool_name = tool_name.split(":", 1)[1].strip()
        if tool_name.startswith("调用工具："):
            tool_name = tool_name.split("：", 1)[1].strip()

        args = _coerce_dict(tool_payload.get("args"))
        if not args:
            args = _coerce_dict(meta.get("args"))

        tool_call_key = str(
            tool_payload.get("tool_call_id")
            or meta.get("tool_call_id")
            or tool_payload.get("id")
            or tool_payload.get("call_id")
            or f"{tool_name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
        ).strip()
        if tool_call_key in self._seen_tool_call_keys:
            return []
        self._seen_tool_call_keys.add(tool_call_key)

        return [{
            "type": "tool_call",
            "tool_name": tool_name or "tool_call",
            "args": args,
        }]

    def _normalize_tool_result_event(self, workflow_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        meta = _coerce_dict(workflow_payload.get("meta"))
        tool_result_key = str(
            meta.get("tool_call_id")
            or workflow_payload.get("id")
            or workflow_payload.get("timestamp")
            or ""
        ).strip()
        if tool_result_key and tool_result_key in self._seen_tool_result_keys:
            return []
        if tool_result_key:
            self._seen_tool_result_keys.add(tool_result_key)

        raw_status = str(workflow_payload.get("status") or "").strip().lower()
        status = "error" if raw_status in {"failed", "error"} or meta.get("error") else "success"
        data = meta.get("error")
        if not data:
            data = meta.get("result")
        if not data:
            data = workflow_payload.get("summary") or workflow_payload.get("title") or ""

        return [{
            "type": "tool_result",
            "status": status,
            "data": _stringify_data(data),
        }]

    def _emit_done(self) -> List[Dict[str, Any]]:
        if self._done_sent:
            return []
        self._done_sent = True
        return [{"type": "done"}]


class SseToWsEventBridge(WsEventFormatter):
    """兼容旧调用方名称。"""


def ws_event_formatter(
    chunk: str,
    formatter: Optional[WsEventFormatter] = None,
) -> List[Dict[str, Any]]:
    """把单个 SSE chunk 格式化为前端可直接消费的 WS JSON 事件。"""
    active_formatter = formatter or WsEventFormatter()
    return active_formatter.consume(chunk)


def bridge_sse_chunk_to_ws_events(
    chunk: str,
    bridge: Optional[SseToWsEventBridge] = None,
) -> List[Dict[str, Any]]:
    """兼容旧入口。"""
    active_bridge = bridge or SseToWsEventBridge()
    return active_bridge.consume(chunk)


__all__ = [
    "WsEventFormatter",
    "SseToWsEventBridge",
    "bridge_sse_chunk_to_ws_events",
    "parse_sse_chunk",
    "ws_event_formatter",
]
