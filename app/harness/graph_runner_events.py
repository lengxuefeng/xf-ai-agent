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
import re
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


def _normalize_usage_payload(value: Any) -> Optional[Dict[str, int]]:
    if not isinstance(value, dict):
        return None

    def _to_int(raw: Any) -> int:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0

    input_tokens = _to_int(value.get("input_tokens") or value.get("prompt_tokens"))
    output_tokens = _to_int(value.get("output_tokens") or value.get("completion_tokens"))
    total_tokens = _to_int(value.get("total_tokens") or value.get("total"))
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens

    if input_tokens <= 0 and output_tokens <= 0 and total_tokens <= 0:
        return None

    return {
        "input": input_tokens,
        "output": output_tokens,
        "total": total_tokens,
    }


def _extract_usage_from_nested_payload(value: Any, *, depth: int = 0) -> Optional[Dict[str, int]]:
    if depth > 5 or value is None:
        return None

    normalized = _normalize_usage_payload(value)
    if normalized:
        return normalized

    if isinstance(value, dict):
        priority_keys = (
            "token_usage",
            "usage",
            "usage_metadata",
            "response_metadata",
            "llm_output",
            "data",
            "payload",
            "chunk",
            "message",
            "delta",
            "choices",
        )
        for key in priority_keys:
            if key in value:
                nested = _extract_usage_from_nested_payload(value.get(key), depth=depth + 1)
                if nested:
                    return nested

        for nested_value in value.values():
            nested = _extract_usage_from_nested_payload(nested_value, depth=depth + 1)
            if nested:
                return nested

    if isinstance(value, (list, tuple)):
        for item in value:
            nested = _extract_usage_from_nested_payload(item, depth=depth + 1)
            if nested:
                return nested

    return None


def _extract_usage_from_response_metadata(response_metadata: Any) -> Optional[Dict[str, int]]:
    if not isinstance(response_metadata, dict):
        return None
    for key in ("token_usage", "usage", "usage_metadata"):
        normalized = _normalize_usage_payload(response_metadata.get(key))
        if normalized:
            return normalized
    return _extract_usage_from_nested_payload(response_metadata)


def _resolve_usage_from_chunk_metadata(payload: Any) -> Optional[Dict[str, int]]:
    if not isinstance(payload, dict):
        return None

    chunk = payload.get("chunk")
    if chunk is None:
        return None

    usage_metadata = None
    response_metadata = None
    if isinstance(chunk, dict):
        usage_metadata = chunk.get("usage_metadata")
        response_metadata = chunk.get("response_metadata")
    else:
        usage_metadata = getattr(chunk, "usage_metadata", None)
        response_metadata = getattr(chunk, "response_metadata", None)

    normalized = _normalize_usage_payload(usage_metadata)
    if normalized:
        return normalized

    response_metadata_dict = _coerce_dict(response_metadata)
    if response_metadata_dict:
        normalized = _normalize_usage_payload(response_metadata_dict.get("token_usage"))
        if normalized:
            return normalized
        normalized = _extract_usage_from_response_metadata(response_metadata_dict)
        if normalized:
            return normalized
    return None


def _resolve_usage(payload: Any) -> Optional[Dict[str, int]]:
    if not isinstance(payload, dict):
        return None

    chunk_usage = _resolve_usage_from_chunk_metadata(payload)
    if chunk_usage:
        return chunk_usage

    meta = _coerce_dict(payload.get("meta"))
    raw_event = _coerce_dict(payload.get("raw_event"))
    raw_meta_event = _coerce_dict(meta.get("raw_event"))

    usage_candidates = (
        payload.get("usage"),
        payload.get("usage_metadata"),
        payload.get("token_usage"),
        meta.get("usage"),
        meta.get("usage_metadata"),
        meta.get("token_usage"),
        raw_event.get("usage"),
        raw_event.get("usage_metadata"),
        raw_event.get("token_usage"),
        raw_meta_event.get("usage"),
        raw_meta_event.get("usage_metadata"),
        raw_meta_event.get("token_usage"),
    )
    for candidate in usage_candidates:
        normalized = _normalize_usage_payload(candidate)
        if normalized:
            return normalized

    response_metadata_candidates = (
        payload.get("response_metadata"),
        meta.get("response_metadata"),
        payload.get("llm_response_metadata"),
        raw_event.get("response_metadata"),
        raw_meta_event.get("response_metadata"),
    )
    for candidate in response_metadata_candidates:
        normalized = _extract_usage_from_response_metadata(candidate)
        if normalized:
            return normalized

    for candidate in usage_candidates + response_metadata_candidates:
        normalized = _extract_usage_from_nested_payload(candidate)
        if normalized:
            return normalized

    return None


def _first_non_empty_value(*candidates: Any) -> Any:
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, str) and not candidate.strip():
            continue
        if isinstance(candidate, (dict, list)) and not candidate:
            continue
        return candidate
    return None


def _summary_from_workflow_event(workflow_payload: Dict[str, Any]) -> str:
    return str(
        workflow_payload.get("summary")
        or workflow_payload.get("title")
        or workflow_payload.get("phase")
        or ""
    ).strip()


_NODE_NAME_ALIASES = {
    "parent_planner": "planner",
    "planner_node": "planner",
    "parent_planner_node": "planner",
    "chat_agent": "chat_agent",
    "chat_node": "chat_node",
    "aggregator": "aggregator",
    "aggregator_node": "aggregator",
    "supervisor_final": "supervisor_final",
    "worker_node": "worker",
    "dispatch_node": "dispatch",
    "dispatcher_node": "dispatch",
    "reflection_node": "reflection",
}

INTERMEDIATE_NODES = {
    "planner",
    "weather_agent",
    "sql_agent",
    "yunyou_agent",
    "search_agent",
    "medical_agent",
    "code_agent",
    "tool_node",
}
FINAL_MESSAGE_NODES = {
    "chat_node",
    "chat_agent",
    "supervisor_final",
    "aggregator",
}

_RUNNING_STATUSES = {"active", "running", "in_progress", "processing"}
_COMPLETED_STATUSES = {"completed", "done", "success", "succeeded", "finished"}
_WAITING_STATUSES = {"waiting", "waiting_approval", "pending"}
_ERROR_STATUSES = {"failed", "error"}
_CANCELLED_STATUSES = {"cancelled", "canceled"}


def _to_snake_identifier(value: Any) -> str:
    raw_value = str(value or "").strip()
    if not raw_value:
        return ""

    normalized = raw_value.replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized)
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_").lower()
    return normalized


def _normalize_node_name(value: Any) -> str:
    normalized = _to_snake_identifier(value)
    if not normalized:
        return ""
    return _NODE_NAME_ALIASES.get(normalized, normalized)


def _resolve_workflow_node(workflow_payload: Dict[str, Any]) -> str:
    meta = _coerce_dict(workflow_payload.get("meta"))
    candidates = [
        workflow_payload.get("node"),
        workflow_payload.get("node_name"),
        workflow_payload.get("agent_name"),
        workflow_payload.get("agent_label"),
        meta.get("node"),
        meta.get("node_name"),
        meta.get("agent_name"),
        meta.get("agent"),
    ]
    for candidate in candidates:
        normalized = _normalize_node_name(candidate)
        if normalized:
            return normalized
    return ""


def _resolve_trace_status(status: Any) -> str:
    raw_status = str(status or "").strip().lower()
    if raw_status in _RUNNING_STATUSES:
        return "active"
    if raw_status in _COMPLETED_STATUSES:
        return "completed"
    if raw_status in _WAITING_STATUSES:
        return "waiting"
    if raw_status in _ERROR_STATUSES:
        return "error"
    if raw_status in _CANCELLED_STATUSES:
        return "cancelled"
    return raw_status or "info"


def _resolve_lifecycle_status(status: Any) -> str:
    raw_status = str(status or "").strip().lower()
    if raw_status in _RUNNING_STATUSES:
        return "running"
    return "completed"


def _normalize_allowed_decisions(value: Any) -> List[str]:
    if not isinstance(value, list):
        return ["approve", "reject"]

    normalized = [
        str(item or "").strip().lower()
        for item in value
        if str(item or "").strip()
    ]
    return normalized or ["approve", "reject"]


def _normalize_action_request(action_request: Any, *, fallback_message_id: str = "") -> Dict[str, Any]:
    action_payload = _coerce_dict(action_request)
    if not action_payload:
        return {}

    action_name = str(
        action_payload.get("action_name")
        or action_payload.get("name")
        or action_payload.get("tool_name")
        or action_payload.get("title")
        or ""
    ).strip()
    action_args = _coerce_dict(
        action_payload.get("action_args")
        or action_payload.get("args")
    )
    action_id = str(
        action_payload.get("id")
        or action_payload.get("message_id")
        or fallback_message_id
        or ""
    ).strip()

    normalized = dict(action_payload)
    normalized["id"] = action_id
    normalized["name"] = action_name or "tool_call"
    normalized["args"] = action_args
    normalized["action_name"] = action_name or "tool_call"
    normalized["action_args"] = action_args
    return normalized


def _normalize_interrupt_payload(raw_payload: Any, parsed_payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = _coerce_dict(raw_payload)
    if not payload:
        payload = dict(parsed_payload)
        payload.pop("type", None)

    raw_content = str(
        payload.get("content")
        or parsed_payload.get("content")
        or payload.get("message")
        or ""
    ).strip()
    if not payload and raw_content:
        payload = {"message": raw_content}

    nested_payload = _coerce_dict(payload.get("payload"))
    if nested_payload:
        payload = {**nested_payload, **payload}

    message_id = str(payload.get("message_id") or "").strip()
    action_requests = []
    raw_action_requests = payload.get("action_requests")
    if isinstance(raw_action_requests, list):
        for raw_action in raw_action_requests:
            normalized_action = _normalize_action_request(raw_action, fallback_message_id=message_id)
            if normalized_action:
                action_requests.append(normalized_action)

    if not message_id and action_requests:
        message_id = str(action_requests[0].get("id") or "").strip()

    agent_name = str(
        payload.get("agent_name")
        or payload.get("node")
        or payload.get("node_name")
        or parsed_payload.get("agent_name")
        or ""
    ).strip()
    normalized_node = _resolve_workflow_node({
        **parsed_payload,
        **payload,
        "agent_name": agent_name,
    })

    normalized = dict(payload)
    normalized["message_id"] = message_id
    normalized["message"] = str(
        payload.get("message")
        or payload.get("summary")
        or payload.get("title")
        or raw_content
        or "需要人工审核"
    ).strip()
    normalized["allowed_decisions"] = _normalize_allowed_decisions(payload.get("allowed_decisions"))
    normalized["action_requests"] = action_requests
    normalized["agent_name"] = agent_name or normalized_node
    normalized["node"] = normalized_node
    normalized["node_name"] = normalized_node
    return normalized


def _normalize_workflow_payload(workflow_payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(workflow_payload)
    raw_status = str(payload.get("status") or "").strip()
    node = _resolve_workflow_node(payload)
    normalized_status = _resolve_trace_status(raw_status)
    usage = _resolve_usage(payload)

    meta = _coerce_dict(payload.get("meta"))
    if node and not meta.get("node"):
        meta["node"] = node
    if raw_status and not meta.get("workflow_status"):
        meta["workflow_status"] = raw_status
    if payload.get("node_name") and not meta.get("node_name"):
        meta["node_name"] = payload.get("node_name")
    if payload.get("agent_name") and not meta.get("agent_name"):
        meta["agent_name"] = payload.get("agent_name")
    if usage and not meta.get("usage"):
        meta["usage"] = usage

    payload["input"] = _first_non_empty_value(
        payload.get("input"),
        meta.get("input"),
        meta.get("args"),
        meta.get("task"),
        meta.get("tasks"),
        meta.get("request"),
        _coerce_dict(payload.get("raw_event")).get("input"),
    )
    payload["output"] = _first_non_empty_value(
        payload.get("output"),
        meta.get("output"),
        meta.get("result"),
        meta.get("error"),
        meta.get("error_payload"),
        _coerce_dict(payload.get("raw_event")).get("output"),
    )

    payload["node"] = node
    payload["status"] = normalized_status
    payload["meta"] = meta
    if usage:
        payload["usage"] = usage
    return payload


def _build_workflow_event_message(workflow_payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized_payload = _normalize_workflow_payload(workflow_payload)
    node = str(normalized_payload.get("node") or "").strip()
    workflow_status = str(
        normalized_payload.get("meta", {}).get("workflow_status")
        or normalized_payload.get("status")
        or ""
    ).strip()
    return {
        "type": "workflow_event",
        "node": node or "workflow",
        "phase": str(normalized_payload.get("phase") or "").strip(),
        "status": _resolve_lifecycle_status(workflow_status),
        "workflow_status": normalized_payload.get("status") or "info",
        "timestamp": normalized_payload.get("timestamp"),
        "title": str(normalized_payload.get("title") or "").strip(),
        "summary": _summary_from_workflow_event(normalized_payload),
        "usage": normalized_payload.get("usage"),
        "payload": normalized_payload,
    }


def _is_planner_node(node_name: Any) -> bool:
    normalized = _normalize_node_name(node_name)
    return normalized == "planner" or "planner" in normalized


def _rewrite_frontend_event_type(event_payload: Dict[str, Any], fallback_type: str) -> Dict[str, Any]:
    payload = dict(event_payload or {})
    source_node = _normalize_node_name(
        payload.get("node_name")
        or payload.get("node")
        or payload.get("name")
        or payload.get("agent_name")
    )
    payload["node"] = source_node
    payload["node_name"] = source_node
    content_str = str(payload.get("content", ""))

    # 核心防火墙：LangGraph 底层消息对象绝不能作为普通正文透传到前端。
    is_graph_state_dump = (
        "AIMessage(" in content_str
        or "HumanMessage(" in content_str
        or "SystemMessage(" in content_str
    )

    if source_node in FINAL_MESSAGE_NODES:
        if is_graph_state_dump:
            payload["type"] = "workflow_event"
            payload["status"] = "completed"
            return payload
        payload["type"] = "message"
        return payload

    if source_node in INTERMEDIATE_NODES:
        payload["type"] = "thinking"
        return payload

    payload["type"] = fallback_type
    return payload


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
            if not content:
                return []

            payload = parsed.get(SsePayloadField.PAYLOAD.value)
            source_payload = payload if isinstance(payload, dict) else parsed
            node = _resolve_workflow_node(source_payload)
            frontend_event = _rewrite_frontend_event_type({
                "content": content,
                "status": "thinking" if _is_planner_node(node) else "streaming",
                "node": node,
                "node_name": node,
                "usage": _resolve_usage(source_payload) or _resolve_usage(parsed),
            }, "message")
            if frontend_event.get("type") == "thinking" and not frontend_event.get("status"):
                frontend_event["status"] = "thinking"
            return [frontend_event]

        if event_type == SseEventType.THINKING.value:
            content = self._extract_content(parsed)
            if not content:
                return []
            payload = parsed.get(SsePayloadField.PAYLOAD.value)
            source_payload = payload if isinstance(payload, dict) else parsed
            return [_rewrite_frontend_event_type({
                "content": content,
                "status": "thinking",
                "node": _resolve_workflow_node(source_payload),
                "node_name": _resolve_workflow_node(source_payload),
                "usage": _resolve_usage(source_payload) or _resolve_usage(parsed),
            }, "thinking")]

        if event_type == SseEventType.LOG.value:
            content = str(parsed.get("message") or self._extract_content(parsed) or "").strip()
            return [{
                "type": "thinking",
                "content": content,
                "status": "thinking",
                "usage": _resolve_usage(parsed),
            }] if content else []

        if event_type == SseEventType.INTERRUPT.value:
            interrupt_source = (
                parsed.get(SsePayloadField.PAYLOAD.value)
                or parsed.get(SsePayloadField.CONTENT.value)
                or parsed
            )
            interrupt_payload = _normalize_interrupt_payload(interrupt_source, parsed)
            content = self._extract_interrupt_content(interrupt_payload)
            if not content:
                return []
            return [{
                "type": "approval_required",
                "content": content,
                "message": content,
                "status": "waiting_approval",
                "message_id": str(interrupt_payload.get("message_id") or "").strip(),
                "allowed_decisions": interrupt_payload.get("allowed_decisions") or ["approve", "reject"],
                "action_requests": interrupt_payload.get("action_requests") or [],
                "agent_name": str(interrupt_payload.get("agent_name") or "").strip(),
                "node": str(interrupt_payload.get("node") or "").strip(),
                "node_name": str(interrupt_payload.get("node_name") or "").strip(),
                "payload": interrupt_payload,
                "usage": _resolve_usage(parsed) or _resolve_usage(interrupt_payload),
            }]

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
        interrupt_payload = _normalize_interrupt_payload(payload, payload)
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
        events: List[Dict[str, Any]] = [_build_workflow_event_message(workflow_payload)]
        if phase == "tool_called":
            events.extend(self._normalize_tool_call_event(workflow_payload))
            return events
        if phase == "tool_completed":
            events.extend(self._normalize_tool_result_event(workflow_payload))
            return events
        return events

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
            "tool_call_id": tool_call_key,
            "args": args,
            "status": "running",
            "content": f"🛠 调用工具：{tool_name or 'tool_call'}",
            "node": _resolve_workflow_node(tool_payload),
            "timestamp": tool_payload.get("timestamp"),
            "usage": _resolve_usage(tool_payload),
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
            "tool_call_id": str(meta.get("tool_call_id") or workflow_payload.get("tool_call_id") or "").strip(),
            "tool_name": str(meta.get("tool_name") or meta.get("name") or "").strip(),
            "data": data,
            "raw": data,
            "content": _stringify_data(data),
            "node": _resolve_workflow_node(workflow_payload),
            "timestamp": workflow_payload.get("timestamp"),
            "meta": meta,
            "usage": _resolve_usage(workflow_payload),
        }]

    def _emit_done(self) -> List[Dict[str, Any]]:
        if self._done_sent:
            return []
        self._done_sent = True
        return [{"type": "done", "status": "completed"}]


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
