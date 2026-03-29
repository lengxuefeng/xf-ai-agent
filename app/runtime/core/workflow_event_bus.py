# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional

from runtime.types import utc_now_iso


def workflow_timestamp() -> str:
    """返回统一 UTC 时间戳。"""
    return utc_now_iso()


def workflow_role_for_agent(agent_name: str) -> str:
    """根据 Agent 名称推断流程展示中的角色。"""
    normalized = str(agent_name or "").strip()
    if normalized in {"ChatAgent", "Aggregator", "chat_node", "aggregator_node"}:
        return "supervisor"
    if normalized:
        return "worker"
    return "system"


def workflow_display_name(agent_name: str) -> str:
    """将内部 Agent 名称转换为前端展示标签。"""
    normalized = str(agent_name or "").strip()
    display_map = {
        "ChatAgent": "掌柜",
        "Aggregator": "总管汇总",
        "yunyou_agent": "云柚专员",
        "sql_agent": "账房先生",
        "weather_agent": "天象司",
        "search_agent": "典籍司",
        "medical_agent": "医馆参谋",
        "code_agent": "工坊司",
        "chat_node": "掌柜",
        "aggregator_node": "总管汇总",
    }
    return display_map.get(normalized, normalized or "流程节点")


def build_workflow_event(
    *,
    session_id: str,
    run_id: str,
    phase: str,
    title: str,
    summary: str = "",
    status: str = "info",
    role: str = "system",
    agent_name: str = "",
    task_id: str = "",
    node_name: str = "",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构造统一结构化流程事件。"""
    payload: Dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "session_id": session_id,
        "run_id": run_id,
        "phase": phase,
        "title": title,
        "summary": summary,
        "status": status,
        "role": role,
        "timestamp": workflow_timestamp(),
    }
    if agent_name:
        payload["agent_name"] = agent_name
        payload["agent_label"] = workflow_display_name(agent_name)
    if task_id:
        payload["task_id"] = task_id
    if node_name:
        payload["node_name"] = node_name
    if meta:
        payload["meta"] = meta
    return payload


def parse_workflow_event_chunk(chunk: str) -> Optional[Dict[str, Any]]:
    """从 workflow_event SSE 中提取 payload。"""
    if not str(chunk or "").startswith("event: workflow_event"):
        return None
    try:
        data_line = next(
            (line for line in chunk.splitlines() if line.startswith("data: ")),
            None,
        )
        if not data_line:
            return None
        data = json.loads(data_line[6:])
    except (StopIteration, TypeError, json.JSONDecodeError):
        return None
    payload = data.get("payload")
    if isinstance(payload, dict):
        return dict(payload)
    return None

