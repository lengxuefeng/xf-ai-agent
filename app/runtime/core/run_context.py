# -*- coding: utf-8 -*-
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from runtime.types import RunContext


def build_run_id(session_id: str, *, is_resume: bool = False) -> str:
    """构造统一 run_id。"""
    normalized_session_id = str(session_id or "").strip() or "anonymous"
    if is_resume:
        return f"{normalized_session_id}:resume:{uuid.uuid4().hex}"
    return f"{normalized_session_id}:{uuid.uuid4().hex}"


def build_run_context(
    *,
    session_id: str,
    user_input: str,
    model_config: Optional[Dict[str, Any]] = None,
    history_messages: Optional[List[Dict[str, Any]]] = None,
    session_context: Optional[Dict[str, Any]] = None,
    is_resume: bool = False,
    run_id: str = "",
) -> RunContext:
    """根据当前请求构造统一 RunContext。"""
    effective_session_id = str(session_id or "").strip()
    effective_context = session_context or {}
    effective_history = history_messages or []
    resolved_run_id = str(run_id or "").strip() or build_run_id(
        effective_session_id,
        is_resume=is_resume,
    )
    return RunContext(
        session_id=effective_session_id,
        run_id=resolved_run_id,
        user_input=str(user_input or ""),
        model_config=dict(model_config or {}),
        history_size=len(effective_history),
        is_resume=is_resume,
        context_summary=str(effective_context.get("context_summary") or ""),
        meta={
            "input_length": len(str(user_input or "")),
            "has_context_slots": bool(effective_context.get("context_slots")),
            "history_size": len(effective_history),
        },
    )

