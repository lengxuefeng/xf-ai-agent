# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional

from runtime.query.models import RuntimeEvent


def build_runtime_event(
    *,
    event_type: str,
    phase: str,
    title: str,
    summary: str = "",
    status: str = "info",
    meta: Optional[Dict[str, Any]] = None,
) -> RuntimeEvent:
    """构造统一 RuntimeEvent。"""
    return RuntimeEvent(
        event_type=str(event_type or "").strip() or "runtime_event",
        phase=str(phase or "").strip(),
        title=str(title or "").strip(),
        summary=str(summary or "").strip(),
        status=str(status or "info").strip() or "info",
        meta=dict(meta or {}),
    )

