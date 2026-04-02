# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from harness.types import utc_now_iso


@dataclass(slots=True)
class ToolPermissionDecision:
    """单次工具调用的权限裁决结果。"""

    tool_name: str
    decision: str
    reason: str = ""
    requires_approval: bool = False
    allowed: bool = False
    created_at: str = field(default_factory=utc_now_iso)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolExecutionRequest:
    """Runtime Tool 调用请求。"""

    tool_name: str
    args: Dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    source_agent: str = ""
    session_id: str = ""
    run_id: str = ""
    approval_granted: bool = False
    created_at: str = field(default_factory=utc_now_iso)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolExecutionReport:
    """Runtime Tool 执行报告。"""

    request: ToolExecutionRequest
    permission: ToolPermissionDecision
    ok: bool = False
    status: str = "pending"
    result: Any = None
    error: str = ""
    duration_ms: int = 0
    created_at: str = field(default_factory=utc_now_iso)
    hooks: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["request"] = self.request.to_dict()
        payload["permission"] = self.permission.to_dict()
        return payload

