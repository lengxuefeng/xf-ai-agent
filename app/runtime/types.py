# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict


def utc_now_iso() -> str:
    """返回统一 UTC 时间戳。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunStatus(str, Enum):
    """运行态状态枚举。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    WAITING_APPROVAL = "waiting_approval"


@dataclass(slots=True)
class RunContext:
    """单次运行的统一上下文。"""

    session_id: str
    run_id: str
    user_input: str
    model_config: Dict[str, Any] = field(default_factory=dict)
    history_size: int = 0
    is_resume: bool = False
    context_summary: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    meta: Dict[str, Any] = field(default_factory=dict)

    def graph_config(self) -> Dict[str, Dict[str, str]]:
        """构造 LangGraph 使用的 configurable 配置。"""
        return {
            "configurable": {
                "thread_id": self.session_id,
                "run_id": self.run_id,
            }
        }


@dataclass(slots=True)
class RunStateSnapshot:
    """运行态快照，供健康检查和调试复用。"""

    run_id: str
    session_id: str
    status: str
    current_phase: str = ""
    title: str = ""
    summary: str = ""
    error: str = ""
    agent_name: str = ""
    last_workflow_event: Dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=utc_now_iso)
    meta: Dict[str, Any] = field(default_factory=dict)

