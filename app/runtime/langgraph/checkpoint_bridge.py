# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict

from harness.types import RunContext


class CheckpointBridge:
    """把 RunContext 投影成 LangGraph checkpoint 元信息。"""

    def describe_run(self, run_context: RunContext) -> Dict[str, str | bool]:
        return {
            "thread_id": str(run_context.session_id or ""),
            "run_id": str(run_context.run_id or ""),
            "request_id": str(run_context.request_id or ""),
            "is_resume": bool(run_context.is_resume),
        }


checkpoint_bridge = CheckpointBridge()

