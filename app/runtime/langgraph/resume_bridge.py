# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict

from langgraph.types import Command


class ResumeBridge:
    """统一 resume 命令桥。"""

    @staticmethod
    def build_resume_command(decision: str) -> Command:
        return Command(resume=str(decision or "").strip())

    @staticmethod
    def describe_resume_meta(resume_meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "message_id": resume_meta.get("message_id"),
            "agent_name": resume_meta.get("agent_name"),
            "decision": resume_meta.get("decision"),
            "checkpoint_id": resume_meta.get("checkpoint_id"),
            "checkpoint_ns": resume_meta.get("checkpoint_ns"),
        }


resume_bridge = ResumeBridge()

