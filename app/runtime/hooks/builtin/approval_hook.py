# -*- coding: utf-8 -*-
from __future__ import annotations

from runtime.hooks.base import HookResult
from runtime.types import RunContext


def run_approval_hook(run_context: RunContext) -> HookResult:
    if run_context.is_resume:
        return HookResult(
            name="approval_gate",
            status="info",
            summary="当前请求为审批恢复流程，沿用既有批示结果。",
            meta={"is_resume": True},
        )
    return HookResult(
        name="approval_gate",
        status="completed",
        summary="审批门禁已就绪，敏感动作将进入人工批示链路。",
    )

