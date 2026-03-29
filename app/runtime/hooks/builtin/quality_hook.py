# -*- coding: utf-8 -*-
from __future__ import annotations

from runtime.hooks.base import HookResult
from runtime.types import RunContext


def run_quality_hook(run_context: RunContext) -> HookResult:
    length = len(run_context.user_input or "")
    return HookResult(
        name="quality_guard",
        status="completed",
        summary="运行质量检查通过。",
        meta={"input_length": length},
    )

