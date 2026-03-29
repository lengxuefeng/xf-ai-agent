# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List

from runtime.hooks.base import HookResult
from runtime.hooks.builtin.approval_hook import run_approval_hook
from runtime.hooks.builtin.quality_hook import run_quality_hook
from runtime.hooks.builtin.tool_guard_hook import run_tool_guard_hook
from runtime.types import RunContext


class RuntimeHookManager:
    """统一 Hook 管理器。"""

    def run_pre_run_hooks(self, run_context: RunContext) -> List[HookResult]:
        return [
            run_approval_hook(run_context),
            run_tool_guard_hook(run_context),
            run_quality_hook(run_context),
        ]

    def run_post_run_hooks(self, run_context: RunContext, final_text: str = "") -> List[HookResult]:
        summary = "已生成最终答复，可继续触发落库与产物归档。"
        if not str(final_text or "").strip():
            summary = "运行结束，但未收到最终正文，后续需检查补偿链路。"
        return [
            HookResult(
                name="post_run_finalize",
                status="completed",
                summary=summary,
                meta={"final_text_length": len(str(final_text or ""))},
            )
        ]


runtime_hook_manager = RuntimeHookManager()

