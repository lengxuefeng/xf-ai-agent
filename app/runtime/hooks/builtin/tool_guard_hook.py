# -*- coding: utf-8 -*-
from __future__ import annotations

from runtime.hooks.base import HookResult
from runtime.types import RunContext


def run_tool_guard_hook(run_context: RunContext) -> HookResult:
    text = (run_context.user_input or "").lower()
    suspicious = any(keyword in text for keyword in ("drop table", "rm -rf", "删除数据库", "清空数据"))
    if suspicious:
        return HookResult(
            name="tool_guard",
            status="waiting",
            summary="检测到高风险操作语义，后续工具执行将进入更严格审批。",
            meta={"high_risk": True},
        )
    return HookResult(
        name="tool_guard",
        status="completed",
        summary="工具安全护栏检查通过。",
        meta={"high_risk": False},
    )

