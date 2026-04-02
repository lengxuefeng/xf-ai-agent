# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional

from harness.hooks.base import HookResult
from harness.hooks.builtin.approval_hook import run_approval_hook
from harness.hooks.builtin.quality_hook import run_quality_hook
from harness.hooks.builtin.tool_guard_hook import run_tool_guard_hook
from harness.types import RunContext


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

    def run_pre_tool_hooks(
        self,
        run_context: Optional[RunContext],
        *,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        permission: Optional[Dict[str, Any]] = None,
    ) -> List[HookResult]:
        hook_results: List[HookResult] = []
        if run_context is not None:
            guard_result = run_tool_guard_hook(run_context)
            guard_meta = dict(guard_result.meta or {})
            guard_meta.update({
                "tool_name": str(tool_name or "").strip(),
                "args": dict(args or {}),
            })
            hook_results.append(
                HookResult(
                    name=f"{guard_result.name}:{tool_name}",
                    status=guard_result.status,
                    summary=guard_result.summary,
                    meta=guard_meta,
                )
            )

        decision = dict(permission or {})
        decision_value = str(decision.get("decision") or "").strip()
        if decision_value == "ask":
            hook_results.append(
                HookResult(
                    name="tool_approval_gate",
                    status="waiting",
                    summary=f"工具 `{tool_name}` 需要人工审批。",
                    meta={"tool_name": tool_name, "permission": decision},
                )
            )
        elif decision_value == "deny":
            hook_results.append(
                HookResult(
                    name="tool_permission_gate",
                    status="rejected",
                    summary=str(decision.get("reason") or f"工具 `{tool_name}` 未获授权。"),
                    meta={"tool_name": tool_name, "permission": decision},
                )
            )
        else:
            hook_results.append(
                HookResult(
                    name="tool_permission_gate",
                    status="completed",
                    summary=f"工具 `{tool_name}` 权限校验通过。",
                    meta={"tool_name": tool_name, "permission": decision},
                )
            )
        return hook_results

    def run_post_tool_hooks(
        self,
        run_context: Optional[RunContext],
        *,
        tool_name: str,
        report: Optional[Dict[str, Any]] = None,
    ) -> List[HookResult]:
        normalized_report = dict(report or {})
        status = str(normalized_report.get("status") or "completed")
        summary = f"工具 `{tool_name}` 调用完成。"
        if status == "failed":
            summary = f"工具 `{tool_name}` 调用失败，需要检查执行结果。"
        elif status == "waiting_approval":
            summary = f"工具 `{tool_name}` 已挂起，等待审批恢复。"
        elif status == "denied":
            summary = f"工具 `{tool_name}` 已被权限层拦截。"
        meta = {
            "tool_name": str(tool_name or "").strip(),
            "status": status,
            "duration_ms": int(normalized_report.get("duration_ms") or 0),
        }
        if run_context is not None:
            meta["run_id"] = str(run_context.run_id or "")
        return [
            HookResult(
                name="tool_finalize",
                status="completed" if status == "completed" else status,
                summary=summary,
                meta=meta,
            )
        ]


runtime_hook_manager = RuntimeHookManager()
