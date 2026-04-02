# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from config.runtime_settings import get_run_mode
from harness.hooks.hook_manager import runtime_hook_manager
from harness.types import RunContext
from runtime.modes import runtime_mode_profile_resolver
from runtime.tools.executor import RuntimeToolExecutor, runtime_tool_executor
from runtime.tools.models import ToolExecutionReport, ToolExecutionRequest
from runtime.tools.permissions import ToolPermissionResolver, tool_permission_resolver
from tools.runtime_tools.tool_registry import runtime_tool_registry


class ToolRuntimeOrchestrator:
    """统一 Tool Runtime 编排器。"""

    def __init__(
        self,
        *,
        permission_resolver: ToolPermissionResolver | None = None,
        executor: RuntimeToolExecutor | None = None,
    ) -> None:
        self._permission_resolver = permission_resolver or tool_permission_resolver
        self._executor = executor or runtime_tool_executor

    def build_request(
        self,
        tool_name: str,
        *,
        args: Optional[Dict[str, Any]] = None,
        source_agent: str = "",
        run_context: Optional[RunContext] = None,
        approval_granted: bool = False,
        meta: Optional[Dict[str, Any]] = None,
    ) -> ToolExecutionRequest:
        return ToolExecutionRequest(
            tool_name=str(tool_name or "").strip(),
            args=dict(args or {}),
            call_id=str((meta or {}).get("call_id") or uuid.uuid4().hex),
            source_agent=str(source_agent or "").strip(),
            session_id=str(getattr(run_context, "session_id", "") or ""),
            run_id=str(getattr(run_context, "run_id", "") or ""),
            approval_granted=bool(approval_granted),
            meta=dict(meta or {}),
        )

    def execute(
        self,
        request: ToolExecutionRequest,
        *,
        run_context: Optional[RunContext] = None,
    ) -> ToolExecutionReport:
        permission = self._permission_resolver.decide(request, run_context=run_context)
        pre_hooks = runtime_hook_manager.run_pre_tool_hooks(
            run_context,
            tool_name=request.tool_name,
            args=request.args,
            permission=permission.to_dict(),
        )
        hook_payloads = [hook.to_dict() for hook in pre_hooks]

        if permission.decision == "deny":
            return ToolExecutionReport(
                request=request,
                permission=permission,
                ok=False,
                status="denied",
                error=permission.reason,
                hooks=hook_payloads,
                meta={"run_mode": get_run_mode().value},
            )

        if permission.decision == "ask" and not request.approval_granted:
            return ToolExecutionReport(
                request=request,
                permission=permission,
                ok=False,
                status="waiting_approval",
                error=permission.reason,
                hooks=hook_payloads,
                meta={"run_mode": get_run_mode().value},
            )

        started_at = time.perf_counter()
        raw_result = self._executor.execute(request)
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        report = ToolExecutionReport(
            request=request,
            permission=permission,
            ok=bool(raw_result.get("ok")),
            status="completed" if raw_result.get("ok") else "failed",
            result=raw_result.get("result"),
            error=str(raw_result.get("error") or ""),
            duration_ms=duration_ms,
            hooks=hook_payloads,
            meta={
                "tool": raw_result.get("tool") or {},
                "run_mode": get_run_mode().value,
            },
        )
        post_hooks = runtime_hook_manager.run_post_tool_hooks(
            run_context,
            tool_name=request.tool_name,
            report=report.to_dict(),
        )
        report.hooks.extend(hook.to_dict() for hook in post_hooks)
        return report

    def execute_tool(
        self,
        tool_name: str,
        *,
        args: Optional[Dict[str, Any]] = None,
        run_context: Optional[RunContext] = None,
        source_agent: str = "",
        approval_granted: bool = False,
        meta: Optional[Dict[str, Any]] = None,
    ) -> ToolExecutionReport:
        request = self.build_request(
            tool_name,
            args=args,
            source_agent=source_agent,
            run_context=run_context,
            approval_granted=approval_granted,
            meta=meta,
        )
        return self.execute(request, run_context=run_context)

    def capability_snapshot(
        self,
        *,
        run_context: Optional[RunContext] = None,
        dynamic_tool_catalog: Optional[list[dict]] = None,
    ) -> Dict[str, Any]:
        tool_catalog = runtime_tool_registry.build_tool_catalog(dynamic_tool_catalog)
        tool_stats = runtime_tool_registry.build_tool_stats(dynamic_tool_catalog)
        mode_profile = runtime_mode_profile_resolver.resolve(tool_catalog=tool_catalog)
        return {
            "run_mode": get_run_mode().value,
            "tool_catalog": tool_catalog,
            "tool_stats": tool_stats,
            "mode_profile": mode_profile,
            "tool_restriction_enabled": bool(
                runtime_tool_registry.is_tool_restriction_enabled(
                    getattr(run_context, "model_config", {}) if run_context else {}
                )
            ),
        }


runtime_tool_orchestrator = ToolRuntimeOrchestrator()
