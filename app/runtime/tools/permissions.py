# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional

from config.constants.runtime_hook_keywords import TOOL_GUARD_SUSPICIOUS_KEYWORDS
from config.runtime_settings import get_run_mode
from harness.types import RunContext
from runtime.tools.models import ToolExecutionRequest, ToolPermissionDecision
from tools.runtime_tools.tool_registry import ToolDescriptor, ToolType, runtime_tool_registry


class ToolPermissionResolver:
    """统一工具权限裁决器。"""

    def decide(
        self,
        request: ToolExecutionRequest,
        *,
        run_context: Optional[RunContext] = None,
    ) -> ToolPermissionDecision:
        descriptor = runtime_tool_registry.get_tool(request.tool_name)
        if descriptor is None:
            return ToolPermissionDecision(
                tool_name=request.tool_name,
                decision="deny",
                reason=f"未注册工具: {request.tool_name}",
                allowed=False,
                meta={"reason": "not_registered"},
            )

        if not runtime_tool_registry.is_tool_allowed_in_current_mode(descriptor):
            return ToolPermissionDecision(
                tool_name=descriptor.name,
                decision="deny",
                reason=f"当前 RUN_MODE={get_run_mode().value} 不允许工具 `{descriptor.name}`。",
                allowed=False,
                meta={"reason": "run_mode_denied", "allowed_modes": list(descriptor.allowed_modes)},
            )

        llm_config = dict(getattr(run_context, "model_config", {}) or {})
        restriction_decision = self._check_tool_restriction(descriptor, llm_config=llm_config)
        if restriction_decision is not None:
            return restriction_decision

        high_risk = self._is_high_risk_request(run_context)
        if descriptor.requires_approval and not request.approval_granted:
            return ToolPermissionDecision(
                tool_name=descriptor.name,
                decision="ask",
                reason=f"工具 `{descriptor.name}` 需要人工审批后才能执行。",
                requires_approval=True,
                allowed=False,
                meta={"reason": "requires_approval"},
            )

        if high_risk and descriptor.category in {"exec", "workspace", "database"} and not request.approval_granted:
            return ToolPermissionDecision(
                tool_name=descriptor.name,
                decision="ask",
                reason="检测到高风险操作语义，当前工具调用需要人工审批。",
                requires_approval=True,
                allowed=False,
                meta={"reason": "high_risk_guard"},
            )

        return ToolPermissionDecision(
            tool_name=descriptor.name,
            decision="allow",
            reason="工具权限校验通过。",
            requires_approval=bool(descriptor.requires_approval),
            allowed=True,
            meta={
                "reason": "allowed",
                "tool_type": descriptor.tool_type.value,
                "category": descriptor.category,
            },
        )

    @staticmethod
    def _check_tool_restriction(
        descriptor: ToolDescriptor,
        *,
        llm_config: Dict[str, Any],
    ) -> ToolPermissionDecision | None:
        if not runtime_tool_registry.is_tool_restriction_enabled(llm_config):
            return None
        if descriptor.tool_type in {ToolType.MCP, ToolType.SKILL}:
            return None

        allowed_builtin_tools = runtime_tool_registry.get_allowed_builtin_tools(llm_config)
        resolved_bound_tools = {
            str(item or "").strip()
            for item in (llm_config.get("resolved_bound_tools") or [])
            if str(item or "").strip()
        }
        normalized = runtime_tool_registry.normalize_bound_tools([descriptor.name])
        canonical = normalized[0] if normalized else descriptor.name
        if canonical in allowed_builtin_tools or descriptor.name in resolved_bound_tools:
            return None

        return ToolPermissionDecision(
            tool_name=descriptor.name,
            decision="deny",
            reason=f"当前会话未授权使用工具 `{descriptor.name}`。",
            allowed=False,
            meta={
                "reason": "tool_restricted",
                "allowed_builtin_tools": sorted(allowed_builtin_tools),
                "resolved_bound_tools": sorted(resolved_bound_tools),
            },
        )

    @staticmethod
    def _is_high_risk_request(run_context: Optional[RunContext]) -> bool:
        if run_context is None:
            return False
        user_text = str(getattr(run_context, "user_input", "") or "").lower()
        return any(keyword in user_text for keyword in TOOL_GUARD_SUSPICIOUS_KEYWORDS)


tool_permission_resolver = ToolPermissionResolver()

