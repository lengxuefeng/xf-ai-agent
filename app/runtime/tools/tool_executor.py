# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Callable, Dict

from runtime.tools.tool_registry import runtime_tool_registry


class ToolExecutor:
    """统一工具执行包装器。"""

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[..., Any]] = {}

    def register_handler(self, name: str, handler: Callable[..., Any]) -> None:
        if name and callable(handler):
            self._handlers[str(name).strip()] = handler

    def execute(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        descriptor = runtime_tool_registry.get_tool(name)
        handler = self._handlers.get(str(name or "").strip())
        if descriptor is None:
            return {"ok": False, "error": f"未注册工具: {name}"}
        if handler is None:
            return {
                "ok": False,
                "error": f"工具 {name} 暂未挂接执行器",
                "tool": descriptor.to_dict(),
            }
        try:
            result = handler(**kwargs)
            return {"ok": True, "tool": descriptor.to_dict(), "result": result}
        except Exception as exc:
            return {"ok": False, "tool": descriptor.to_dict(), "error": str(exc)}


tool_executor = ToolExecutor()

