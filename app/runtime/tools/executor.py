# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict

from runtime.tools.models import ToolExecutionRequest
from tools.runtime_tools.tool_executor import tool_executor


class RuntimeToolExecutor:
    """统一 Runtime Tool 执行适配层。"""

    def execute(self, request: ToolExecutionRequest) -> Dict[str, Any]:
        return tool_executor.execute(request.tool_name, **dict(request.args or {}))


runtime_tool_executor = RuntimeToolExecutor()

