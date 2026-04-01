# -*- coding: utf-8 -*-
"""工具参数校验辅助函数。"""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError


def format_tool_validation_error(error: ValidationError | Exception) -> str:
    """把 Pydantic 校验异常格式化为对大模型友好的中文提示。"""
    if isinstance(error, ValidationError):
        lines = ["工具参数校验失败，请按以下要求修正后重试："]
        for item in error.errors():
            loc = ".".join(str(part) for part in item.get("loc") or []) or "unknown"
            msg = str(item.get("msg") or "参数不合法")
            lines.append(f"- 参数 `{loc}`：{msg}")
        return "\n".join(lines)
    return f"工具参数校验失败：{error}"


def handle_tool_validation_error(error: ValidationError | Exception) -> str:
    """LangChain ToolNode 可直接复用的校验失败处理器。"""
    return format_tool_validation_error(error)


def raise_or_format_tool_validation_error(error: Exception) -> str:
    """统一兜底工具执行器中的校验异常文案。"""
    if isinstance(error, ValidationError):
        return format_tool_validation_error(error)
    raise error
