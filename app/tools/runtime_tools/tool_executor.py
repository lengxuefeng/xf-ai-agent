# -*- coding: utf-8 -*-
"""
工具执行器（Tool Executor）。

提供统一的工具执行接口，支持自定义执行器和LangChain工具的自动适配。
这是运行时工具系统的核心执行组件。

设计要点：
1. 统一接口：所有工具通过execute方法执行
2. 灵活扩展：支持自定义执行器注册
3. 自动适配：自动将LangChain工具适配为执行器
4. 错误处理：统一的错误处理和返回格式

使用场景：
- Agent调用工具执行具体操作
- 自定义工具的注册和执行
- LangChain工具的自动适配
- 工具执行的统一管理

返回格式：
成功：{"ok": True, "tool": {...}, "result": ...}
失败：{"ok": False, "tool": {...}, "error": "..."}
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from pydantic import ValidationError

from common.utils.tool_validation import format_tool_validation_error
from tools.runtime_tools.tool_registry import runtime_tool_registry


class ToolExecutor:
    """
    统一工具执行包装器。

    核心职责：
    1. 注册自定义工具执行器
    2. 提供统一的执行接口
    3. 自动适配LangChain工具
    4. 统一的错误处理和返回格式

    设计理由：
    1. 统一接口，简化上层代码
    2. 灵活扩展，支持自定义逻辑
    3. 自动适配，减少重复代码
    4. 错误处理，提高稳定性

    执行流程：
    1. 检查工具是否注册
    2. 查找或创建执行器
    3. 执行工具调用
    4. 处理异常，返回统一格式
    """

    def __init__(self) -> None:
        """初始化工具执行器。"""
        self._handlers: Dict[str, Callable[..., Any]] = {}  # 自定义执行器字典

    def register_handler(self, name: str, handler: Callable[..., Any]) -> None:
        """
        注册自定义工具执行器。

        设计要点：
        1. 允许覆盖默认的执行器
        2. 校验参数有效性

        Args:
            name: 工具名称
            handler: 执行器函数，签名为handler(**kwargs) -> Any

        场景：
        - 为特定工具提供自定义执行逻辑
        - 添加日志、监控等横切逻辑
        - 修改默认的工具行为
        """
        if name and callable(handler):
            self._handlers[str(name).strip()] = handler

    def execute(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        """
        执行工具调用。

        设计要点：
        1. 先查找自定义执行器
        2. 没有则自动适配LangChain工具
        3. 统一的返回格式
        4. 完善的错误处理

        Args:
            name: 工具名称
            **kwargs: 工具参数

        Returns:
            Dict[str, Any]: 执行结果
                - ok: 是否成功
                - tool: 工具信息
                - result: 执行结果（成功时）
                - error: 错误信息（失败时）

        执行流程：
        1. 检查工具是否注册
        2. 查找执行器（自定义或自动适配）
        3. 调用执行器
        4. 处理异常，返回统一格式
        """
        # 获取工具描述
        descriptor = runtime_tool_registry.get_tool(name)
        if descriptor is None:
            return {"ok": False, "error": f"未注册工具: {name}"}

        # 查找执行器
        handler = self._handlers.get(str(name or "").strip())
        tool_ref = runtime_tool_registry.get_langchain_tool(name)
        payload = dict(kwargs)
        args_schema = getattr(tool_ref, "args_schema", None)
        if args_schema is not None:
            try:
                payload = args_schema(**payload).model_dump()
            except ValidationError as exc:
                return {
                    "ok": False,
                    "tool": descriptor.to_dict(),
                    "error": format_tool_validation_error(exc),
                }
        if handler is None:
            # 没有自定义执行器，尝试自动适配LangChain工具
            if tool_ref is None or not hasattr(tool_ref, "invoke"):
                # 无法适配，返回错误
                return {
                    "ok": False,
                    "error": f"工具 {name} 暂未挂接执行器",
                    "tool": descriptor.to_dict(),
                }
            # 自动适配为执行器
            handler = lambda **payload: tool_ref.invoke(payload)

        # 执行工具调用
        try:
            result = handler(**payload)
            return {"ok": True, "tool": descriptor.to_dict(), "result": result}
        except ValidationError as exc:
            return {
                "ok": False,
                "tool": descriptor.to_dict(),
                "error": format_tool_validation_error(exc),
            }
        except Exception as exc:
            # 捕获异常，返回错误
            return {"ok": False, "tool": descriptor.to_dict(), "error": str(exc)}


# 全局唯一的工具执行器实例
tool_executor = ToolExecutor()
