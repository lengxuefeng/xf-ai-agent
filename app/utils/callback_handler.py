# -*- coding: utf-8 -*-
"""
自定义回调处理器，用于捕获 LangChain 执行过程中的事件
"""
import json
import logging
from typing import Any, Dict, Optional, Union
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult, ChatGenerationChunk
from langchain_core.messages import BaseMessage

from constants.sse_constants import SseEventType


class LoggingCallbackHandler(BaseCallbackHandler):
    """日志回调处理器，将 LangChain 事件转换为 SSE 格式"""

    def __init__(self, output_callback=None, include_thinking: bool = True):
        """
        初始化回调处理器

        Args:
            output_callback: 输出回调函数，接收 SSE 格式数据
            include_thinking: 是否包含思考过程
        """
        super().__init__()
        self.output_callback = output_callback
        self.include_thinking = include_thinking
        self.logger = logging.getLogger(__name__)

    def _emit(self, data: dict):
        """发送 SSE 格式数据"""
        if self.output_callback:
            self.output_callback(f"data: {json.dumps(data, ensure_ascii=False)}\n\n")

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        """LLM 调用开始"""
        if self.include_thinking:
            self._emit({
                "type": SseEventType.THINKING.value,
                "content": f"🤖 模型调用中..."
            })

    def on_llm_new_token(
        self, token: str, **kwargs: Any
    ) -> None:
        """LLM 生成新 token"""
        # 对于流式输出，这里可以处理
        pass

    def on_llm_end(
        self, response: LLMResult, **kwargs: Any
    ) -> None:
        """LLM 调用结束"""
        if self.include_thinking and response.llm_output:
            # 检查是否有 reasoning_content（某些模型支持）
            if hasattr(response, 'llm_output') and response.llm_output:
                reasoning_content = response.llm_output.get('reasoning_content', '')
                if reasoning_content:
                    self._emit({
                        "type": SseEventType.THINKING.value,
                        "content": f"💭 思考: {reasoning_content}"
                    })

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Chain 调用开始"""
        if self.include_thinking:
            chain_name = serialized.get('name', 'unknown')
            self._emit({
                "type": SseEventType.THINKING.value,
                "content": f"🔗 执行: {chain_name}"
            })

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Chain 调用结束"""
        if self.include_thinking:
            self._emit({
                "type": SseEventType.THINKING.value,
                "content": "✅ Chain 执行完成"
            })

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """工具调用开始"""
        if self.include_thinking:
            tool_name = serialized.get('name', 'unknown')
            self._emit({
                "type": SseEventType.THINKING.value,
                "content": f"🛠️ 调用工具: {tool_name}"
            })

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """工具调用结束"""
        if self.include_thinking:
            self._emit({
                "type": SseEventType.THINKING.value,
                "content": "✅ 工具调用完成"
            })


class LoggerHandler(logging.Handler):
    """自定义日志处理器，将日志输出转换为 SSE 格式"""

    def __init__(self, output_callback=None, level=logging.INFO):
        """
        初始化日志处理器

        Args:
            output_callback: 输出回调函数，接收 SSE 格式数据
            level: 日志级别
        """
        super().__init__(level=level)
        self.output_callback = output_callback
        self.filters = []

        # 添加过滤器，只捕获特定模式的日志
        # 例如：agent.graphs.supervisor, agent.graph_runner 等
        self.add_filter(lambda record: 'agent' in record.name.lower())

    def emit(self, record: logging.LogRecord) -> None:
        """发送日志"""
        try:
            if self.output_callback:
                # 格式化日志
                log_message = self.format(record)

                # 提取日志类型
                log_type = "info"
                if record.levelno >= logging.ERROR:
                    log_type = "error"
                elif record.levelno >= logging.WARNING:
                    log_type = "warning"

                payload = {
                    "type": SseEventType.LOG.value,
                    "log_type": log_type,
                    "logger": record.name,
                    "message": log_message,
                }
                self.output_callback(
                    f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                )
        except Exception:
            self.handleError(record)
