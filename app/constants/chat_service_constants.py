# -*- coding: utf-8 -*-
"""聊天服务层常量。"""

from typing import Dict, Final

from constants.sse_constants import SseContentType


CHAT_SERVICE_ERROR_TIMEOUT: Final[str] = "【系统提示】网络请求超时，请稍后重试。"
CHAT_SERVICE_ERROR_CONNECTION: Final[str] = "【系统提示】模型服务连接断开，请检查网络或重试。"
CHAT_SERVICE_ERROR_RUNTIME_TEMPLATE: Final[str] = "【系统提示】底层服务执行异常: {error}"
CHAT_SERVICE_ERROR_FALLBACK_TEMPLATE: Final[str] = "发生异常: {error}"

CHAT_SERVICE_INTERRUPTED_TEMPLATE: Final[str] = "【系统提示: 生成被中断 ({error})】"
CHAT_SERVICE_INTERRUPTED_APPEND_TEMPLATE: Final[str] = "\n\n> [系统提示: 输出异常中断 ({error})]"

STREAM_HEADERS: Final[Dict[str, str]] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

CHAT_DEFAULT_MODEL_CONFIG: Final[Dict[str, object]] = {
    "model": "glm-4.7",
    "model_service": "zhipu",
    "service_type": "zhipu",
    "deep_thinking_mode": "auto",
    "rag_enabled": False,
    "similarity_threshold": 0.7,
    "embedding_model": "bge-m3:latest",
    "embedding_model_key": "",
    "temperature": 0.2,
    "top_p": 1.0,
    "max_tokens": 2000,
    "model_key": "",
    "model_url": "",
}

CHAT_AI_CONTENT_TYPES = (SseContentType.STREAM, SseContentType.MESSAGE)
