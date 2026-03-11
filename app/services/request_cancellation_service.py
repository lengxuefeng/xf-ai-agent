# -*- coding: utf-8 -*-
"""
请求级取消服务。

用途：
1. 在 SSE 客户端断开时，向后端执行链路广播取消信号；
2. 让工具层在重试/退避前主动检查取消状态，减少无效等待。
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

from utils.custom_logger import get_logger

log = get_logger(__name__)

# 工具层通过 contextvar 读取“当前执行请求”标识。
_CURRENT_REQUEST_ID: ContextVar[str] = ContextVar("current_request_id", default="")


class RequestCancellationService:
    """线程安全的请求取消状态管理器。"""

    def __init__(self):
        self._lock = threading.RLock()
        self._events: dict[str, threading.Event] = {}

    def register_request(self, request_id: str) -> threading.Event:
        """注册请求（或复用已有请求），并清除旧取消标记。"""
        if not request_id:
            raise ValueError("request_id 不能为空")
        with self._lock:
            event = self._events.get(request_id)
            if event is None:
                event = threading.Event()
                self._events[request_id] = event
            else:
                event.clear()
            return event

    def cancel_request(self, request_id: str) -> None:
        """触发请求取消信号。"""
        if not request_id:
            return
        with self._lock:
            event = self._events.get(request_id)
            if event is not None:
                event.set()

    def cleanup_request(self, request_id: str) -> None:
        """清理请求取消状态，避免内存堆积。"""
        if not request_id:
            return
        with self._lock:
            self._events.pop(request_id, None)

    def is_cancelled(self, request_id: Optional[str] = None) -> bool:
        """判断请求是否已取消；request_id 为空时读取当前上下文。"""
        effective_id = (request_id or _CURRENT_REQUEST_ID.get("")).strip()
        if not effective_id:
            return False
        with self._lock:
            event = self._events.get(effective_id)
            return bool(event and event.is_set())

    def current_request_id(self) -> str:
        """返回当前上下文绑定的请求 ID。"""
        return _CURRENT_REQUEST_ID.get("")

    @contextmanager
    def bind_request(self, request_id: str) -> Iterator[None]:
        """
        将请求 ID 绑定到当前执行上下文。

        说明：
        - 绑定后，工具层可通过 is_cancelled() 无参调用读取取消状态；
        - 离开上下文自动恢复，避免跨请求污染。
        """
        token = _CURRENT_REQUEST_ID.set((request_id or "").strip())
        try:
            yield
        finally:
            _CURRENT_REQUEST_ID.reset(token)


request_cancellation_service = RequestCancellationService()

