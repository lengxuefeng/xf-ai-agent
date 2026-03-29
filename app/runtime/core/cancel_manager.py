# -*- coding: utf-8 -*-
"""
运行态请求取消管理器。

说明：
- 保留原有 threading.Event + asyncio.Event 双通道设计；
- 作为 runtime/core 的基础能力，对外由 services/request_cancellation_service.py 兼容导出。
"""
from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import AsyncIterator, Iterator, Optional

from utils.custom_logger import get_logger

log = get_logger(__name__)

_CURRENT_REQUEST_ID: ContextVar[str] = ContextVar("current_request_id", default="")
_STALE_TTL_SECONDS = 600
_CLEANUP_INTERVAL_SECONDS = 120


class _CancellationEntry:
    __slots__ = ("thread_event", "async_event", "registered_at", "loop")

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        self.thread_event = threading.Event()
        self.loop = loop
        try:
            self.async_event: Optional[asyncio.Event] = asyncio.Event()
        except RuntimeError:
            self.async_event = None
        self.registered_at = time.monotonic()

    def set(self) -> None:
        self.thread_event.set()
        if self.async_event is not None:
            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(self.async_event.set)
            else:
                try:
                    self.async_event.set()
                except Exception:
                    pass

    def clear(self) -> None:
        self.thread_event.clear()
        if self.async_event is not None:
            self.async_event.clear()
        self.registered_at = time.monotonic()

    def is_set(self) -> bool:
        return self.thread_event.is_set()

    def is_stale(self, ttl: float = _STALE_TTL_SECONDS) -> bool:
        return (time.monotonic() - self.registered_at) > ttl


class RequestCancellationService:
    """线程安全的请求取消状态管理器。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, _CancellationEntry] = {}
        self._children: dict[str, set[str]] = {}
        self._parents: dict[str, set[str]] = {}
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="runtime-cancellation-cleanup",
        )
        self._cleanup_thread.start()

    def register_request(self, request_id: str) -> threading.Event:
        if not request_id:
            raise ValueError("request_id 不能为空")
        try:
            loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is None:
                entry = _CancellationEntry(loop=loop)
                self._entries[request_id] = entry
            else:
                entry.clear()
                if loop:
                    entry.loop = loop
            return entry.thread_event

    def cancel_request(self, request_id: str) -> None:
        if not request_id:
            return
        with self._lock:
            pending_ids: list[str] = []
            stack = [request_id]
            seen: set[str] = set()
            while stack:
                current = stack.pop()
                if not current or current in seen:
                    continue
                seen.add(current)
                pending_ids.append(current)
                stack.extend(self._children.get(current, set()))
            entries = [(rid, self._entries.get(rid)) for rid in pending_ids]
        for rid, entry in entries:
            if entry is not None:
                entry.set()
                log.debug(f"请求已取消: request_id={rid[:16]}...")

    def cleanup_request(self, request_id: str) -> None:
        if not request_id:
            return
        with self._lock:
            self._unlink_request_locked(request_id)
            self._entries.pop(request_id, None)

    def link_request(self, parent_request_id: str, child_request_id: str) -> None:
        parent_id = str(parent_request_id or "").strip()
        child_id = str(child_request_id or "").strip()
        if not parent_id or not child_id or parent_id == child_id:
            return

        should_cancel_child = False
        with self._lock:
            self._children.setdefault(parent_id, set()).add(child_id)
            self._parents.setdefault(child_id, set()).add(parent_id)
            parent_entry = self._entries.get(parent_id)
            should_cancel_child = bool(parent_entry and parent_entry.is_set())

        if should_cancel_child:
            self.cancel_request(child_id)

    def is_cancelled(self, request_id: Optional[str] = None) -> bool:
        effective_id = (request_id or _CURRENT_REQUEST_ID.get("")).strip()
        if not effective_id:
            return False
        with self._lock:
            entry = self._entries.get(effective_id)
        return bool(entry and entry.is_set())

    def current_request_id(self) -> str:
        return _CURRENT_REQUEST_ID.get("")

    @contextmanager
    def bind_request(self, request_id: str) -> Iterator[None]:
        token = _CURRENT_REQUEST_ID.set((request_id or "").strip())
        try:
            yield
        finally:
            _CURRENT_REQUEST_ID.reset(token)

    async def wait_for_cancellation(self, request_id: str) -> None:
        with self._lock:
            entry = self._entries.get(request_id)
        if entry is None:
            return
        if entry.async_event is None:
            with self._lock:
                entry2 = self._entries.get(request_id)
                if entry2 and entry2.async_event is None:
                    entry2.async_event = asyncio.Event()
                    if entry2.thread_event.is_set():
                        entry2.async_event.set()
                    entry = entry2
        if entry.async_event:
            await entry.async_event.wait()

    @asynccontextmanager
    async def cancel_on_disconnect(
        self,
        request_id: str,
        request,
        poll_interval: float = 0.5,
    ) -> AsyncIterator[None]:
        stop_event = asyncio.Event()

        async def _poll_disconnect() -> None:
            try:
                while not stop_event.is_set():
                    try:
                        disconnected = await asyncio.wait_for(
                            request.is_disconnected(),
                            timeout=poll_interval,
                        )
                    except (asyncio.TimeoutError, Exception):
                        disconnected = False
                    if disconnected:
                        log.info(f"检测到客户端断连，触发取消。request_id={request_id[:16]}")
                        self.cancel_request(request_id)
                        break
                    await asyncio.sleep(poll_interval)
            except asyncio.CancelledError:
                pass

        poll_task = asyncio.create_task(_poll_disconnect())
        try:
            yield
        finally:
            stop_event.set()
            poll_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(poll_task), timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

    def _cleanup_loop(self) -> None:
        while True:
            time.sleep(_CLEANUP_INTERVAL_SECONDS)
            try:
                self._evict_stale()
            except Exception as exc:
                log.warning(f"取消服务 stale 清理异常: {exc}")

    def _evict_stale(self) -> None:
        with self._lock:
            stale_ids = [
                rid for rid, entry in self._entries.items()
                if entry.is_stale()
            ]
            for rid in stale_ids:
                self._unlink_request_locked(rid)
                self._entries.pop(rid, None)
        if stale_ids:
            log.debug(f"取消服务清理 {len(stale_ids)} 条 stale 记录。")

    def active_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def _unlink_request_locked(self, request_id: str) -> None:
        parent_ids = self._parents.pop(request_id, set())
        for parent_id in parent_ids:
            children = self._children.get(parent_id)
            if not children:
                continue
            children.discard(request_id)
            if not children:
                self._children.pop(parent_id, None)

        child_ids = self._children.pop(request_id, set())
        for child_id in child_ids:
            parents = self._parents.get(child_id)
            if not parents:
                continue
            parents.discard(request_id)
            if not parents:
                self._parents.pop(child_id, None)


runtime_cancel_manager = RequestCancellationService()

