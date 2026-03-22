# -*- coding: utf-8 -*-
"""
请求级取消服务（优化版）。

改进点：
1. 双通道取消信号：threading.Event（同步工具层）+ asyncio.Event（异步协程层）并存，
   cancel_request 同时设置两者，同步/异步均可无阻塞感知。
2. TTL 自动清理：注册时记录时间戳，后台定时任务清理过期 stale 记录，
   防止长时间运行的服务内存无限增长。
3. cancel_on_disconnect：异步上下文管理器，在后台轮询 FastAPI Request.is_disconnected()，
   客户端断开时自动触发取消，解决 StreamingResponse 仅在生成器退出时才感知断连的问题。
4. 保持向后兼容：所有现有调用接口不变。
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

# 工具层通过 contextvar 读取"当前执行请求"标识。
_CURRENT_REQUEST_ID: ContextVar[str] = ContextVar("current_request_id", default="")

# 过期 stale 记录的 TTL（秒），超过此时间未清理的注册记录将被后台任务回收。
_STALE_TTL_SECONDS = 600  # 10 分钟
# 后台清理任务的检查间隔（秒）
_CLEANUP_INTERVAL_SECONDS = 120


class _CancellationEntry:
    """单个请求的取消状态载体，包含同步和异步两个事件对象。"""

    __slots__ = ("thread_event", "async_event", "registered_at", "loop")

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        self.thread_event = threading.Event()
        self.loop = loop
        # asyncio.Event 必须在有事件循环的线程中创建
        try:
            self.async_event: Optional[asyncio.Event] = asyncio.Event()
        except RuntimeError:
            # 在没有事件循环的线程中创建，延迟初始化
            self.async_event = None
        self.registered_at = time.monotonic()

    def set(self) -> None:
        """同时触发同步和异步取消信号。"""
        self.thread_event.set()
        if self.async_event is not None:
            # asyncio.Event.set() 必须在事件循环所在线程调用
            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(self.async_event.set)
            else:
                try:
                    self.async_event.set()
                except Exception:
                    pass

    def clear(self) -> None:
        """重置取消信号（复用请求 ID 时调用）。"""
        self.thread_event.clear()
        if self.async_event is not None:
            self.async_event.clear()
        self.registered_at = time.monotonic()

    def is_set(self) -> bool:
        return self.thread_event.is_set()

    def is_stale(self, ttl: float = _STALE_TTL_SECONDS) -> bool:
        """判断注册记录是否超过 TTL（已完成的请求未及时清理时触发）。"""
        return (time.monotonic() - self.registered_at) > ttl


class RequestCancellationService:
    """线程安全的请求取消状态管理器（优化版）。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, _CancellationEntry] = {}
        # 父请求（如 session_id）到子请求（如 run_id）的取消联动关系。
        self._children: dict[str, set[str]] = {}
        self._parents: dict[str, set[str]] = {}
        # 启动后台 stale 清理线程
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="cancellation-cleanup",
        )
        self._cleanup_thread.start()

    # ------------------------------------------------------------------ #
    #  核心接口（向后兼容）                                                  #
    # ------------------------------------------------------------------ #

    def register_request(self, request_id: str) -> threading.Event:
        """注册请求（或复用已有请求），并清除旧取消标记。"""
        if not request_id:
            raise ValueError("request_id 不能为空")
        # 尝试获取当前运行的事件循环（供 asyncio.Event 跨线程 set 使用）
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
        """触发请求取消信号（同时通知同步和异步等待者）。"""
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
            entries = [
                (rid, self._entries.get(rid))
                for rid in pending_ids
            ]
        for rid, entry in entries:
            if entry is not None:
                entry.set()
                log.debug(f"请求已取消: request_id={rid[:16]}...")

    def cleanup_request(self, request_id: str) -> None:
        """清理请求取消状态，避免内存堆积。"""
        if not request_id:
            return
        with self._lock:
            self._unlink_request_locked(request_id)
            self._entries.pop(request_id, None)

    def link_request(self, parent_request_id: str, child_request_id: str) -> None:
        """
        建立“父请求 -> 子请求”的取消联动关系。

        典型场景：
        - 路由层只知道 `session_id`
        - GraphRunner 内部真正执行的是 `run_id`
        当 session 被取消时，应级联取消当前 run。
        """
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
        """判断请求是否已取消；request_id 为空时读取当前上下文。"""
        effective_id = (request_id or _CURRENT_REQUEST_ID.get("")).strip()
        if not effective_id:
            return False
        with self._lock:
            entry = self._entries.get(effective_id)
        return bool(entry and entry.is_set())

    def current_request_id(self) -> str:
        """返回当前上下文绑定的请求 ID。"""
        return _CURRENT_REQUEST_ID.get("")

    @contextmanager
    def bind_request(self, request_id: str) -> Iterator[None]:
        """
        将请求 ID 绑定到当前执行上下文（同步版）。

        说明：
        - 绑定后，工具层可通过 is_cancelled() 无参调用读取取消状态；
        - 离开上下文自动恢复，避免跨请求污染。
        """
        token = _CURRENT_REQUEST_ID.set((request_id or "").strip())
        try:
            yield
        finally:
            _CURRENT_REQUEST_ID.reset(token)

    # ------------------------------------------------------------------ #
    #  新增：异步接口                                                        #
    # ------------------------------------------------------------------ #

    async def wait_for_cancellation(self, request_id: str) -> None:
        """
        异步等待指定请求被取消。

        协程会挂起直到 cancel_request(request_id) 被调用，
        适合在 asyncio 任务中监听取消信号。
        """
        with self._lock:
            entry = self._entries.get(request_id)
        if entry is None:
            return
        if entry.async_event is None:
            # 延迟创建 asyncio.Event（首次在有事件循环的协程中被调用时）
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
        """
        异步上下文管理器：后台轮询 FastAPI Request.is_disconnected()，
        客户端断开时自动触发 cancel_request。

        Args:
            request_id:    当前请求的 run_id。
            request:       FastAPI Request 对象。
            poll_interval: 轮询间隔（秒），默认 0.5s。
        """
        stop_event = asyncio.Event()

        async def _poll_disconnect() -> None:
            """后台任务：持续轮询客户端断连状态。"""
            try:
                while not stop_event.is_set():
                    try:
                        disconnected = await asyncio.wait_for(
                            request.is_disconnected(), timeout=poll_interval
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

    # ------------------------------------------------------------------ #
    #  后台 stale 清理                                                       #
    # ------------------------------------------------------------------ #

    def _cleanup_loop(self) -> None:
        """后台线程：定期清理超过 TTL 的 stale 记录。"""
        while True:
            time.sleep(_CLEANUP_INTERVAL_SECONDS)
            try:
                self._evict_stale()
            except Exception as exc:
                log.warning(f"取消服务 stale 清理异常: {exc}")

    def _evict_stale(self) -> None:
        """清理所有超过 TTL 的 stale 取消记录。"""
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
        """返回当前注册的请求数量（含已取消未清理的）。"""
        with self._lock:
            return len(self._entries)

    def _unlink_request_locked(self, request_id: str) -> None:
        """移除请求与其父子联动关系。调用方需已持有锁。"""
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


request_cancellation_service = RequestCancellationService()
