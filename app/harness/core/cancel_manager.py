# -*- coding: utf-8 -*-
"""
运行态请求取消管理器。

这是一个核心的运行时能力，用于管理请求的取消状态。
当用户点击停止按钮、客户端断开连接或系统超时时，需要及时取消正在执行的任务。

设计要点：
1. 双通道设计：同时支持threading.Event（同步代码）和asyncio.Event（异步代码）
2. 级联取消：父请求被取消时，所有子请求也会被取消
3. 自动清理：后台线程定期清理过期的取消令牌，避免内存泄漏
4. 线程安全：使用RLock保护共享数据，支持多线程并发访问

使用场景：
- 用户点击停止按钮，立即停止正在生成的AI回复
- 客户端断开连接（如关闭网页），自动取消正在执行的任务
- 系统超时，自动取消长时间运行的任务
- 审批拒绝，取消正在执行的代码或SQL操作

架构说明：
- 本模块是runtime/core的基础能力，提供底层的取消管理
- services/request_cancellation_service.py作为兼容层，对外提供服务接口
- 这种分层设计保证了核心能力的稳定性和扩展性
"""
from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import AsyncIterator, Iterator, Optional

from common.utils.custom_logger import get_logger

log = get_logger(__name__)

# 当前请求ID的上下文变量，用于在异步调用链中传递request_id
# 场景：Agent执行工具调用时，需要知道当前请求的ID来检查取消状态
_CURRENT_REQUEST_ID: ContextVar[str] = ContextVar("current_request_id", default="")

# 过期时间：超过这个时间未使用的取消令牌会被自动清理
# 10分钟的TTL应该足够覆盖绝大多数正常的请求执行时间
_STALE_TTL_SECONDS = 600

# 清理间隔：后台线程每隔2分钟执行一次清理
# 平衡了清理频率和CPU开销
_CLEANUP_INTERVAL_SECONDS = 120


class _CancellationEntry:
    """
    单个取消令牌的内部表示。

    设计理由：
    1. 需要同时支持同步和异步代码的等待
    2. 需要记录注册时间，用于自动清理过期令牌
    3. 使用__slots__优化内存占用，减少对象创建开销

    注意事项：
    - asyncio.Event必须在对应的事件循环中创建
    - 跨线程设置async_event需要使用call_soon_threadsafe
    - thread_event可以在任何线程中安全使用
    """
    __slots__ = ("thread_event", "async_event", "registered_at", "loop")

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """
        初始化取消令牌。

        Args:
            loop: 可选的事件循环对象，用于创建async_event
                 如果为None，会尝试获取当前运行的事件循环
        """
        self.thread_event = threading.Event()  # 同步代码使用的Event
        self.loop = loop  # 记录事件循环，用于跨线程调用
        try:
            # 尝试在当前事件循环中创建async_event
            self.async_event: Optional[asyncio.Event] = asyncio.Event()
        except RuntimeError:
            # 没有运行中的事件循环，先设为None，后续使用时再创建
            self.async_event = None
        self.registered_at = time.monotonic()  # 记录注册时间，用于判断是否过期

    def set(self) -> None:
        """
        设置取消状态，通知所有等待的代码。

        设计要点：
        1. 总是设置thread_event，因为同步代码可能在任何线程中运行
        2. 设置async_event时需要区分同步和异步上下文
        3. 如果事件循环还在运行，使用call_soon_threadsafe保证线程安全
        4. 如果事件循环已经关闭，直接设置可能报错，需要捕获异常

        场景：
        - 用户点击停止按钮，立即设置取消状态
        - 检测到客户端断开，触发取消
        - 超时后自动取消
        """
        # 设置同步Event，立即生效
        self.thread_event.set()

        # 设置异步Event
        if self.async_event is not None:
            if self.loop and self.loop.is_running():
                # 事件循环还在运行，使用线程安全的方式设置
                self.loop.call_soon_threadsafe(self.async_event.set)
            else:
                # 事件循环已关闭，尝试直接设置（可能失败，捕获异常）
                try:
                    self.async_event.set()
                except Exception:
                    pass  # 忽略设置失败，thread_event已经可以工作

    def clear(self) -> None:
        """
        清除取消状态，重置为未取消。

        设计要点：
        1. 同时重置thread_event和async_event
        2. 更新注册时间，重新开始计算TTL
        3. 场景：请求重试、重新执行等

        注意事项：
        - 清除后，该令牌可以再次被使用
        - 注册时间更新，不会被立即清理
        """
        self.thread_event.clear()
        if self.async_event is not None:
            self.async_event.clear()
        self.registered_at = time.monotonic()

    def is_set(self) -> bool:
        """
        检查是否已取消。

        Returns:
            bool: True表示已取消，False表示未取消

        设计理由：
        1. 优先检查thread_event，因为它在所有线程中都可用
        2. 对于同步代码，检查thread_event足够
        3. 对于异步代码，可能需要检查async_event，但这里简化处理

        场景：
        - 在循环中检查是否需要提前退出
        - 在执行长时间操作前检查是否取消
        """
        return self.thread_event.is_set()

    def is_stale(self, ttl: float = _STALE_TTL_SECONDS) -> bool:
        """
        检查是否过期，过期后应该被清理。

        Args:
            ttl: 生存时间（秒），超过这个时间未更新的令牌被认为是过期的

        Returns:
            bool: True表示已过期，False表示还在有效期内

        设计理由：
        1. 使用monotonic时间，不受系统时间调整影响
        2. 根据注册时间计算存活时间
        3. 自动清理机制避免内存泄漏

        场景：
        - 后台清理线程定期检查，清理过期的令牌
        - 避免因为异常退出导致的资源泄漏
        """
        return (time.monotonic() - self.registered_at) > ttl


class RequestCancellationService:
    """
    线程安全的请求取消状态管理器。

    核心职责：
    1. 注册和管理取消令牌
    2. 提供取消查询和触发接口
    3. 支持父子请求的级联取消
    4. 自动清理过期的取消令牌

    数据结构：
    - _entries: request_id -> _CancellationEntry（取消令牌）
    - _children: parent_id -> set(child_id)（子请求集合）
    - _parents: child_id -> set(parent_id)（父请求集合）

    设计要点：
    1. 使用RLock保护所有共享数据的访问
    2. 支持级联取消，父取消时所有子请求也取消
    3. 后台线程自动清理过期令牌
    4. 使用ContextVar在异步调用链中传递request_id
    """

    def __init__(self) -> None:
        """初始化取消服务管理器。"""
        self._lock = threading.RLock()  # 保护共享数据的锁
        self._entries: dict[str, _CancellationEntry] = {}  # 取消令牌字典
        self._children: dict[str, set[str]] = {}  # 父->子关系
        self._parents: dict[str, set[str]] = {}  # 子->父关系

        # 启动后台清理线程
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,  # 守护线程，主进程退出时自动结束
            name="runtime-cancellation-cleanup",
        )
        self._cleanup_thread.start()

    def register_request(self, request_id: str) -> threading.Event:
        """
        注册一个请求的取消令牌。

        设计要点：
        1. 如果request_id已存在，清除旧状态重新使用
        2. 尝试获取当前事件循环，用于创建async_event
        3. 返回thread_event，同步代码可以直接使用

        Args:
            request_id: 请求的唯一标识符

        Returns:
            threading.Event: 线程事件对象，可以等待或检查状态

        Raises:
            ValueError: request_id为空

        场景：
        - HTTP请求进来时注册取消令牌
        - 图执行开始时注册取消令牌
        - Agent执行开始时注册取消令牌
        """
        if not request_id:
            raise ValueError("request_id 不能为空")

        # 尝试获取当前运行的事件循环
        try:
            loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            loop = None  # 没有运行中的事件循环

        with self._lock:
            # 检查是否已存在
            entry = self._entries.get(request_id)
            if entry is None:
                # 创建新的取消令牌
                entry = _CancellationEntry(loop=loop)
                self._entries[request_id] = entry
            else:
                # 清除旧状态，重新使用
                entry.clear()
                if loop:
                    entry.loop = loop
            return entry.thread_event

    def cancel_request(self, request_id: str) -> None:
        """
        取消一个请求及其所有子请求。

        设计要点：
        1. 支持级联取消，父请求被取消时所有子请求也取消
        2. 使用DFS遍历所有相关请求，确保不遗漏
        3. 先收集所有需要取消的ID，再统一设置，避免重复加锁

        Args:
            request_id: 要取消的请求ID

        场景：
        - 用户点击停止按钮
        - 客户端断开连接
        - 系统超时
        """
        if not request_id:
            return

        with self._lock:
            # 收集所有需要取消的请求ID（包括子请求）
            pending_ids: list[str] = []
            stack = [request_id]
            seen: set[str] = set()

            while stack:
                current = stack.pop()
                if not current or current in seen:
                    continue
                seen.add(current)
                pending_ids.append(current)
                # 添加所有子请求
                stack.extend(self._children.get(current, set()))

            # 取出对应的取消令牌
            entries = [(rid, self._entries.get(rid)) for rid in pending_ids]

        # 在锁外执行取消操作，避免持有锁太久
        for rid, entry in entries:
            if entry is not None:
                entry.set()
                log.debug(f"请求已取消: request_id={rid[:16]}...")

    def cleanup_request(self, request_id: str) -> None:
        """
        清理一个请求的取消令牌和父子关系。

        设计要点：
        1. 解除父子关系，避免影响其他请求
        2. 删除取消令牌，释放内存
        3. 正常结束后调用，取消后也应该调用

        Args:
            request_id: 要清理的请求ID

        场景：
        - 请求正常结束后清理
        - 请求被取消后清理
        - 异常退出时清理（在finally中）
        """
        if not request_id:
            return
        with self._lock:
            self._unlink_request_locked(request_id)
            self._entries.pop(request_id, None)

    def link_request(self, parent_request_id: str, child_request_id: str) -> None:
        """
        建立父子请求关系，支持级联取消。

        设计要点：
        1. 建立双向的父子关系，方便后续查询和取消
        2. 如果父请求已经被取消，立即取消子请求
        3. 检查参数有效性，避免建立无效关系

        Args:
            parent_request_id: 父请求ID
            child_request_id: 子请求ID

        场景：
        - Session -> Run：取消Session时也取消Run
        - Run -> Agent：取消Run时也取消所有Agent
        - Agent -> Tool：取消Agent时也取消所有工具调用
        """
        parent_id = str(parent_request_id or "").strip()
        child_id = str(child_request_id or "").strip()
        if not parent_id or not child_id or parent_id == child_id:
            return  # 参数无效，不建立关系

        should_cancel_child = False
        with self._lock:
            # 建立父子关系
            self._children.setdefault(parent_id, set()).add(child_id)
            self._parents.setdefault(child_id, set()).add(parent_id)

            # 检查父请求是否已被取消
            parent_entry = self._entries.get(parent_id)
            should_cancel_child = bool(parent_entry and parent_entry.is_set())

        # 如果父请求已被取消，立即取消子请求
        if should_cancel_child:
            self.cancel_request(child_id)

    def is_cancelled(self, request_id: Optional[str] = None) -> bool:
        """
        检查请求是否已被取消。

        设计要点：
        1. 支持显式传入request_id
        2. 也支持从ContextVar中获取当前的request_id
        3. 如果request_id不存在或未注册，认为未取消

        Args:
            request_id: 可选的请求ID，为空时使用ContextVar中的值

        Returns:
            bool: True表示已取消，False表示未取消

        场景：
        - 在循环中检查是否需要提前退出
        - 在执行长时间操作前检查是否取消
        - 工具调用前检查是否取消
        """
        effective_id = (request_id or _CURRENT_REQUEST_ID.get("")).strip()
        if not effective_id:
            return False
        with self._lock:
            entry = self._entries.get(effective_id)
        return bool(entry and entry.is_set())

    def current_request_id(self) -> str:
        """
        获取当前请求ID（从ContextVar中获取）。

        Returns:
            str: 当前请求ID，如果未设置则返回空字符串

        场景：
        - 在异步调用链中获取当前请求ID
        - 在没有显式传入request_id的函数中使用
        """
        return _CURRENT_REQUEST_ID.get("")

    @contextmanager
    def bind_request(self, request_id: str) -> Iterator[None]:
        """
        绑定当前请求ID到ContextVar，用于异步调用链。

        设计要点：
        1. 使用ContextVar，在异步调用链中自动传递
        2. 支持嵌套调用，外层的bind_request不会影响内层
        3. 使用完后自动重置，避免污染后续调用

        Args:
            request_id: 要绑定的请求ID

        Yields:
            None

        场景：
        - 在Agent执行开始时绑定，所有子工具调用都能获取到request_id
        - 在工具调用时绑定，工具内部的异步代码都能获取到request_id
        """
        token = _CURRENT_REQUEST_ID.set((request_id or "").strip())
        try:
            yield
        finally:
            _CURRENT_REQUEST_ID.reset(token)

    async def wait_for_cancellation(self, request_id: str) -> None:
        """
        等待请求被取消（异步版本）。

        设计要点：
        1. 使用async_event进行异步等待
        2. 如果async_event未创建，会自动创建
        3. 可以配合asyncio.wait_for实现超时等待

        Args:
            request_id: 要等待的请求ID

        场景：
        - 在后台任务中等待取消信号
        - 实现优雅退出逻辑
        """
        with self._lock:
            entry = self._entries.get(request_id)
        if entry is None:
            return  # 请求不存在，直接返回
        if entry.async_event is None:
            # 异步Event还未创建，现在创建
            with self._lock:
                entry2 = self._entries.get(request_id)
                if entry2 and entry2.async_event is None:
                    entry2.async_event = asyncio.Event()
                    # 如果已经被取消，立即设置
                    if entry2.thread_event.is_set():
                        entry2.async_event.set()
                    entry = entry2
        if entry.async_event:
            await entry.async_event.wait()

    async def _poll_disconnect(
            self,
            request_id: str,
            request,
            stop_event: asyncio.Event,
            poll_interval: float,
    ) -> None:
        """
        轮询检测客户端断开连接。

        设计要点：
        1. 使用request.is_disconnected()检测连接状态
        2. 设置超时避免长时间阻塞
        3. 检测到断开时触发取消

        Args:
            request_id: 请求ID
            request: FastAPI Request对象
            stop_event: 停止事件，用于退出循环
            poll_interval: 轮询间隔（秒）

        场景：
        - 客户端关闭网页时自动取消正在执行的任务
        - 网络中断时及时释放资源
        """
        try:
            while not stop_event.is_set():
                try:
                    # 检测客户端是否断开
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
            pass  # 被取消时直接退出

    @asynccontextmanager
    async def cancel_on_disconnect(
            self,
            request_id: str,
            request,
            poll_interval: float = 0.5,
    ) -> AsyncIterator[None]:
        """
        上下文管理器：自动检测客户端断开并取消请求。

        设计要点：
        1. 进入时启动轮询线程
        2. 退出时停止轮询线程
        3. 使用shield保护清理逻辑，确保一定执行

        Args:
            request_id: 请求ID
            request: FastAPI Request对象
            poll_interval: 轮询间隔（秒），默认0.5秒

        Yields:
            None

        场景：
        - 在SSE流式响应中使用
        - 确保客户端断开时及时释放资源
        """
        stop_event = asyncio.Event()
        poll_task = asyncio.create_task(
            self._poll_disconnect(
                request_id=request_id,
                request=request,
                stop_event=stop_event,
                poll_interval=poll_interval,
            )
        )
        try:
            yield
        finally:
            stop_event.set()  # 停止轮询
            poll_task.cancel()  # 取消轮询任务
            try:
                # 使用shield保护清理逻辑，确保一定执行
                await asyncio.wait_for(asyncio.shield(poll_task), timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass  # 忽略清理过程中的异常

    def _cleanup_loop(self) -> None:
        """
        后台清理循环，定期清理过期的取消令牌。

        设计要点：
        1. 每隔一段时间执行一次清理
        2. 清理异常不影响主循环
        3. 守护线程，主进程退出时自动结束

        场景：
        - 自动清理因为异常退出导致的残留令牌
        - 避免内存泄漏
        - 保持内存使用在合理范围
        """
        while True:
            time.sleep(_CLEANUP_INTERVAL_SECONDS)
            try:
                self._evict_stale()
            except Exception as exc:
                log.warning(f"取消服务 stale 清理异常: {exc}")

    def _evict_stale(self) -> None:
        """
        清理过期的取消令牌。

        设计要点：
        1. 查找所有过期的令牌
        2. 解除父子关系
        3. 删除令牌

        场景：
        - 后台线程定期调用
        - 清理因为异常退出导致的残留
        """
        with self._lock:
            # 查找所有过期的令牌
            stale_ids = [
                rid for rid, entry in self._entries.items()
                if entry.is_stale()
            ]
            # 清理过期令牌
            for rid in stale_ids:
                self._unlink_request_locked(rid)
                self._entries.pop(rid, None)
        if stale_ids:
            log.debug(f"取消服务清理 {len(stale_ids)} 条 stale 记录。")

    def active_count(self) -> int:
        """
        获取当前活跃的取消令牌数量。

        Returns:
            int: 活跃的令牌数量

        场景：
        - 监控系统负载
        - 调试和排错
        - 性能分析
        """
        with self._lock:
            return len(self._entries)

    def _unlink_request_locked(self, request_id: str) -> None:
        """
        解除请求的父子关系（需要在持有锁的情况下调用）。

        设计要点：
        1. 从父请求的子集合中删除自己
        2. 从子请求的父集合中删除自己
        3. 如果集合为空，删除整个条目

        Args:
            request_id: 要解除关系的请求ID

        场景：
        - 清理请求时调用
        - 取消请求时可能也需要调用（视需求而定）
        """
        # 从父请求的子集合中删除
        child_ids = self._parents.pop(request_id, set())
        self._id_pop(request_id, child_ids)

        # 从子请求的父集合中删除
        child_ids = self._children.pop(request_id, set())
        self._id_pop(request_id, child_ids)

    def _id_pop(self, request_id: str, ids: set[str]):
        for id in ids:
            parents = self._parents.get(id)
            if not parents:
                continue
            parents.discard(request_id)
            if not parents:
                self._parents.pop(id, None)


# 全局唯一的取消服务管理器实例
# 所有模块都使用这个实例，实现统一的取消管理
runtime_cancel_manager = RequestCancellationService()
