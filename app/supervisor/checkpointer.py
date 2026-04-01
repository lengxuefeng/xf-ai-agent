# -*- coding: utf-8 -*-
"""LangGraph checkpointer 统一入口。"""
from __future__ import annotations

import asyncio
import inspect
import os
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from common.utils.custom_logger import get_logger
from config.runtime_settings import CHECKPOINTER_POLICY, CHECKPOINTER_POOL_CONFIG

log = get_logger(__name__)

_DURABLE_KINDS = {"supervisor", "sql_agent", "yunyou_agent"}
_CHECKPOINTERS: dict[str, Any] = {}
_CP_LOCK = threading.RLock()
_CHECKPOINTER_POOL: Optional[AsyncConnectionPool] = None


def build_postgres_uri() -> str:
    """读取环境变量并拼接 PostgreSQL 连接串。"""
    pg_user = os.getenv("POSTGRES_USER", "postgres")
    pg_pwd = os.getenv("POSTGRES_PASSWORD", "")
    pg_host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    pg_port = os.getenv("POSTGRES_PORT", "5432")
    pg_db = os.getenv("POSTGRES_DB", "xf_ai_agent")
    return f"postgresql://{pg_user}:{pg_pwd}@{pg_host}:{pg_port}/{pg_db}"


def _safe_backend(backend: str) -> str:
    normalized = str(backend or "").strip().lower()
    if normalized not in {"postgres", "memory"}:
        return "memory"
    return normalized


def _select_backend(kind: str = "default") -> str:
    """根据策略与调用方类型选择后端。"""
    policy = str(CHECKPOINTER_POLICY or "hybrid").strip().lower()
    env_backend = _safe_backend(os.getenv("CHECKPOINTER_BACKEND", "postgres"))
    if policy == "all_memory":
        return "memory"
    if policy == "all_durable":
        return env_backend
    safe_kind = str(kind or "default").strip().lower()
    if safe_kind in _DURABLE_KINDS:
        return env_backend
    return "memory"


def _run_coro_sync(coro):
    """为极少数同步兼容路径执行协程。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_box: dict[str, Any] = {}
    done = threading.Event()

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result_box["value"] = loop.run_until_complete(coro)
        except Exception as exc:  # pragma: no cover - 仅同步兜底路径进入
            result_box["error"] = exc
        finally:
            loop.close()
            done.set()

    worker = threading.Thread(target=_runner, daemon=True, name="checkpointer-sync-bridge")
    worker.start()
    done.wait()
    if "error" in result_box:
        raise result_box["error"]
    return result_box.get("value")


async def initialize_checkpointer_pool(app: Any | None = None) -> Optional[AsyncConnectionPool]:
    """初始化全局异步连接池，并确保 checkpoint 表结构只校验一次。"""
    global _CHECKPOINTER_POOL

    backend = _safe_backend(os.getenv("CHECKPOINTER_BACKEND", "postgres"))
    if backend != "postgres":
        log.info("Checkpointer: 当前后端非 PostgreSQL，跳过异步连接池初始化。")
        if app is not None:
            app.state.checkpointer_pool = None
        return None

    if _CHECKPOINTER_POOL is not None:
        if app is not None:
            app.state.checkpointer_pool = _CHECKPOINTER_POOL
        return _CHECKPOINTER_POOL

    pool = AsyncConnectionPool(
        conninfo=build_postgres_uri(),
        min_size=CHECKPOINTER_POOL_CONFIG.min_size,
        max_size=CHECKPOINTER_POOL_CONFIG.max_size,
        timeout=CHECKPOINTER_POOL_CONFIG.timeout_seconds,
        max_idle=CHECKPOINTER_POOL_CONFIG.max_idle_seconds,
        max_lifetime=CHECKPOINTER_POOL_CONFIG.max_lifetime_seconds,
        open=False,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
        },
    )
    try:
        await pool.open()
        async with pool.connection(timeout=CHECKPOINTER_POOL_CONFIG.timeout_seconds) as conn:
            saver = AsyncPostgresSaver(conn=conn)
            await saver.setup()
        _CHECKPOINTER_POOL = pool
        if app is not None:
            app.state.checkpointer_pool = pool
        log.info("Checkpointer: 全局异步连接池已初始化。")
        return pool
    except Exception:
        await pool.close()
        raise


async def close_checkpointer_pool() -> None:
    """关闭全局异步连接池。"""
    global _CHECKPOINTER_POOL
    if _CHECKPOINTER_POOL is None:
        return
    await _CHECKPOINTER_POOL.close()
    _CHECKPOINTER_POOL = None
    log.info("Checkpointer: 全局异步连接池已关闭。")


def get_checkpointer_pool() -> AsyncConnectionPool:
    """获取已初始化的全局异步连接池。"""
    if _CHECKPOINTER_POOL is None:
        raise RuntimeError("Checkpointer 连接池尚未初始化，请先执行 FastAPI lifespan。")
    return _CHECKPOINTER_POOL


class PoolBackedCheckpointer(BaseCheckpointSaver[Any]):
    """基于全局 AsyncConnectionPool 的异步 checkpointer 代理。"""

    def __init__(self, backend: str) -> None:
        super().__init__()
        self._backend = _safe_backend(backend)
        self._memory_saver = InMemorySaver()
        # 统一复用默认序列化器，保持与 LangGraph 默认行为一致。
        self.serde = self._memory_saver.serde

    @property
    def config_specs(self) -> list:
        return getattr(self._memory_saver, "config_specs", [])

    async def _with_postgres_saver(self, method_name: str, *args, **kwargs):
        pool = get_checkpointer_pool()
        async with pool.connection(timeout=CHECKPOINTER_POOL_CONFIG.timeout_seconds) as conn:
            saver = AsyncPostgresSaver(conn=conn, serde=self.serde)
            method = getattr(saver, method_name)
            result = method(*args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        if self._backend == "memory":
            return await self._memory_saver.aget_tuple(config)
        return await self._with_postgres_saver("aget_tuple", config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        if self._backend == "memory":
            async for item in self._memory_saver.alist(config, filter=filter, before=before, limit=limit):
                yield item
            return

        pool = get_checkpointer_pool()
        async with pool.connection(timeout=CHECKPOINTER_POOL_CONFIG.timeout_seconds) as conn:
            saver = AsyncPostgresSaver(conn=conn, serde=self.serde)
            async for item in saver.alist(config, filter=filter, before=before, limit=limit):
                yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        if self._backend == "memory":
            return await self._memory_saver.aput(config, checkpoint, metadata, new_versions)
        return await self._with_postgres_saver("aput", config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        if self._backend == "memory":
            await self._memory_saver.aput_writes(config, writes, task_id, task_path)
            return
        await self._with_postgres_saver("aput_writes", config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        if self._backend == "memory":
            await self._memory_saver.adelete_thread(thread_id)
            return
        await self._with_postgres_saver("adelete_thread", thread_id)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        if self._backend == "memory":
            return self._memory_saver.get_tuple(config)
        return _run_coro_sync(self.aget_tuple(config))

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        if self._backend == "memory":
            return self._memory_saver.list(config, filter=filter, before=before, limit=limit)
        return iter(_run_coro_sync(self._collect_list(config, filter=filter, before=before, limit=limit)))

    async def _collect_list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> list[CheckpointTuple]:
        result: list[CheckpointTuple] = []
        async for item in self.alist(config, filter=filter, before=before, limit=limit):
            result.append(item)
        return result

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        if self._backend == "memory":
            return self._memory_saver.put(config, checkpoint, metadata, new_versions)
        return _run_coro_sync(self.aput(config, checkpoint, metadata, new_versions))

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        if self._backend == "memory":
            self._memory_saver.put_writes(config, writes, task_id, task_path)
            return
        _run_coro_sync(self.aput_writes(config, writes, task_id, task_path))

    def delete_thread(self, thread_id: str) -> None:
        if self._backend == "memory":
            self._memory_saver.delete_thread(thread_id)
            return
        _run_coro_sync(self.adelete_thread(thread_id))

    def get_next_version(self, current: Any, channel: None) -> Any:
        return self._memory_saver.get_next_version(current, channel)

    def close(self) -> None:
        """实例级别无需关闭，连接池由 lifespan 统一管理。"""
        return None


def _create_checkpointer(backend: str) -> Any:
    safe_backend = _safe_backend(backend)
    if safe_backend == "memory":
        return PoolBackedCheckpointer("memory")
    return PoolBackedCheckpointer("postgres")


def get_checkpointer(kind: str = "default") -> Any:
    """获取按策略路由后的 checkpointer（单例缓存）。"""
    backend = _select_backend(kind)
    with _CP_LOCK:
        cp = _CHECKPOINTERS.get(backend)
        if cp is None:
            cp = _create_checkpointer(backend)
            _CHECKPOINTERS[backend] = cp
        return cp


checkpointer = get_checkpointer("default")

__all__ = [
    "PoolBackedCheckpointer",
    "build_postgres_uri",
    "checkpointer",
    "close_checkpointer_pool",
    "get_checkpointer",
    "get_checkpointer_pool",
    "initialize_checkpointer_pool",
]
