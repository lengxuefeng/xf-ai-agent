# -*- coding: utf-8 -*-
# app/agent/graphs/checkpointer.py
import atexit
import functools
import inspect
import os
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Callable

from config.runtime_settings import CHECKPOINTER_POLICY
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver
from utils.custom_logger import get_logger

log = get_logger(__name__)

"""
LangGraph 记忆核心（Checkpointer）。

说明：
1. 统一管理 Postgres / Memory 后端。
2. 对 Postgres 增加“连接关闭自动重连”，避免运行中断。
3. 保留单例导出，兼容现有 compile(checkpointer=...) 调用方式。
"""


def _build_postgres_uri() -> str:
    """读取环境变量并拼接 Postgres 连接串。"""
    pg_user = os.getenv("POSTGRES_USER", "postgres")
    pg_pwd = os.getenv("POSTGRES_PASSWORD", "")
    pg_host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    pg_port = os.getenv("POSTGRES_PORT", "5432")
    pg_db = os.getenv("POSTGRES_DB", "xf_ai_agent")
    return f"postgresql://{pg_user}:{pg_pwd}@{pg_host}:{pg_port}/{pg_db}"


class ResilientCheckpointer(BaseCheckpointSaver[Any]):
    """
    可自愈的 Checkpointer 代理。

    设计目标：
    - 外部仍然把它当成普通 checkpointer 使用；
    - 当底层抛出连接关闭错误时，自动重建 PostgresSaver 后重试一次；
    - 避免“the connection is closed”直接打断业务链路。
    """

    # 连接失效的典型错误关键字
    _CLOSED_ERROR_KEYWORDS = (
        "connection is closed",
        "closed the connection",
        "connection not open",
        "terminating connection",
        "server closed the connection",
    )

    def __init__(self, backend: str, postgres_uri: str = ""):
        """初始化代理并创建底层 saver。"""
        super().__init__()
        self._backend = (backend or "memory").strip().lower()
        self._postgres_uri = postgres_uri
        self._lock = threading.RLock()
        self._context_manager: Any = None
        self._saver: Any = None
        self._init_backend()

    @classmethod
    def _is_closed_connection_error(cls, exc: Exception) -> bool:
        """判断异常是否属于连接已关闭场景。"""
        msg = str(exc or "").lower()
        return any(k in msg for k in cls._CLOSED_ERROR_KEYWORDS)

    def _init_backend(self):
        """按后端类型创建底层 saver。"""
        if self._backend == "memory":
            self._saver = InMemorySaver()
            self._context_manager = None
            self.serde = self._saver.serde
            log.info("Checkpointer: 使用内存存储 (InMemorySaver)")
            return

        if self._backend != "postgres":
            log.warning(f"Checkpointer: 未识别后端 [{self._backend}]，降级为内存存储。")
            self._saver = InMemorySaver()
            self._context_manager = None
            self.serde = self._saver.serde
            return

        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError:
            log.error("缺失依赖: uv add langgraph-checkpoint-postgres psycopg[binary]")
            raise

        # 通过 context manager 获取 saver，并保留 manager 以便后续优雅关闭
        context_manager = PostgresSaver.from_conn_string(self._postgres_uri)
        saver = context_manager.__enter__()
        saver.setup()
        self._context_manager = context_manager
        self._saver = saver
        # 复用底层序列化器，保持与原生 saver 一致
        self.serde = saver.serde
        log.info("Checkpointer: PostgreSQL 状态表校验成功，连接已建立。")

    def _reconnect_postgres_locked(self):
        """在持锁状态下重建 PostgresSaver 连接。"""
        if self._backend != "postgres":
            return

        # 先尝试关闭旧连接，避免连接泄漏
        if self._context_manager is not None:
            try:
                self._context_manager.__exit__(None, None, None)
            except Exception as close_exc:
                log.warning(f"Checkpointer: 关闭旧连接时出现告警: {close_exc}")

        # 重新初始化
        self._init_backend()
        log.warning("Checkpointer: 检测到连接失效，已自动重建数据库连接。")

    def _call_with_retry(self, method_name: str, *args, **kwargs):
        """调用底层方法；若遇到连接关闭错误则重连并重试一次。"""
        with self._lock:
            saver = self._saver
        method: Callable[..., Any] = getattr(saver, method_name)

        try:
            return method(*args, **kwargs)
        except Exception as exc:
            if self._backend != "postgres" or not self._is_closed_connection_error(exc):
                raise
            log.warning(f"Checkpointer: 方法 {method_name} 首次调用失败（连接已关闭），准备自动重连。")
            with self._lock:
                self._reconnect_postgres_locked()
                retry_method: Callable[..., Any] = getattr(self._saver, method_name)
            return retry_method(*args, **kwargs)

    async def _acall_with_retry(self, method_name: str, *args, **kwargs):
        """异步调用底层方法；连接关闭时自动重连后重试一次。"""
        with self._lock:
            saver = self._saver
        method: Callable[..., Any] = getattr(saver, method_name)

        try:
            result = method(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result
        except Exception as exc:
            if self._backend != "postgres" or not self._is_closed_connection_error(exc):
                raise
            log.warning(f"Checkpointer: 异步方法 {method_name} 首次调用失败（连接已关闭），准备自动重连。")
            with self._lock:
                self._reconnect_postgres_locked()
                retry_method: Callable[..., Any] = getattr(self._saver, method_name)
            retry_result = retry_method(*args, **kwargs)
            return await retry_result if inspect.isawaitable(retry_result) else retry_result

    def close(self):
        """释放底层连接资源。"""
        with self._lock:
            if self._context_manager is not None:
                try:
                    self._context_manager.__exit__(None, None, None)
                except Exception as exc:
                    log.warning(f"Checkpointer: 关闭连接失败: {exc}")
                finally:
                    self._context_manager = None

    @property
    def config_specs(self) -> list:
        """透传底层 saver 的配置规格。"""
        with self._lock:
            saver = self._saver
        return getattr(saver, "config_specs", [])

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """同步读取单条 checkpoint。"""
        return self._call_with_retry("get_tuple", config)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """同步列出 checkpoint。"""
        return self._call_with_retry(
            "list",
            config,
            filter=filter,
            before=before,
            limit=limit,
        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """同步写入 checkpoint。"""
        return self._call_with_retry(
            "put",
            config,
            checkpoint,
            metadata,
            new_versions,
        )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """同步写入中间写集。"""
        self._call_with_retry(
            "put_writes",
            config,
            writes,
            task_id,
            task_path,
        )

    def delete_thread(self, thread_id: str) -> None:
        """同步删除线程下所有 checkpoint。"""
        self._call_with_retry("delete_thread", thread_id)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """异步读取单条 checkpoint。"""
        with self._lock:
            saver = self._saver
        if hasattr(saver, "aget_tuple"):
            return await self._acall_with_retry("aget_tuple", config)
        return self._call_with_retry("get_tuple", config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """异步列出 checkpoint。"""
        with self._lock:
            saver = self._saver
        if hasattr(saver, "alist"):
            iterator = await self._acall_with_retry(
                "alist",
                config,
                filter=filter,
                before=before,
                limit=limit,
            )
            async for item in iterator:
                yield item
            return

        for item in self._call_with_retry(
            "list",
            config,
            filter=filter,
            before=before,
            limit=limit,
        ):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """异步写入 checkpoint。"""
        with self._lock:
            saver = self._saver
        if hasattr(saver, "aput"):
            return await self._acall_with_retry(
                "aput",
                config,
                checkpoint,
                metadata,
                new_versions,
            )
        return self._call_with_retry(
            "put",
            config,
            checkpoint,
            metadata,
            new_versions,
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """异步写入中间写集。"""
        with self._lock:
            saver = self._saver
        if hasattr(saver, "aput_writes"):
            await self._acall_with_retry(
                "aput_writes",
                config,
                writes,
                task_id,
                task_path,
            )
            return
        self._call_with_retry(
            "put_writes",
            config,
            writes,
            task_id,
            task_path,
        )

    async def adelete_thread(self, thread_id: str) -> None:
        """异步删除线程下所有 checkpoint。"""
        with self._lock:
            saver = self._saver
        if hasattr(saver, "adelete_thread"):
            await self._acall_with_retry("adelete_thread", thread_id)
            return
        self._call_with_retry("delete_thread", thread_id)

    def get_next_version(self, current: Any, channel: None) -> Any:
        """透传底层版本号生成策略，保证版本递增规则一致。"""
        return self._call_with_retry("get_next_version", current, channel)

    def __getattr__(self, name: str) -> Any:
        """
        代理到底层 saver。

        对可调用对象统一包裹重试逻辑；非函数属性直接透传。
        """
        with self._lock:
            saver = self._saver
        attr = getattr(saver, name)
        if not callable(attr):
            return attr
        wrapped = functools.partial(self._call_with_retry, name)
        functools.update_wrapper(wrapped, attr)
        return wrapped


_DURABLE_KINDS = {"supervisor", "sql_agent", "yunyou_agent"}
_CHECKPOINTERS: dict[str, Any] = {}
_REGISTERED_FOR_CLOSE: set[int] = set()
_CP_LOCK = threading.RLock()


def _safe_backend(backend: str) -> str:
    normalized = str(backend or "").strip().lower()
    if normalized not in {"postgres", "memory"}:
        return "memory"
    return normalized


def _select_backend(kind: str = "default") -> str:
    """
    根据策略与调用方类型选择后端。

    策略:
    - all_durable: 全部走 CHECKPOINTER_BACKEND
    - all_memory: 全部走 memory
    - hybrid(默认): supervisor/sql_agent/yunyou_agent 走 CHECKPOINTER_BACKEND，其他走 memory
    """
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


def _create_checkpointer(backend: str) -> Any:
    """按后端创建 checkpointer 实例。"""
    safe_backend = _safe_backend(backend)
    postgres_uri = _build_postgres_uri()
    try:
        cp = ResilientCheckpointer(backend=safe_backend, postgres_uri=postgres_uri)
    except Exception as exc:
        # 生产兜底：若 Postgres 暂不可用，不让服务整体启动失败，降级到内存模式保障可用性。
        if safe_backend == "postgres":
            log.warning(f"Checkpointer: PostgreSQL 初始化失败，已降级为内存模式。原因: {exc}")
            cp = ResilientCheckpointer(backend="memory", postgres_uri=postgres_uri)
        else:
            raise
    return cp


def get_checkpointer(kind: str = "default") -> Any:
    """获取按策略路由后的 checkpointer（单例缓存）。"""
    backend = _select_backend(kind)
    with _CP_LOCK:
        cp = _CHECKPOINTERS.get(backend)
        if cp is None:
            cp = _create_checkpointer(backend)
            _CHECKPOINTERS[backend] = cp
        cp_id = id(cp)
        if cp_id not in _REGISTERED_FOR_CLOSE:
            atexit.register(cp.close)
            _REGISTERED_FOR_CLOSE.add(cp_id)
        return cp


# 兼容旧调用：默认导出仍可直接使用
checkpointer = get_checkpointer("default")
