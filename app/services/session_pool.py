# -*- coding: utf-8 -*-
"""
Supervisor 图实例预热池（SessionPool）。

【设计背景】
每次新会话首次请求时，GraphRunner._get_or_create_supervisor() 需要调用
create_supervisor_graph()，该过程涉及模型初始化、StateGraph 编译和
Checkpointer 连接建立，在冷启动场景下耗时可达 300ms-2s。

【解决方案】
SessionPool 在服务启动/空闲期间预先编译若干 Supervisor 图实例并缓存起来。
当新请求到来时，直接从池中借取已编译好的实例，将首次响应延迟降至接近零。

【设计约束】
- 池实例按 model_config 指纹分组，不同配置各自独立预热。
- 池中实例有最大空闲时间限制，超时后丢弃重建，避免持有陈旧连接。
- 借取失败（池空/超时）时静默降级为按需创建，不影响主链路。
- 后台 refill 线程以 daemon 模式运行，服务退出时自动销毁。
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import defaultdict
from typing import Any, Dict, Optional

from config.runtime_settings import SESSION_POOL_CONFIG
from utils.custom_logger import get_logger

log = get_logger(__name__)


class _PoolEntry:
    """池中单个预热实例的封装。"""

    __slots__ = ("graph", "created_at", "last_used_at")

    def __init__(self, graph: Any) -> None:
        self.graph = graph
        self.created_at = time.monotonic()
        self.last_used_at = time.monotonic()

    def is_expired(self, max_idle_seconds: float) -> bool:
        """判断实例是否超过最大空闲时间。"""
        return (time.monotonic() - self.last_used_at) > max_idle_seconds


class SessionPool:
    """
    Supervisor 图实例预热池。

    用法：
        # 服务启动时初始化
        session_pool.start(default_model_config)

        # GraphRunner 中借取实例
        graph = session_pool.borrow(model_config) or create_supervisor_graph(model_config)

        # 服务关闭时停止后台线程
        session_pool.stop()
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # 按 config_key 分组的实例队列：{config_key: [_PoolEntry, ...]}
        self._pools: Dict[str, list[_PoolEntry]] = defaultdict(list)
        # 每个 config_key 对应的原始 model_config，用于后台 refill
        self._configs: Dict[str, Dict[str, Any]] = {}
        # 正在后台预热的配置，避免重复触发并发 refill
        self._inflight_refills: set[str] = set()
        # 后台 refill 线程
        self._refill_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False

    # ------------------------------------------------------------------ #
    #  公共接口                                                             #
    # ------------------------------------------------------------------ #

    def start(self, default_model_config: Optional[Dict[str, Any]] = None) -> None:
        """
        启动预热池，注册默认配置并启动后台 refill 线程。

        Args:
            default_model_config: 服务默认 model_config，用于预热第一批实例。
        """
        if not SESSION_POOL_CONFIG.enabled:
            log.info("SessionPool 已禁用（SESSION_POOL_ENABLED=false），跳过启动。")
            return

        with self._lock:
            if self._started:
                return
            self._started = True

        if default_model_config:
            self._register_config(default_model_config)
            self._refill_once(default_model_config)

        self._refill_thread = threading.Thread(
            target=self._refill_loop,
            daemon=True,
            name="session-pool-refill",
        )
        self._refill_thread.start()
        log.info(
            f"SessionPool 已启动，pool_size={SESSION_POOL_CONFIG.pool_size}，"
            f"max_idle={SESSION_POOL_CONFIG.max_idle_seconds}s"
        )

    def stop(self) -> None:
        """停止后台 refill 线程，释放所有缓存实例。"""
        self._stop_event.set()
        if self._refill_thread and self._refill_thread.is_alive():
            self._refill_thread.join(timeout=3.0)
        with self._lock:
            self._pools.clear()
            self._started = False
        log.info("SessionPool 已停止。")

    def register_config(self, model_config: Dict[str, Any]) -> None:
        """
        注册一个新的 model_config，下次 refill 时自动预热。

        在 GraphRunner._get_or_create_supervisor() 编译新配置时调用，
        让池提前为该配置预热备用实例。
        """
        if not SESSION_POOL_CONFIG.enabled:
            return
        config_key = self._register_config(model_config)
        self._trigger_refill_async(model_config, config_key=config_key)

    def borrow(
        self,
        model_config: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Optional[Any]:
        """
        从池中借取一个与 model_config 匹配的已编译 Supervisor 图实例。

        Args:
            model_config: 当前请求的模型配置。
            timeout:      等待超时（秒），None 则使用配置默认值。

        Returns:
            成功借取时返回图实例；池空或超时时返回 None（调用方按需创建）。
        """
        if not SESSION_POOL_CONFIG.enabled:
            return None

        config_key = self._build_config_key(model_config)
        max_wait = timeout if timeout is not None else SESSION_POOL_CONFIG.borrow_timeout_seconds
        max_wait = max(0.0, float(max_wait))

        # 快速路径：显式零等待时只尝试一次，不进入 sleep 轮询。
        if max_wait <= 0:
            entry = self._try_pop(config_key)
            if entry is not None:
                entry.last_used_at = time.monotonic()
                log.debug(f"SessionPool 借取成功（nowait），config_key={config_key[:8]}")
                return entry.graph
            log.debug(f"SessionPool 借取未命中（nowait），降级为按需创建。config_key={config_key[:8]}")
            return None

        deadline = time.monotonic() + max_wait

        while time.monotonic() < deadline:
            entry = self._try_pop(config_key)
            if entry is not None:
                entry.last_used_at = time.monotonic()
                log.debug(f"SessionPool 借取成功，config_key={config_key[:8]}")
                return entry.graph
            # 池暂时为空，短暂等待后重试（最多等到 deadline）
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(0.02, remaining))

        log.debug(f"SessionPool 借取超时，降级为按需创建。config_key={config_key[:8]}")
        return None

    def size(self, model_config: Optional[Dict[str, Any]] = None) -> int:
        """返回池中当前可用实例数量。"""
        with self._lock:
            if model_config is None:
                return sum(len(q) for q in self._pools.values())
            key = self._build_config_key(model_config)
            return len(self._pools.get(key, []))

    # ------------------------------------------------------------------ #
    #  内部方法                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_config_key(model_config: Dict[str, Any]) -> str:
        """将 model_config 序列化为 MD5 指纹，与 GraphRunner 保持一致。"""
        stable_json = json.dumps(model_config or {}, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(stable_json.encode("utf-8")).hexdigest()

    def _register_config(self, model_config: Dict[str, Any]) -> str:
        """内部：注册配置（不加锁，调用方负责）。"""
        key = self._build_config_key(model_config)
        with self._lock:
            if key not in self._configs:
                self._configs[key] = dict(model_config)
                log.debug(f"SessionPool 注册新配置，config_key={key[:8]}")
        return key

    def _trigger_refill_async(self, model_config: Dict[str, Any], *, config_key: Optional[str] = None) -> None:
        """为新配置立即触发一次后台预热，减少首轮冷启动抖动。"""
        if not SESSION_POOL_CONFIG.enabled or self._stop_event.is_set():
            return

        key = config_key or self._build_config_key(model_config)
        with self._lock:
            if key in self._inflight_refills:
                return
            self._inflight_refills.add(key)

        threading.Thread(
            target=self._run_prefill_task,
            args=(dict(model_config), key),
            daemon=True,
            name=f"session-pool-prefill-{key[:8]}",
        ).start()

    def _run_prefill_task(self, model_config: Dict[str, Any], config_key: str) -> None:
        try:
            self._refill_once(model_config)
        except Exception as exc:
            log.warning(f"SessionPool 异步预热异常，config_key={config_key[:8]}: {exc}")
        finally:
            with self._lock:
                self._inflight_refills.discard(config_key)

    def _try_pop(self, config_key: str) -> Optional[_PoolEntry]:
        """尝试从对应队列中取出一个未过期实例，线程安全。"""
        max_idle = SESSION_POOL_CONFIG.max_idle_seconds
        with self._lock:
            queue = self._pools.get(config_key)
            if not queue:
                return None
            # 从尾部取（LIFO：最近编译的实例最热）
            entry = queue.pop()
            if entry.is_expired(max_idle):
                log.debug(f"SessionPool 丢弃过期实例，config_key={config_key[:8]}")
                return None
            return entry

    def _refill_once(self, model_config: Dict[str, Any]) -> None:
        """
        为指定配置补充预热实例至 pool_size。

        每次 refill 只补充缺口数量，避免重复创建已有实例。
        编译过程在调用线程中同步执行（refill 线程/启动线程）。
        """
        config_key = self._build_config_key(model_config)
        target = SESSION_POOL_CONFIG.pool_size

        with self._lock:
            current = len(self._pools.get(config_key, []))
        missing = target - current
        if missing <= 0:
            return

        # 延迟导入，避免循环依赖
        try:
            from agent.graphs.supervisor import create_graph as create_supervisor_graph
        except Exception as import_exc:
            log.warning(f"SessionPool refill 导入 create_graph 失败: {import_exc}")
            return

        for _ in range(missing):
            try:
                graph = create_supervisor_graph(model_config)
                entry = _PoolEntry(graph)
                with self._lock:
                    self._pools[config_key].append(entry)
                log.debug(f"SessionPool 预热一个实例，config_key={config_key[:8]}")
            except Exception as exc:
                log.warning(f"SessionPool 预热失败，跳过: {exc}")
                break

    def _evict_expired(self) -> None:
        """清除所有队列中的过期实例。"""
        max_idle = SESSION_POOL_CONFIG.max_idle_seconds
        with self._lock:
            for key in list(self._pools.keys()):
                before = len(self._pools[key])
                self._pools[key] = [
                    e for e in self._pools[key] if not e.is_expired(max_idle)
                ]
                evicted = before - len(self._pools[key])
                if evicted:
                    log.debug(f"SessionPool 清理 {evicted} 个过期实例，config_key={key[:8]}")

    def _refill_loop(self) -> None:
        """后台 refill 线程主循环：定期清理过期实例并补充预热。"""
        interval = SESSION_POOL_CONFIG.refill_interval_seconds
        log.info(f"SessionPool refill 线程启动，间隔={interval}s")
        while not self._stop_event.wait(timeout=interval):
            try:
                self._evict_expired()
                with self._lock:
                    configs_snapshot = dict(self._configs)
                for model_config in configs_snapshot.values():
                    if self._stop_event.is_set():
                        break
                    self._refill_once(model_config)
            except Exception as exc:
                log.warning(f"SessionPool refill 循环异常，已跳过本轮: {exc}")
        log.info("SessionPool refill 线程已退出。")


# 全局单例
session_pool = SessionPool()
