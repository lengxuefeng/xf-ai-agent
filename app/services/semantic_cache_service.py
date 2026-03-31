# -*- coding: utf-8 -*-
import hashlib
import threading
import time
from typing import Dict, Any, Optional

from config.runtime_settings import SEMANTIC_CACHE_CONFIG
from common.utils.custom_logger import get_logger, LogTarget

log = get_logger(__name__)


class SemanticCacheService:
    """
    轻量语义缓存服务（内存版）。

    说明：
    1. 主要用于缓存高频重复查询结果，降低数据库与模型压力。
    2. 当前接入 SQL 查询链路，后续可扩展到搜索与聚合结果。
    """

    def __init__(self, default_ttl_seconds: int = 120, max_size: int = 1000):
        """初始化缓存参数和内存存储。"""
        self.default_ttl_seconds = default_ttl_seconds
        self.max_size = max_size
        self._lock = threading.Lock()
        self._store: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _normalize_text(text: str) -> str:
        """统一文本规范，减少“同义空白差异”造成的缓存抖动。"""
        return " ".join((text or "").strip().lower().split())

    def build_key(self, domain: str, text: str) -> str:
        """按域名+文本摘要构建稳定缓存键。"""
        normalized = self._normalize_text(text)
        digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        return f"{(domain or 'GENERAL').upper()}:{digest}"

    def get(self, key: str) -> Optional[str]:
        """读取缓存；若已过期则顺便清理。"""
        now = time.time()
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            if now >= item["expire_at"]:
                self._store.pop(key, None)
                return None
            item["last_access"] = now
            return item["value"]

    def set(self, key: str, value: str, ttl_seconds: Optional[int] = None):
        """写入缓存；容量超限时淘汰最久未访问项。"""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        now = time.time()
        with self._lock:
            if len(self._store) >= self.max_size:
                # 近似 LRU：淘汰最久未访问项
                oldest_key = min(self._store.keys(), key=lambda k: self._store[k].get("last_access", 0))
                self._store.pop(oldest_key, None)
            self._store[key] = {
                "value": value,
                "expire_at": now + max(1, ttl),
                "last_access": now,
            }

    def snapshot(self) -> Dict[str, Any]:
        """输出缓存规模与配置，方便排查。"""
        with self._lock:
            return {
                "size": len(self._store),
                "default_ttl_seconds": self.default_ttl_seconds,
                "max_size": self.max_size,
            }


semantic_cache_service = SemanticCacheService(
    default_ttl_seconds=SEMANTIC_CACHE_CONFIG.default_ttl_seconds,
    max_size=SEMANTIC_CACHE_CONFIG.max_size,
)
