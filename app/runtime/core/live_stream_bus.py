# -*- coding: utf-8 -*-
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Callable, Dict, Set


class AgentStreamBus:
    """
    子 Agent 实时出字总线（进程内）。

    设计要点：
    1. GraphRunner 按 run_id 注册回调，接收子 Agent 的实时 chunk；
    2. 子 Agent 通过 run_id 发布 chunk；
    3. 记录“哪些 Agent 已实时输出”，供上层抑制 synthetic 重复输出。
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._callbacks: Dict[str, Callable[[Dict[str, Any]], None]] = {}
        self._streamed_agents: Dict[str, Set[str]] = defaultdict(set)

    def register_callback(self, run_id: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        key = str(run_id or "").strip()
        if not key or callback is None:
            return
        with self._lock:
            self._callbacks[key] = callback
            self._streamed_agents.pop(key, None)

    def unregister_callback(self, run_id: str) -> None:
        key = str(run_id or "").strip()
        if not key:
            return
        with self._lock:
            self._callbacks.pop(key, None)
            self._streamed_agents.pop(key, None)

    def publish(self, run_id: str, agent_name: str, content: str) -> None:
        key = str(run_id or "").strip()
        text = str(content or "")
        if not key or not text:
            return
        agent = str(agent_name or "").strip()
        callback = None
        with self._lock:
            if agent:
                self._streamed_agents[key].add(agent)
            callback = self._callbacks.get(key)
        if callback is None:
            return
        try:
            callback(
                {
                    "run_id": key,
                    "agent_name": agent,
                    "content": text,
                }
            )
        except Exception:
            return

    def has_streamed(self, run_id: str, agent_name: str) -> bool:
        key = str(run_id or "").strip()
        agent = str(agent_name or "").strip()
        if not key or not agent:
            return False
        with self._lock:
            return agent in self._streamed_agents.get(key, set())


live_stream_bus = AgentStreamBus()

