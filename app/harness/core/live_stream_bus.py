# -*- coding: utf-8 -*-
"""
子Agent实时出字总线（Agent Stream Bus）。

这是一个进程内的消息总线，用于子Agent向GraphRunner实时推送文本内容。
它实现了生产者-消费者模式，子Agent是生产者，GraphRunner是消费者。

设计要点：
1. 基于run_id的回调注册机制，GraphRunner可以注册回调函数
2. 子Agent通过run_id发布实时内容，回调函数会被调用
3. 记录哪些Agent已经实时输出，避免重复推送synthetic输出
4. 线程安全，支持多线程并发发布

使用场景：
- Chat Agent正在生成回复，实时推送每一段文本给前端
- 其他Agent有输出内容时，也能实时推送给前端
- 避免重复推送：如果一个Agent已经实时输出了，Aggregator就不要再重复输出

架构说明：
- 本模块是进程内的轻量级消息总线
- 适用于单进程场景，分布式场景需要使用其他机制
- 设计为单例，全局共享同一个实例
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Callable, Dict, Set


class AgentStreamBus:
    """
    子 Agent 实时出字总线（进程内）。

    核心职责：
    1. 注册回调函数：GraphRunner注册接收实时内容的回调
    2. 发布实时内容：子Agent发布实时文本
    3. 记录输出状态：记录哪些Agent已经输出过
    4. 查询输出状态：上层可以查询Agent是否已输出

    数据结构：
    - _callbacks: run_id -> callback (回调函数）
    - _streamed_agents: run_id -> set(agent_name) (已输出的Agent集合）

    设计要点：
    1. 使用RLock保护共享数据，支持多线程并发
    2. 回调函数在锁外调用，避免死锁
    3. 异常处理：回调抛出异常不影响其他Agent的输出
    4. 轻量级：只传递文本内容，不传递复杂对象
    """

    def __init__(self):
        """初始化实时流总线。"""
        self._lock = threading.RLock()  # 保护共享数据的锁
        self._callbacks: Dict[str, Callable[[Dict[str, Any]], None]] = {}  # 回调函数字典
        self._streamed_agents: Dict[str, Set[str]] = defaultdict(set)  # 已输出的Agent集合

    def register_callback(self, run_id: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        注册实时流回调函数。

        设计要点：
        1. 注册后，所有Agent发布的实时内容都会通过这个回调传递
        2. 同一个run_id只能注册一个回调，新注册会覆盖旧的
        3. 注册时清除已输出的Agent记录，开始新的一次运行

        Args:
            run_id: 运行ID，用于标识一次运行
            callback: 回调函数，签名为callback(dict) -> None
                     回调参数包含run_id, agent_name, content

        场景：
        - GraphRunner开始执行前注册回调
        - 回调函数将实时内容推送到SSE流
        """
        key = str(run_id or "").strip()
        if not key or callback is None:
            return
        with self._lock:
            self._callbacks[key] = callback
            # 清除已输出的Agent记录，开始新的一次运行
            self._streamed_agents.pop(key, None)

    def unregister_callback(self, run_id: str) -> None:
        """
        注销实时流回调函数。

        设计要点：
        1. 运行结束后应该注销回调，避免内存泄漏
        2. 同时清除已输出的Agent记录

        Args:
            run_id: 运行ID

        场景：
        - 运行正常结束后注销
        - 运行被取消时注销
        - 异常退出时注销（在finally中）
        """
        key = str(run_id or "").strip()
        if not key:
            return
        with self._lock:
            self._callbacks.pop(key, None)
            self._streamed_agents.pop(key, None)

    def publish(
        self,
        run_id: str,
        agent_name: str,
        content: str,
        *,
        task_id: str = "",
        body_stream: bool = False,
    ) -> None:
        """
        发布实时内容。

        设计要点：
        1. 记录Agent已输出，避免重复推送
        2. 调用注册的回调函数
        3. 回调函数在锁外调用，避免死锁
        4. 回调抛出异常不影响其他Agent

        Args:
            run_id: 运行ID
            agent_name: Agent名称
            content: 实时文本内容

        场景：
        - Chat Agent生成文本时逐段发布
        - 其他Agent有输出时发布
        """
        key = str(run_id or "").strip()
        text = str(content or "")
        if not key or not text:
            return
        agent = str(agent_name or "").strip()
        callback = None

        with self._lock:
            # 记录Agent已输出
            if agent:
                self._streamed_agents[key].add(agent)
            # 获取回调函数
            callback = self._callbacks.get(key)

        if callback is None:
            return

        try:
            # 在锁外调用回调，避免死锁
            callback(
                {
                    "run_id": key,
                    "agent_name": agent,
                    "content": text,
                    "task_id": str(task_id or "").strip(),
                    "body_stream": bool(body_stream),
                }
            )
        except Exception:
            # 回调抛出异常不影响其他Agent的输出
            return

    def has_streamed(self, run_id: str, agent_name: str) -> bool:
        """
        查询Agent是否已经实时输出过。

        设计要点：
        1. 用于判断是否需要抑制synthetic输出
        2. 如果Agent已经实时输出，Aggregator不要再重复输出

        Args:
            run_id: 运行ID
            agent_name: Agent名称

        Returns:
            bool: True表示已输出，False表示未输出

        场景：
        - Aggregator聚合结果时，检查Agent是否已实时输出
        - 如果已实时输出，不重复推送synthetic输出
        """
        key = str(run_id or "").strip()
        agent = str(agent_name or "").strip()
        if not key or not agent:
            return False
        with self._lock:
            return agent in self._streamed_agents.get(key, set())


# 全局唯一的实时流总线实例
live_stream_bus = AgentStreamBus()
