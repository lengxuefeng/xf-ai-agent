# -*- coding: utf-8 -*-
"""
图执行器模块（GraphRunner）

这个模块是整个 AI 对话链路的核心调度中枢。简单来说：
- API 接口直接调用 GraphRunner 处理用户请求
- GraphRunner 负责启动 Supervisor 图并驱动其执行
- 执行过程中的所有事件（模型输出、工具调用、中断、错误）都会被封装成标准 SSE 格式推给前端
- 内置了前置规则拦截引擎，可以快速响应简单问题（不用调用大模型）
- 负责 Interrupt 审批流的恢复和状态回填

【核心设计要点】

1. 生产者-消费者解耦：图的执行（阻塞同步）在后台 daemon 线程运行，主线程只消费队列推 SSE
   这样做是为了避免同步阻塞导致日志延迟积压

2. Supervisor 编译图缓存：同一配置只编译一次，下次直接使用
   这样可以避免重复编译带来的初始化开销

3. 前置规则拦截：只在输入较短时触发，避免扫描超长文本的性能损耗
   简单问题（如"你好"、"几点了"）可以直接返回，不用进入完整的图执行流程
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import queue
import re
import threading
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Set, Tuple

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.types import Command

from agent.graph_state import AgentRequest
from agent.graphs.supervisor import create_graph as create_supervisor_graph
from agent.llm.unified_loader import create_model_from_config
from agent.rag.vector_store import vector_store_service
from agent.registry import agent_classes
from agent.rules.actions import handle_action
from agent.rules.registry import rule_registry
from config.runtime_settings import (
    AGENT_LIVE_STREAM_ENABLED,
    AGENT_LOOP_CONFIG,
    GRAPH_RUNNER_TUNING,
    SESSION_POOL_CONFIG,
)
from constants.agent_messages import GRAPH_RUNNER_REJECTED_MESSAGE
from constants.approval_constants import (
    ApprovalDecision,
    ApprovalStatus,
    DEFAULT_ALLOWED_DECISIONS,
    DEFAULT_INTERRUPT_MESSAGE,
    SQL_APPROVAL_ACTION_NAME,
)
from constants.sse_constants import SseEventType, SseMessage, SsePayloadField
from constants.workflow_constants import (
    GRAPH_STREAM_MODES,
    GraphQueueItemType,
    WORKER_CANCELLED_RESULT,
    WORKER_PENDING_APPROVAL_RESULT,
)
from services.interrupt_service import interrupt_service
from runtime.core.run_context import build_run_context
from runtime.core.session_manager import runtime_session_manager
from runtime.core.workflow_event_bus import (
    build_workflow_event as runtime_build_workflow_event,
    workflow_display_name,
    workflow_role_for_agent,
    workflow_timestamp,
)
from runtime.context.context_builder import runtime_context_builder
from runtime.hooks.hook_manager import runtime_hook_manager
from runtime.tools.search_gateway import search_gateway
from runtime.tools.tool_registry import runtime_tool_registry
from runtime.workspace.manager import workspace_manager
from services.request_cancellation_service import request_cancellation_service
from utils.custom_logger import CustomLogger, get_logger
from utils.date_utils import get_agent_date_context
from utils.history_compressor import compress_history_messages

log = get_logger(__name__)

# 这些正则表达式用于前端展示时解析 Markdown 代码块和列表格式
_DISPLAY_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_DISPLAY_HEADING_SPLIT_RE = re.compile(r"(?m)^([：:；;])\s*(#{1,6})(?=\S)")
_DISPLAY_LINE_HEADING_RE = re.compile(r"(?m)^(\s{0,3}#{1,6})(?=[^\s#])")
_DISPLAY_SECTION_HEADING_RE = re.compile(
    r"(?m)^(#{1,6}\s*[^\n-：]{2,18}?)(?=(今天|当前是|以下为|日期[:：]|时间[:：]|说明[:：]|建议[:：]|活动[:：]|结果[:：]))"
)
_DISPLAY_HEADING_FIELD_SPLIT_RE = re.compile(
    r"(?m)^(#{1,6}\s*[^\n-]{2,18})-(?=(日期[:：]|时间[:：]|提交时刻[:：]|法定假期[:：]|地点[:：]|主题[:：]|内容[:：]|举办地[:：]|主办[:：]|承办[:：]|说明[:：]))"
)
_DISPLAY_NUMBERED_LIST_RE = re.compile(r"([：:。；;])\s*(\d+\.)")
_DISPLAY_BULLET_LIST_RE = re.compile(r"([：:。；;])\s*([-*])\s+(?=\S)")


class _CodeBlockStash:
    """
    临时存储 Markdown 代码块的辅助类

    前端在渲染 Markdown 时需要区分代码块和普通文本，
    但直接用正则解析比较困难，所以这里用占位符临时替换，
    渲染后再还原回来
    """

    def __init__(self) -> None:
        self.blocks: List[str] = []

    def __call__(self, match: re.Match[str]) -> str:
        # 遇到代码块就用占位符替换
        self.blocks.append(match.group(0))
        return f"@@CODE_BLOCK_{len(self.blocks) - 1}@@"

    def restore(self, text: str) -> str:
        # 把占位符还原回原始代码块
        normalized = text
        for index, block in enumerate(self.blocks):
            normalized = normalized.replace(f"@@CODE_BLOCK_{index}@@", block)
        return normalized


class _GraphStreamAsyncBridge:
    """
    连接图执行和事件队列的桥梁

    图执行是同步阻塞的，而且需要在后台线程运行（因为 LangGraph ToolNode
    遇到 async 工具时会调用 asyncio.run()，要求当前线程没有运行中的事件循环）

    主线程需要通过 asyncio.Queue 异步消费事件并转为 SSE 推送给前端

    这个类封装了这种跨线程通信的逻辑
    """

    def __init__(
        self,
        *,
        runner: "GraphRunner",
        loop: asyncio.AbstractEventLoop,
        event_queue: asyncio.Queue,
        run_context,
        graph: Any,
        graph_inputs: Dict[str, Any],
        graph_config: Dict[str, Any],
    ) -> None:
        self.runner = runner
        self.loop = loop
        self.event_queue = event_queue
        self.run_context = run_context
        self.graph = graph
        self.graph_inputs = graph_inputs
        self.graph_config = graph_config
        self.run_id = run_context.run_id

        # 记录参与本次请求的线程 ID，用于日志拦截器过滤跨会话串流
        # 日志拦截器需要知道哪些日志是当前请求产生的
        self.owner_thread_ids: Set[int] = {threading.get_ident()}

        # 记录上一条日志 SSE，连续重复的日志直接丢弃，避免前端 thinking 刷屏
        self.last_log_sse = ""

    def _enqueue_nowait(self, item: Tuple[str, Any], drop_on_full: bool) -> None:
        """
        不阻塞地尝试向队列写入事件

        如果队列满了：
        - drop_on_full=True：直接丢弃（用于日志等非关键事件）
        - drop_on_full=False：异步等待（用于关键事件）
        """
        try:
            self.event_queue.put_nowait(item)
        except asyncio.QueueFull:
            if not drop_on_full:
                # 异步等待队列有空间
                asyncio.ensure_future(self.event_queue.put(item), loop=self.loop)

    def safe_enqueue(self, item: Tuple[str, Any], *, drop_on_full: bool = False) -> None:
        """
        从后台线程安全地向 asyncio.Queue 写入事件

        调用线程和事件循环线程不是同一个，不能直接操作队列
        必须通过 loop.call_soon_threadsafe 调度到事件循环线程
        """
        self.loop.call_soon_threadsafe(self._enqueue_nowait, item, drop_on_full)

    def log_interceptor(self, sse_message: str) -> None:
        """
        拦截当前请求线程产生的日志并推入事件队列

        CustomLogger 会通过 SSE 回调把所有日志转到这里
        我们只推送属于当前请求的日志（通过线程 ID 判断）
        连续重复的日志直接丢弃，避免前端 thinking 刷屏
        """
        # 不属于当前请求的日志不推送
        if threading.get_ident() not in self.owner_thread_ids:
            return

        # 连续重复的日志不推送
        if sse_message == self.last_log_sse:
            return
        self.last_log_sse = sse_message

        # 日志可以丢弃，不会阻塞执行
        self.safe_enqueue((GraphQueueItemType.LOG.value, sse_message), drop_on_full=True)

    def live_stream_interceptor(self, payload: Dict[str, Any]) -> None:
        """
        将子 Agent 实时正文流推入事件队列

        子 Agent（如 CodeAgent）在执行过程中可能需要实时输出
        通过这个回调可以把实时输出流推送给前端
        """
        # 实时流可以丢弃，不会阻塞执行
        self.safe_enqueue((GraphQueueItemType.LIVE_STREAM.value, payload), drop_on_full=True)

    def graph_worker(self) -> None:
        """
        后台工作线程：驱动 Supervisor 图执行并将事件推入队列

        遇到 Interrupt 异常（子图挂起等待审批）视为正常终止，
        其他未知异常推入 ERROR 事件由消费协程处理

        【为什么要在后台线程运行图执行】
        1. LangGraph ToolNode 遇到 async 工具时会调用 asyncio.run()，
           该函数要求当前线程没有运行中的事件循环
        2. 后台线程默认没有事件循环，asyncio.run() 会自动创建并销毁
        3. 需要显式创建新 loop 来兼容某些 Python 版本的边界情况

        【事件循环说明】
        后台线程创建独立的 asyncio 事件循环，用于执行 async 工具
        主线程通过 call_soon_threadsafe 调度任务到后台线程
        """
        # 为后台线程创建独立事件循环，确保异步工具（async tool）
        # 通过 asyncio.run() 或 loop.run_until_complete() 可正常调用
        _thread_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_thread_loop)

        # 记录这个工作线程的 ID，用于日志过滤
        self.owner_thread_ids.add(threading.get_ident())

        # 绑定运行上下文，用于 SessionManager 等模块获取当前运行信息
        with runtime_session_manager.bind_run(self.run_context):
            try:
                # 流式执行图，获取所有事件
                # stream_mode 同时返回 messages 和 updates，覆盖完整的执行流程
                for ev_type, ev in self.graph.stream(
                    self.graph_inputs,
                    config=self.graph_config,
                    stream_mode=list(GRAPH_STREAM_MODES),
                ):
                    # 检查是否收到取消信号
                    if request_cancellation_service.is_cancelled(self.run_id):
                        log.info(f"Graph Worker 收到取消信号，停止事件采集。run_id={self.run_id}")
                        break

                    # 将图事件推入队列
                    self.safe_enqueue((GraphQueueItemType.GRAPH.value, (ev_type, ev)))

                # 执行完成，推送 DONE 事件
                self.safe_enqueue((GraphQueueItemType.DONE.value, None))

            except Exception as worker_exc:
                err_msg = str(worker_exc)

                # Interrupt 异常是正常终止（等待审批）
                # 其他异常才是真正的错误
                if "Interrupt(" in err_msg or worker_exc.__class__.__name__ == "GraphInterrupt":
                    log.info(f"Graph Worker 检测到 Interrupt 挂起: {err_msg[:200]}")
                    self.safe_enqueue((GraphQueueItemType.DONE.value, None))
                else:
                    log.exception(f"Graph Worker 执行异常: {worker_exc}")
                    # 推送 ERROR 事件，让消费协程处理
                    self.safe_enqueue(
                        (
                            GraphQueueItemType.ERROR.value,
                            self.runner._normalize_graph_error_message(worker_exc),
                        )
                    )
            finally:
                # 关闭线程本地事件循环，释放资源
                try:
                    _thread_loop.close()
                except Exception:
                    # 关闭失败不应该影响主流程
                    pass


class GraphRunner:
    """
    图执行器：AI 对话链路的核心调度中枢

    这个类是整个系统的核心，负责：

    1. 管理 Supervisor 编译图缓存
       同一配置的 Supervisor 图只编译一次，下次直接使用
       可以避免重复编译带来的初始化开销

    2. 将同步图执行包装为可 yield 的 SSE 流
       供 FastAPI StreamingResponse 消费
       前端可以实时收到流式响应

    3. 拦截前置规则命中、RAG 注入、Interrupt 审批恢复等横切关注点
       这些是整个执行流程中需要特殊处理的节点

    【设计原则】
    - 所有可能阻塞的操作都放在后台线程
    - 主线程只负责消费队列和推送 SSE
    - 这样可以支持真正的并发流式推送，不会阻塞事件循环
    """

    def __init__(self, model_config: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化执行器，可选注入默认模型配置

        model_config: 默认的模型配置，每次请求的配置会与这个默认配置合并
        """
        self.model_config: Dict[str, Any] = model_config or {}

        # Supervisor 编译图缓存：按模型配置的 MD5 指纹缓存
        # 同一配置只编译一次，下次直接使用缓存
        self._supervisor_cache: Dict[str, Any] = {}
        self._supervisor_cache_lock = threading.Lock()

        # Supervisor 编译事件：用于通知等待中的线程
        # SessionPool 可以等待某个配置的 Supervisor 编译完成
        self._supervisor_build_events: Dict[str, threading.Event] = {}
