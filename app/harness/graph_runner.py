# -*- coding: utf-8 -*-
from __future__ import annotations

from services.session_pool import session_pool

"""
图执行器模块（GraphRunner）。

【模块职责】
- API 接口的直接消费者，负责拉起 Supervisor 图并驱动其执行。
- 将图执行过程中产生的所有事件（模型输出、工具调用、中断、错误）
  封装为标准 SSE 格式推送给前端。
- 内置前置规则拦截引擎（Zero-LLM 快速响应）。
- 负责 Interrupt 审批流的恢复和状态回填。

【设计要点】
- 生产者-消费者解耦：图执行跑在后台 daemon 线程，主线程只消费队列推 SSE，
  避免同步阻塞导致日志延迟积压。
- Supervisor 编译图按 model_config 指纹缓存，同一配置只编译一次。
- 前置规则拦截仅在输入较短时触发，避免扫描超长文本的性能损耗。
"""

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

from harness.graph_runner_core import rule_handle
from supervisor.graph_state import AgentRequest
from supervisor.supervisor import create_graph as create_supervisor_graph
from common.llm.unified_loader import create_model_from_config
from tools.rag.vector_store import vector_store_service
from supervisor.registry import agent_classes
from supervisor.rules.actions import handle_action
from supervisor.rules.registry import rule_registry
from config.runtime_settings import (
    AGENT_LIVE_STREAM_ENABLED,
    AGENT_LOOP_CONFIG,
    GRAPH_RUNNER_TUNING,
    SESSION_POOL_CONFIG,
)
from config.constants.agent_messages import GRAPH_RUNNER_REJECTED_MESSAGE
from config.constants.approval_constants import (
    ApprovalDecision,
    ApprovalStatus,
    DEFAULT_ALLOWED_DECISIONS,
    DEFAULT_INTERRUPT_MESSAGE,
    SQL_APPROVAL_ACTION_NAME,
)
from config.constants.sse_constants import SseEventType, SseMessage, SsePayloadField
from config.constants.workflow_constants import (
    GRAPH_STREAM_MODES,
    GraphQueueItemType,
    WORKER_CANCELLED_RESULT,
    WORKER_PENDING_APPROVAL_RESULT,
)
from services.interrupt_service import interrupt_service
from harness.core.run_context import build_run_context
from harness.core.session_manager import runtime_session_manager
from harness.core.workflow_event_bus import (
    build_workflow_event as runtime_build_workflow_event,
    workflow_display_name,
    workflow_role_for_agent,
    workflow_timestamp,
)
from harness.context.context_builder import runtime_context_builder
from harness.hooks.hook_manager import runtime_hook_manager
from tools.runtime_tools.search_gateway import search_gateway
from tools.runtime_tools.tool_registry import runtime_tool_registry
from harness.workspace.manager import workspace_manager
from services.request_cancellation_service import request_cancellation_service
from common.utils.custom_logger import CustomLogger, get_logger
from common.utils.date_utils import get_agent_date_context
from common.utils.history_compressor import compress_history_messages

log = get_logger(__name__)

_DISPLAY_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_DISPLAY_HEADING_SPLIT_RE = re.compile(r"([：:。；;])\s*(#{1,6})(?=\S)")
_DISPLAY_LINE_HEADING_RE = re.compile(r"(?m)^(\s{0,3}#{1,6})(?=[^\s#])")
_DISPLAY_SECTION_HEADING_RE = re.compile(
    r"(?m)^(#{1,6}\s*[^\n:：-]{2,18}?)(?=(今天是|当前是|以下为|日期[:：]|时间[:：]|说明[:：]|建议[:：]|活动[:：]|结果[:：]))"
)
_DISPLAY_HEADING_FIELD_SPLIT_RE = re.compile(
    r"(?m)^(#{1,6}\s*[^\n-]{2,18})-(?=(日期[:：]|时间[:：]|交节时刻[:：]|法定假期[:：]|地点[:：]|主题[:：]|内容[:：]|举办地[:：]|主办[:：]|承办[:：]|说明[:：]))"
)
_DISPLAY_NUMBERED_LIST_RE = re.compile(r"([：:。；;])\s*(\d+\.)")
_DISPLAY_BULLET_LIST_RE = re.compile(r"([：:。；;])\s*([-*])\s+(?=\S)")


class _CodeBlockStash:
    def __init__(self) -> None:
        self.blocks: List[str] = []

    def __call__(self, match: re.Match[str]) -> str:
        self.blocks.append(match.group(0))
        return f"@@CODE_BLOCK_{len(self.blocks) - 1}@@"

    def restore(self, text: str) -> str:
        normalized = text
        for index, block in enumerate(self.blocks):
            normalized = normalized.replace(f"@@CODE_BLOCK_{index}@@", block)
        return normalized


class _GraphStreamAsyncBridge:
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
        self.owner_thread_ids: Set[int] = {threading.get_ident()}
        self.last_log_sse = ""

    def _enqueue_nowait(self, item: Tuple[str, Any], drop_on_full: bool) -> None:
        try:
            self.event_queue.put_nowait(item)
        except asyncio.QueueFull:
            if not drop_on_full:
                asyncio.ensure_future(self.event_queue.put(item), loop=self.loop)

    def safe_enqueue(self, item: Tuple[str, Any], *, drop_on_full: bool = False) -> None:
        self.loop.call_soon_threadsafe(self._enqueue_nowait, item, drop_on_full)

    def log_interceptor(self, sse_message: str) -> None:
        if threading.get_ident() not in self.owner_thread_ids:
            return
        if sse_message == self.last_log_sse:
            return
        self.last_log_sse = sse_message
        self.safe_enqueue((GraphQueueItemType.LOG.value, sse_message), drop_on_full=True)

    def live_stream_interceptor(self, payload: Dict[str, Any]) -> None:
        self.safe_enqueue((GraphQueueItemType.LIVE_STREAM.value, payload), drop_on_full=True)

    def graph_worker(self) -> None:
        _thread_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_thread_loop)
        self.owner_thread_ids.add(threading.get_ident())
        with runtime_session_manager.bind_run(self.run_context):
            try:
                for ev_type, ev in self.graph.stream(
                        self.graph_inputs,
                        config=self.graph_config,
                        stream_mode=list(GRAPH_STREAM_MODES),
                ):
                    if request_cancellation_service.is_cancelled(self.run_id):
                        log.info(f"Graph Worker 收到取消信号，停止事件采集。run_id={self.run_id}")
                        break
                    self.safe_enqueue((GraphQueueItemType.GRAPH.value, (ev_type, ev)))
                self.safe_enqueue((GraphQueueItemType.DONE.value, None))
            except Exception as worker_exc:
                err_msg = str(worker_exc)
                if "Interrupt(" in err_msg or worker_exc.__class__.__name__ == "GraphInterrupt":
                    log.info(f"Graph Worker 检测到 Interrupt 挂起: {err_msg[:200]}")
                    self.safe_enqueue((GraphQueueItemType.DONE.value, None))
                else:
                    log.exception(f"Graph Worker 执行异常: {worker_exc}")
                    self.safe_enqueue(
                        (
                            GraphQueueItemType.ERROR.value,
                            self.runner._normalize_graph_error_message(worker_exc),
                        )
                    )
            finally:
                try:
                    _thread_loop.close()
                except Exception:
                    pass


class GraphRunner:
    """
    图执行器：AI 对话链路的核心调度中枢。

    职责：
    1. 管理 Supervisor 编译图缓存，同一配置只编译一次。
    2. 将同步图执行包装为可 yield 的 SSE 流，供 FastAPI StreamingResponse 消费。
    3. 拦截前置规则命中、RAG 注入、Interrupt 审批恢复等横切关注点。
    """

    def __init__(self, model_config: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化执行器，可选注入默认模型配置

        model_config: 默认的模型配置，每次请求的配置会与这个默认配置合并
        请求级配置的优先级更高
        """
        self.model_config: Dict[str, Any] = model_config or {}

        # Supervisor 编译图缓存：按模型配置的 MD5 指纹缓存
        # 同一配置只编译一次，避免重复编译带来的初始化开销
        # 编译图比较耗时，缓存可以显著提升响应速度
        self._supervisor_cache: Dict[str, Any] = {}
        self._supervisor_cache_lock = threading.Lock()

        # Supervisor 编译事件：用于通知等待中的线程
        # SessionPool 可以在后台预热 Supervisor 实例
        # 编译完成后通过这个事件通知等待中的线程
        self._supervisor_build_events: Dict[str, threading.Event] = {}

    # ------------------------------------------------------------------ #
    #  Workflow 事件工具                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _workflow_timestamp() -> str:
        """返回统一的 UTC 时间戳，供前端流程轨迹排序使用。"""
        return workflow_timestamp()

    @staticmethod
    def _workflow_role_for_agent(agent_name: str) -> str:
        """根据 Agent 名称推断流程展示中的角色。"""
        return workflow_role_for_agent(agent_name)

    @staticmethod
    def _workflow_display_name(agent_name: str) -> str:
        """将内部 Agent 名称转换为更适合前端展示的标签。"""
        return workflow_display_name(agent_name)

    @classmethod
    def _normalize_display_text_segment(cls, text: str) -> str:
        """最小修正常见 Markdown 粘连问题，保留结构化标记供前端渲染。"""
        if not text:
            return ""

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.replace("\u200b", "")
        if "\\n" in normalized:
            normalized = normalized.replace("\\n", "\n")
        normalized = _DISPLAY_HEADING_SPLIT_RE.sub(r"\1\n\n\2 ", normalized)
        normalized = re.sub(r"([^\n#])\s*(#{1,6}\s)", r"\1\n\n\2", normalized)
        normalized = _DISPLAY_LINE_HEADING_RE.sub(r"\1 ", normalized)
        normalized = _DISPLAY_SECTION_HEADING_RE.sub(r"\1\n", normalized)
        normalized = _DISPLAY_HEADING_FIELD_SPLIT_RE.sub(r"\1\n- ", normalized)
        normalized = _DISPLAY_NUMBERED_LIST_RE.sub(r"\1\n\2 ", normalized)
        normalized = _DISPLAY_BULLET_LIST_RE.sub(r"\1\n\2 ", normalized)
        normalized = re.sub(r"([^\n])\n(?=(?:[-*+]\s|\d+\.\s|>\s|#{1,6}\s|```))", r"\1\n\n", normalized)
        normalized = re.sub(r"[ \t]+\n", "\n", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        normalized = re.sub(r"[ \t]{2,}", " ", normalized)
        return normalized.strip()

    @classmethod
    def _format_user_visible_text(cls, content: Any) -> str:
        """统一整理用户可见正文，保留代码块，并尽量保留 Markdown 结构。"""
        raw_text = str(content or "")
        if not raw_text.strip():
            return ""

        stash = _CodeBlockStash()
        normalized = _DISPLAY_CODE_BLOCK_RE.sub(stash, raw_text)
        normalized = cls._normalize_display_text_segment(normalized)
        return stash.restore(normalized).strip()

    @classmethod
    def _build_workflow_event(
            cls,
            *,
            session_id: str,
            run_id: str,
            phase: str,
            title: str,
            summary: str = "",
            status: str = "info",
            role: str = "system",
            agent_name: str = "",
            task_id: str = "",
            node_name: str = "",
            meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """构造统一结构化流程事件。"""
        return runtime_build_workflow_event(
            session_id=session_id,
            run_id=run_id,
            phase=phase,
            title=title,
            summary=summary,
            status=status,
            role=role,
            agent_name=agent_name,
            task_id=task_id,
            node_name=node_name,
            meta=meta,
        )

    # ------------------------------------------------------------------ #
    #  Supervisor 缓存管理                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_config_key(model_config: Dict[str, Any]) -> str:
        """
        将模型配置序列化为 MD5 指纹，用作 Supervisor 编译图缓存键。

        使用 sort_keys 保证相同内容但字段顺序不同的配置产生相同指纹。
        """
        stable_json = json.dumps(model_config or {}, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(stable_json.encode("utf-8")).hexdigest()

    def _get_or_create_supervisor(
            self,
            model_config: Dict[str, Any],
            borrow_timeout: Optional[float] = None,
    ) -> Any:
        """
        获取（或创建）Supervisor 编译图实例。

        优先级：
        1. 实例缓存（同进程内相同 config_key）
        2. SessionPool 预热池（跨请求共享预热实例）
        3. 按需编译（冷启动降级）
        """
        cache_key = self._build_config_key(model_config)
        with self._supervisor_cache_lock:
            cached = self._supervisor_cache.get(cache_key)
            if cached is not None:
                return cached

            build_event = self._supervisor_build_events.get(cache_key)
            if build_event is None:
                build_event = threading.Event()
                self._supervisor_build_events[cache_key] = build_event
                is_builder = True
            else:
                is_builder = False

        if not is_builder:
            build_event.wait()
            with self._supervisor_cache_lock:
                cached = self._supervisor_cache.get(cache_key)
                if cached is not None:
                    return cached
            # 构建线程若异常退出，当前线程兜底重试一次。
            return self._get_or_create_supervisor(model_config, borrow_timeout=borrow_timeout)
        try:
            # 先尝试从预热池借取，命中则直接写入本地缓存
            pooled = session_pool.borrow(model_config, timeout=borrow_timeout)
            if pooled is not None:
                log.info(f"从 SessionPool 借取预热实例，config_key={cache_key[:8]}...")
                graph = pooled
            else:
                log.info(f"首次编译 Supervisor 图，config_key={cache_key[:8]}...")
                graph = create_supervisor_graph(model_config)
                # 编译完成后通知池注册该配置，下次 refill 时预热备用实例
                session_pool.register_config(model_config)

            with self._supervisor_cache_lock:
                self._supervisor_cache[cache_key] = graph
            return graph
        finally:
            with self._supervisor_cache_lock:
                event = self._supervisor_build_events.pop(cache_key, None)
                if event is not None:
                    event.set()

    def _get_supervisor(self, model_config: Dict[str, Any]) -> Any:
        """
        兼容旧代码路径：保留历史方法名。

        旧单测/调用方仍可能 patch `_get_supervisor`，这里转发到新实现，
        避免接口重构带来非功能性回归。
        """
        return self._get_or_create_supervisor(model_config)

    async def _get_or_create_supervisor_async(self, model_config: Dict[str, Any]) -> Any:
        """
        在后台线程中借取/编译 Supervisor 图，避免阻塞事件循环。
        """
        borrow_timeout = float(SESSION_POOL_CONFIG.borrow_timeout_seconds)
        return await asyncio.to_thread(
            self._get_or_create_supervisor,
            model_config,
            borrow_timeout,
        )

    # ------------------------------------------------------------------ #
    #  核心公共接口：stream_run                                             #
    # ------------------------------------------------------------------ #

    async def stream_run(
            self,
            user_input: str,
            session_id: str,
            model_config: Optional[Dict[str, Any]] = None,
            history_messages: Optional[List[Dict[str, Any]]] = None,
            session_context: Optional[Dict[str, Any]] = None,
            emit_response_start: bool = True,
    ) -> AsyncGenerator[str, None]:
        """
        核心流式执行器，所有用户请求都会经过这个方法

        【执行流程概览】
        1. 合并默认配置和请求配置，请求配置优先级更高
        2. 审批恢复流程（如果 user_input == "[RESUME]"）
        3. 基础输入校验
        4. 前置规则拦截（简单问题直接返回）
        5. 创建运行上下文和工作区
        6. 构建上下文（历史裁剪、记忆注入等）
        7. RAG 注入（如果启用）
        8. 驱动图执行并返回 SSE 流

        【为什么需要异步】
        - 图执行是阻塞同步的，必须在后台线程运行
        - 主线程负责消费事件队列并推送 SSE
        - 这样可以支持真正的并发流式推送，不会阻塞事件循环

        Args:
            user_input: 用户当前输入文本，"[RESUME]" 表示审批恢复
            session_id: 会话 ID，同时作为 LangGraph checkpointer 的 thread_id
            model_config: 覆盖默认配置的模型参数字典
            history_messages: 历史对话列表 [{user_content, model_content}, ...]
            session_context: 结构化会话上下文（城市、用户画像等槽位）
            emit_response_start: 是否在流开始时发送 response_start 事件

        Yields:
            标准 SSE 格式字符串（"event: ...\ndata: ...\n\n"）
        """
        history_messages = history_messages or []
        session_context = session_context or {}

        # 合并默认配置和请求配置，请求级配置优先级更高
        effective_config = {**self.model_config, **(model_config or {})}
        resume_message_id = str(effective_config.get("resume_message_id") or "").strip()

        # ── 分支一：审批恢复流程
        # 用户通过审批后恢复执行，需要找到原始中断点继续
        if user_input == "[RESUME]":
            # 获取 Supervisor 编译图
            graph = await self._get_or_create_supervisor_async(effective_config)
            # 创建恢复上下文（标记 is_resume=True）
            resume_context = runtime_session_manager.create_run_context(
                session_id=session_id,
                user_input=user_input,
                model_config=effective_config,
                history_messages=history_messages,
                session_context=session_context,
                is_resume=True,
            )
            runtime_session_manager.register_run(resume_context)

            # 处理恢复流程并返回流式结果
            async for chunk in self._handle_resume_stream_async(
                    run_context=resume_context,
                    graph=graph,
                    effective_config=effective_config,
                    emit_response_start=emit_response_start,
                    resume_message_id=resume_message_id,
            ):
                yield chunk
            return

        # ── 基础输入校验 ──────────────────────────────────────────────────
        # 空输入直接返回错误，避免进入图执行
        if not (user_input or "").strip():
            yield self._fmt_sse(SseEventType.ERROR.value, SseMessage.INVALID_RESUME_PARAM)
            return

        # ── 分支二：前置规则拦截（Zero-LLM、Zero-Graph） ───────────────────
        # 简单问题直接返回预设答案，不调用大模型
        # 这样可以节省费用、提高响应速度、保证一致性

        # 创建运行上下文
        run_context = build_run_context(
            session_id=session_id,
            user_input=user_input,
            model_config=effective_config,
            history_messages=history_messages,
            session_context=session_context,
        )
        run_id = run_context.run_id

        # 尝试规则拦截
        rule_result = self._try_rule_intercept(user_input)
        if rule_result is not None:
            # 规则命中，直接返回结果，不进入图执行
            rule_handle(self, user_input=user_input, session_id=session_id, rule_result=rule_result,
                        emit_response_start=emit_response_start, run_context=run_context, run_id=run_id)

        # response_start 必须在任何内容之前发出，让前端进入流式接收模式
        if emit_response_start:
            yield self._fmt_sse(SseEventType.RESPONSE_START.value, "")
        yield self._fmt_workflow_event(
            self._build_workflow_event(
                session_id=session_id,
                run_id=run_id,
                phase="user_message_received",
                title="老板下达任务",
                summary=user_input,
                status="completed",
                role="boss",
                meta={"input_length": len(user_input or "")},
            )
        )

        runtime_session_manager.register_run(run_context)

        workspace_meta = workspace_manager.prepare_run_workspace(run_context)
        tool_registry_stats = runtime_tool_registry.stats()
        tool_catalog = runtime_tool_registry.list_tools()
        search_capability = search_gateway.capability_snapshot()
        bootstrap_artifacts = [
            workspace_manager.write_json_artifact(
                run_context,
                name="request_context",
                payload={
                    "session_id": session_id,
                    "run_id": run_id,
                    "user_input": user_input,
                    "history_size": len(history_messages or []),
                    "session_context": session_context,
                },
                category="request",
            )
        ]
        runtime_session_manager.attach_meta(
            run_context,
            workspace=workspace_meta,
            tool_registry_stats=tool_registry_stats,
            search_capability=search_capability,
            artifacts=workspace_manager.list_artifacts(run_context),
        )
        yield self._fmt_workflow_event(
            self._build_workflow_event(
                session_id=session_id,
                run_id=run_id,
                phase="workspace_prepared",
                title="卷宗工位已就绪",
                summary=f"已为本轮任务创建工作区 {workspace_meta.get('relative_root')}",
                status="completed",
                role="system",
                meta={"workspace": workspace_meta, "artifacts": bootstrap_artifacts},
            )
        )

        pre_run_hooks = runtime_hook_manager.run_pre_run_hooks(run_context)
        runtime_session_manager.attach_meta(
            run_context,
            pre_run_hooks=[hook.to_dict() for hook in pre_run_hooks],
        )
        for hook in pre_run_hooks:
            hook_payload = hook.to_dict()
            yield self._fmt_workflow_event(
                self._build_workflow_event(
                    session_id=session_id,
                    run_id=run_id,
                    phase="hook_checked",
                    title=f"护栏检查: {hook_payload['name']}",
                    summary=hook_payload["summary"],
                    status=hook_payload["status"],
                    role="system",
                    meta={"hook": hook_payload},
                )
            )

        # ── 构造入图消息列表 ──────────────────────────────────────────────
        messages, memory_snippets, context_meta = runtime_context_builder.build_messages(
            run_context=run_context,
            history_messages=history_messages,
            session_context=session_context,
            max_tokens=AGENT_LOOP_CONFIG.context_compress_max_tokens,
            max_chars=AGENT_LOOP_CONFIG.context_compress_max_chars,
        )
        workspace_manager.write_json_artifact(
            run_context,
            name="memory_snippets",
            payload=[snippet.to_dict() for snippet in memory_snippets],
            category="memory",
        )
        workspace_manager.write_json_artifact(
            run_context,
            name="context_meta",
            payload=context_meta,
            category="context",
        )
        runtime_session_manager.attach_meta(
            run_context,
            memory_snippets=[snippet.to_dict() for snippet in memory_snippets],
            context_meta=context_meta,
            artifacts=workspace_manager.list_artifacts(run_context),
        )
        yield self._fmt_workflow_event(
            self._build_workflow_event(
                session_id=session_id,
                run_id=run_id,
                phase="memory_loaded",
                title="中枢调入项目与会话记忆",
                summary=f"共注入 {len(memory_snippets)} 条记忆片段",
                status="completed",
                role="system",
                meta={"memory_snippets": [snippet.to_dict() for snippet in memory_snippets]},
            )
        )
        yield self._fmt_workflow_event(
            self._build_workflow_event(
                session_id=session_id,
                run_id=run_id,
                phase="context_ready",
                title="上下文卷宗整理完成",
                summary=f"压缩历史 {context_meta.get('compressed_history_count', 0)} 条，估算 {context_meta.get('estimated_tokens', 0)} tokens",
                status="completed",
                role="system",
                meta=context_meta,
            )
        )
        yield self._fmt_workflow_event(
            self._build_workflow_event(
                session_id=session_id,
                run_id=run_id,
                phase="tool_registry_ready",
                title="运行时工具与外部能力已装载",
                summary=f"已登记 {tool_registry_stats.get('total', 0)} 项工具能力",
                status="completed",
                role="system",
                meta={
                    "tool_registry_stats": tool_registry_stats,
                    "tool_catalog": tool_catalog,
                    "search_capability": search_capability,
                },
            )
        )

        yield self._fmt_workflow_event(
            self._build_workflow_event(
                session_id=session_id,
                run_id=run_id,
                phase="artifacts_indexed",
                title="运行时卷宗已入册",
                summary=f"当前已生成 {len(workspace_manager.list_artifacts(run_context))} 份运行材料",
                status="completed",
                role="system",
                meta={"artifacts": workspace_manager.list_artifacts(run_context)},
            )
        )

        # ── RAG 上下文注入（可选） ────────────────────────────────────────
        rag_thinking_text = self._inject_rag_context(messages, user_input, effective_config)

        if rag_thinking_text:
            yield self._fmt_sse(SseEventType.THINKING.value, rag_thinking_text)
            yield self._fmt_workflow_event(
                self._build_workflow_event(
                    session_id=session_id,
                    run_id=run_id,
                    phase="knowledge_augmented",
                    title="总管调阅知识卷宗",
                    summary=rag_thinking_text,
                    status="completed",
                    role="supervisor",
                    agent_name="ChatAgent",
                )
            )

        # 非阻塞借取：避免在事件循环线程等待 SessionPool 轮询 sleep。
        graph = await self._get_or_create_supervisor_async(effective_config)

        # ── 构造图执行 Config ─────────────────────────────────────────────
        graph_config = run_context.graph_config()
        workspace_root = str(effective_config.get("workspace_root") or "").strip()
        if workspace_root:
            graph_config.setdefault("configurable", {})["workspace_root"] = workspace_root
            runtime_session_manager.attach_meta(run_context, workspace_root=workspace_root)
        graph_inputs = {
            "messages": messages,
            "session_id": session_id,
            "llm_config": effective_config,
            "context_slots": session_context.get("context_slots") or {},
            "context_summary": session_context.get("context_summary") or "",
        }

        # ── 启动后台图执行并消费事件队列（异步） ──────────────────────────
        async for chunk in self._run_graph_stream_async(
                graph=graph,
                graph_inputs=graph_inputs,
                graph_config=graph_config,
                run_context=run_context,
                effective_config=effective_config,
        ):
            yield chunk

    # ------------------------------------------------------------------ #
    #  前置规则拦截引擎                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _try_rule_intercept(user_input: str) -> Optional[Tuple[str, str]]:
        """
        前置规则拦截器（Zero-LLM、Zero-Graph）

        简单问题（如"你好"、"几点了"）不需要调用大模型，直接返回预设的回复即可
        这样可以：
        1. 避免不必要的 LLM 调用，节省费用和响应时间
        2. 保证简单问题的一致性和准确性

        【拦截条件】
        1. 输入命中预设规则
        2. 输入长度在规则能覆盖的范围内
        3. 不是复杂的长句（避免误拦截）

        【为什么设置长度限制】
        - 超长文本的匹配开销大，可能影响性能
        - 超长文本可能包含复杂意图，规则无法准确处理

        Returns:
            命中时返回 (thinking提示文本, 回复正文)
            未命中或放行时返回 None
        """
        user_text_lower = user_input.lower().strip()
        max_scan_len = GRAPH_RUNNER_TUNING.rule_scan_max_len

        # 超长输入直接跳过规则扫描
        # 超长文本可能是复杂问题，需要大模型理解
        # 扫描超长文本的性能开销也较大
        if len(user_text_lower) > max_scan_len:
            return None

        # 按优先级排序规则，高优先级规则先匹配
        sorted_rules = rule_registry.get_rules()
        matched_responses: List[str] = []
        matched_ids: List[str] = []

        # 遍历所有规则，检查是否匹配
        for rule in sorted_rules:
            # 使用预编译正则表达式快速匹配，避免每次都编译
            if any(p.search(user_text_lower) for p in rule._compiled_patterns):
                # 处理规则动作，获取模板参数
                context_kwargs = handle_action(rule.action)
                try:
                    # 使用参数格式化响应模板
                    final_resp = rule.response_template.format(**context_kwargs)
                    matched_responses.append(final_resp)
                    matched_ids.append(rule.id)
                except Exception as fmt_err:
                    # 模板格式化失败，记录错误但不影响其他规则
                    log.error(f"规则拦截器模板 [{rule.id}] 格式化失败: {fmt_err}")

        # 没有匹配的规则，放行到图执行
        if not matched_responses:
            return None

        # 意图覆盖率防爆盾：如果文本长度远超规则能覆盖的范围，放行至大模型
        # 例如：规则"查天气"只能处理短句，长句"帮我想查一下明天北京、上海、深圳的天气"放行
        chars_per_intent = GRAPH_RUNNER_TUNING.chars_per_intent
        estimated_covered_len = len(matched_responses) * chars_per_intent
        if len(user_text_lower) > estimated_covered_len + 5:
            log.info(
                f"规则部分命中但句子较长（{len(user_text_lower)}字），"
                f"疑似复杂意图，放行至大模型: {user_text_lower[:80]}"
            )
            return None

        # 多规则命中时将回复合并
        # 例如：用户同时问了时间地点，可能命中两个规则
        combined_resp = "\n\n".join(matched_responses)
        ids_str = ", ".join(matched_ids)
        thinking_text = f"⚡ 极速拦截复合意图：命中 [{ids_str}]"
        log.info(f"Pre-Graph 规则拦截生效: [{ids_str}]")
        return thinking_text, combined_resp

    # ------------------------------------------------------------------ #
    #  消息构造                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_input_messages(
            history: List[Dict[str, Any]],
            user_input: str,
            session_context: Optional[Dict[str, Any]] = None,
    ) -> List[BaseMessage]:
        """
        将历史消息字典列表转换为 LangChain BaseMessage 列表。

        构建顺序：
        1. 注入当前时间系统消息（避免相对日期理解偏差）。
        2. 注入会话上下文摘要（城市/用户画像）。
        3. 追加历史对话轮次（经过压缩裁剪）。
        4. 追加当前用户输入。
        """
        messages: List[BaseMessage] = [SystemMessage(content=get_agent_date_context())]

        # 注入会话结构化上下文
        context_msg_text = GraphRunner._build_session_context_message(session_context or {})
        if context_msg_text:
            messages.append(SystemMessage(content=context_msg_text))

        # 将历史字典列表转换为 LangChain 消息对象
        raw_history: List[BaseMessage] = []
        for msg in history:
            if msg.get("user_content"):
                raw_history.append(HumanMessage(content=msg["user_content"]))
            if msg.get("model_content"):
                raw_history.append(AIMessage(content=msg["model_content"], name=msg.get("name")))

        # 压缩历史，避免 token 溢出
        compressed_history = compress_history_messages(
            raw_history,
            model=None,
            max_tokens=AGENT_LOOP_CONFIG.context_compress_max_tokens,
            max_chars=AGENT_LOOP_CONFIG.context_compress_max_chars,
        )
        messages.extend(compressed_history)
        messages.append(HumanMessage(content=user_input))
        return messages

    @staticmethod
    def _build_session_context_message(session_context: Dict[str, Any]) -> str:
        """
        将结构化会话槽位渲染为简洁系统提示，减少大模型反复追问城市/画像。

        优先使用由 session_state_service 生成的摘要文本（语义稳定、长度可控），
        仅在摘要缺失时从槽位构造最小提示。
        """
        # 摘要优先：语义稳定、长度可控
        summary_text = (session_context.get("context_summary") or "").strip()
        if summary_text:
            return summary_text

        # 摘要不存在时，从槽位构造最小提示
        slots = session_context.get("context_slots") or {}
        if not isinstance(slots, dict):
            return ""

        fragments: List[str] = []
        city_val = str(slots.get("city", "") or "").strip()
        name_val = str(slots.get("name", "") or "").strip()
        age_val = slots.get("age")
        gender_val = str(slots.get("gender", "") or "").strip()
        height_val = slots.get("height_cm")
        weight_val = slots.get("weight_kg")

        if city_val:
            fragments.append(f"当前城市: {city_val}")
        if name_val:
            fragments.append(f"用户姓名: {name_val}")
        if isinstance(age_val, int):
            fragments.append(f"年龄: {age_val}岁")
        if gender_val:
            label = "男" if gender_val.lower() == "male" else "女" if gender_val.lower() == "female" else gender_val
            fragments.append(f"性别: {label}")
        if isinstance(height_val, int):
            fragments.append(f"身高: {height_val}cm")
        if isinstance(weight_val, (int, float)):
            fragments.append(f"体重: {float(weight_val):.1f}kg")

        if not fragments:
            return ""
        return "【会话关键上下文】\n- " + "\n- ".join(fragments)

    # ------------------------------------------------------------------ #
    #  RAG 上下文注入                                                       #
    # ------------------------------------------------------------------ #

    def _inject_rag_context(
            self,
            messages: List[BaseMessage],
            user_input: str,
            model_config: Dict[str, Any],
    ) -> str:
        """
        检索 RAG 知识库并将结果注入消息列表。

        RAG 未启用、或检索结果为空时静默跳过，不影响主流程。

        Returns:
            RAG 命中时返回思考提示文本，未命中返回空字符串。
        """
        rag_enabled = model_config.get("rag_enabled", False)
        # 兼容字符串类型的布尔值（如来自环境变量的 "true"/"false"）
        if isinstance(rag_enabled, str):
            rag_enabled = rag_enabled.lower() == "true"

        if not rag_enabled:
            return ""

        try:
            rag_context, rag_sources = self._retrieve_rag_context(user_input, model_config)
        except Exception as rag_err:
            log.warning(f"RAG 检索失败，已降级跳过: {rag_err}")
            return ""

        if not rag_context:
            return ""

        # 在最后一条 HumanMessage 之前插入 RAG 上下文
        insert_idx = len(messages) - 1 if isinstance(messages[-1], HumanMessage) else len(messages)
        messages.insert(insert_idx, SystemMessage(content=f"参考以下知识库回答:\n{rag_context}"))
        return f"RAG 命中 {len(rag_sources)} 个来源"

    def _retrieve_rag_context(
            self,
            user_input: str,
            model_config: Dict[str, Any],
    ) -> Tuple[str, List[str]]:
        """
        实际执行 RAG 向量检索，返回拼接后的上下文文本和来源列表。

        失败时抛出异常由调用方统一处理，保证降级策略可控。
        """
        docs = vector_store_service.search_documents(
            user_input,
            threshold=float(model_config.get("similarity_threshold", 0.7)),
        )
        sources: List[str] = []
        for doc in docs:
            metadata = getattr(doc, "metadata", None)
            if isinstance(metadata, dict):
                source_val = metadata.get("source")
                if source_val:
                    sources.append(str(source_val))
        return vector_store_service.get_context(docs), sources

    # ------------------------------------------------------------------ #
    #  图执行核心：后台线程 + 事件队列消费                                   #
    # ------------------------------------------------------------------ #

    async def _run_graph_stream_async(
            self,
            graph: Any,
            graph_inputs: Dict[str, Any],
            graph_config: Dict[str, Any],
            run_context,
            effective_config: Dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """
        启动后台图执行线程，通过 asyncio.Queue 异步消费事件，将所有事件转为 SSE 推送。

        【异步设计要点】
        - event_queue 使用 asyncio.Queue，生产者线程通过 loop.call_soon_threadsafe 写入，
          消费者协程用 await queue.get() 非阻塞取数，全程不占用事件循环线程。
        - owner_thread_ids 追踪参与执行的线程 ID，日志拦截器用它过滤跨会话串流。
        - 所有 finally 清理动作（取消注册回调、join worker）均在协程退出时执行。
        """
        session_id = run_context.session_id
        run_id = run_context.run_id
        loop = asyncio.get_event_loop()
        # asyncio.Queue：生产者线程通过 call_soon_threadsafe 写入，消费协程 await get()
        event_queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        # 记录本轮已实时推流的 Agent，用于抑制 synthetic 重复输出
        live_streamed_agents: Set[str] = set()
        # 跨 chunk 的路由 JSON 前缀过滤状态
        router_prefix_buffer = ""
        router_prefix_done = False
        bridge = _GraphStreamAsyncBridge(
            runner=self,
            loop=loop,
            event_queue=event_queue,
            run_context=run_context,
            graph=graph,
            graph_inputs=graph_inputs,
            graph_config=graph_config,
        )

        # 注册日志拦截回调
        CustomLogger.add_global_sse_callback(bridge.log_interceptor)
        # 注册子 Agent 实时流回调（可配置开关）
        runtime_session_manager.register_live_stream_callback(
            run_context,
            bridge.live_stream_interceptor,
            enabled=AGENT_LIVE_STREAM_ENABLED,
        )
        runtime_session_manager.mark_running(
            run_context,
            phase="graph_stream_started",
            summary="图执行线程已启动",
            title="总管开始调度流程",
        )

        # 启动后台图执行线程
        worker_thread = threading.Thread(target=bridge.graph_worker, daemon=True, name=f"graph-worker-{run_id[:8]}")
        worker_thread.start()
        if worker_thread.ident is not None:
            bridge.owner_thread_ids.add(worker_thread.ident)

        try:
            async for chunk in self._consume_event_queue_async(
                    event_queue=event_queue,
                    worker_thread=worker_thread,
                    run_context=run_context,
                    effective_config=effective_config,
                    graph=graph,
                    live_streamed_agents=live_streamed_agents,
                    router_prefix_buffer=router_prefix_buffer,
                    router_prefix_done=router_prefix_done,
            ):
                yield chunk
        except GeneratorExit:
            runtime_session_manager.cancel_run(run_context, summary="客户端断开，运行已取消")
            log.warning(f"客户端断开连接，已触发取消。session_id={session_id}, run_id={run_id}")
            raise
        finally:
            runtime_session_manager.unregister_live_stream_callback(
                run_context,
                enabled=AGENT_LIVE_STREAM_ENABLED,
            )
            CustomLogger.remove_global_sse_callback(bridge.log_interceptor)
            if worker_thread.is_alive():
                request_cancellation_service.cancel_request(run_id)
                worker_thread.join(timeout=1.0)
            if worker_thread.is_alive():
                log.warning("Graph Worker 线程未能在超时内退出，主流程继续。")
            runtime_session_manager.cleanup_run(run_context)

    # ------------------------------------------------------------------ #
    #  事件队列消费器                                                        #
    # ------------------------------------------------------------------ #

    async def _consume_event_queue_async(
            self,
            event_queue: asyncio.Queue,
            worker_thread: threading.Thread,
            run_context,
            effective_config: Dict[str, Any],
            graph: Any,
            live_streamed_agents: Set[str],
            router_prefix_buffer: str,
            router_prefix_done: bool,
    ) -> AsyncGenerator[str, None]:
        """
        从 asyncio.Queue 中异步消费所有事件并转换为 SSE 字符串。

        超时保护机制：
        - 空闲超时（idle_timeout）：长时间无事件时触发，可配置开关。
        - 硬超时（hard_timeout）：整体执行时长上限，防止无限等待。
        - 心跳（heartbeat）：在无正文输出时定期发送，避免前端误以为卡住。
        """
        session_id = run_context.session_id
        run_id = run_context.run_id
        start_ts = time.time()
        last_event_ts = start_ts
        last_heartbeat_ts = 0.0

        # 流程状态标志
        interrupt_emitted = False
        error_emitted = False
        stream_content_emitted = False
        progress_emitted = False

        # 记录本轮真正参与执行的 Agent，用于定向中断扫描
        active_agent_candidates: Set[str] = set()

        # 从配置读取超时参数，便于运行时调参
        idle_heartbeat_sec = GRAPH_RUNNER_TUNING.idle_heartbeat_sec
        idle_timeout_sec = GRAPH_RUNNER_TUNING.idle_timeout_sec
        idle_timeout_enabled = GRAPH_RUNNER_TUNING.idle_timeout_enabled
        hard_timeout_sec = GRAPH_RUNNER_TUNING.hard_timeout_sec
        poll_timeout_sec = GRAPH_RUNNER_TUNING.queue_poll_timeout_sec

        while True:
            now = time.time()

            # ── 总执行硬超时保护 ────────────────────────────────────────
            if (now - start_ts > hard_timeout_sec) and (now - last_event_ts >= idle_timeout_sec):
                error_emitted = True
                request_cancellation_service.cancel_request(run_id)
                runtime_session_manager.mark_failed(
                    run_context,
                    phase="runtime_timeout",
                    summary=SseMessage.ERROR_TIMEOUT,
                    title="运行超时",
                    error=SseMessage.ERROR_TIMEOUT,
                )
                yield self._fmt_sse(SseEventType.ERROR.value, SseMessage.ERROR_TIMEOUT)
                break

            # ── 从异步队列获取事件（非阻塞，带超时） ────────────────────
            try:
                item_type, item_data = await asyncio.wait_for(
                    event_queue.get(), timeout=poll_timeout_sec
                )
            except asyncio.TimeoutError:
                now = time.time()
                if worker_thread.is_alive():
                    # 可配置的空闲超时
                    if idle_timeout_enabled and (now - last_event_ts >= idle_timeout_sec):
                        error_emitted = True
                        request_cancellation_service.cancel_request(run_id)
                        runtime_session_manager.mark_failed(
                            run_context,
                            phase="runtime_idle_timeout",
                            summary=SseMessage.ERROR_IDLE_TIMEOUT,
                            title="运行空闲超时",
                            error=SseMessage.ERROR_IDLE_TIMEOUT,
                        )
                        yield self._fmt_sse(SseEventType.ERROR.value, SseMessage.ERROR_IDLE_TIMEOUT)
                        break
                    # 在等待首包期间定期发送心跳，避免前端误判为服务无响应
                    waiting_first_output = (
                            not stream_content_emitted
                            and not interrupt_emitted
                            and not error_emitted
                    )
                    if waiting_first_output and (now - last_heartbeat_ts >= idle_heartbeat_sec):
                        yield self._fmt_sse(
                            SseEventType.THINKING.value,
                            f"⏱️ 正在等待模型首包返回（已等待 {int(now - start_ts)}s）",
                        )
                        last_heartbeat_ts = now
                    continue

                # Worker 已退出但没有发送 DONE/ERROR，兜底防止协程永久阻塞
                if not error_emitted:
                    error_emitted = True
                    yield self._fmt_sse(SseEventType.ERROR.value, SseMessage.ERROR_TASK_INTERRUPTED)
                break

            # 更新最后事件时间戳
            last_event_ts = time.time()

            # ── 处理各类事件 ────────────────────────────────────────────
            if item_type == GraphQueueItemType.DONE.value:
                break

            elif item_type == GraphQueueItemType.ERROR.value:
                error_emitted = True
                request_cancellation_service.cancel_request(run_id)
                runtime_session_manager.mark_failed(
                    run_context,
                    phase="graph_worker_error",
                    summary=str(item_data or "运行失败"),
                    title="运行失败",
                    error=str(item_data or "运行失败"),
                )
                yield self._fmt_sse(SseEventType.ERROR.value, item_data)
                break

            elif item_type == GraphQueueItemType.LOG.value:
                progress_emitted = True
                yield item_data

            elif item_type == GraphQueueItemType.LIVE_STREAM.value:
                for sse_chunk in self._handle_live_stream_event(
                        payload=item_data,
                        live_streamed_agents=live_streamed_agents,
                        session_id=session_id,
                        run_id=run_id,
                ):
                    if not sse_chunk:
                        continue
                    progress_emitted = True
                    if sse_chunk.startswith(f"event: {SseEventType.STREAM.value}"):
                        stream_content_emitted = True
                    runtime_session_manager.record_workflow_event_chunk(run_context, sse_chunk)
                    yield sse_chunk

            elif item_type == GraphQueueItemType.GRAPH.value:
                ev_type, ev = item_data

                if ev_type == "messages":
                    sse_chunks, router_prefix_buffer, router_prefix_done = self._handle_message_chunk(
                        ev, router_prefix_buffer, router_prefix_done, session_id, run_id
                    )
                    for sse_chunk in sse_chunks:
                        if sse_chunk:
                            progress_emitted = True
                            stream_content_emitted = True
                            yield sse_chunk

                elif ev_type == "updates":
                    for sse_chunk in self._handle_updates_event(
                            ev,
                            session_id=session_id,
                            effective_config=effective_config,
                            live_streamed_agents=live_streamed_agents,
                            interrupt_emitted=interrupt_emitted,
                            active_agent_candidates=active_agent_candidates,
                            run_id=run_id,
                    ):
                        progress_emitted = True
                        if sse_chunk.startswith(f"event: {SseEventType.INTERRUPT.value}"):
                            interrupt_emitted = True
                            runtime_session_manager.mark_interrupted(
                                run_context,
                                phase="approval_pending",
                                summary="运行等待审批结果",
                                title="运行等待审批",
                            )
                        if sse_chunk.startswith(f"event: {SseEventType.ERROR.value}"):
                            error_emitted = True
                        if sse_chunk.startswith(f"event: {SseEventType.STREAM.value}"):
                            stream_content_emitted = True
                        runtime_session_manager.record_workflow_event_chunk(run_context, sse_chunk)
                        yield sse_chunk
                    if error_emitted:
                        request_cancellation_service.cancel_request(run_id)
                        break

        # ── 循环结束后的兜底处理 ─────────────────────────────────────────

        # 可选：流结束后定向扫描子图中断（默认关闭）
        if (
                not interrupt_emitted
                and GRAPH_RUNNER_TUNING.post_run_interrupt_scan_enabled
                and active_agent_candidates
        ):
            for chunk in self._try_post_run_interrupt_scan(
                    session_id=session_id,
                    effective_config=effective_config,
                    candidate_names=list(active_agent_candidates),
                    run_id=run_id,
            ):
                yield chunk

        # 若图执行结束但没有任何正文输出，尝试从最终状态回捞答案
        if not interrupt_emitted and not stream_content_emitted and not error_emitted:
            fallback_text = self._extract_final_answer_from_state(graph, session_id)
            if fallback_text:
                stream_content_emitted = True
                yield self._fmt_sse(SseEventType.STREAM.value, fallback_text)
            else:
                error_emitted = True
                runtime_session_manager.mark_failed(
                    run_context,
                    phase="missing_final_response",
                    summary="本轮未生成可展示的最终答复，请重试。",
                    title="未生成最终答复",
                    error="missing_final_response",
                )
                yield self._fmt_sse(SseEventType.ERROR.value, "本轮未生成可展示的最终答复，请重试。")

        final_answer = ""
        if not interrupt_emitted and not error_emitted:
            final_answer = self._extract_final_answer_from_state(graph, session_id)

        if interrupt_emitted and not error_emitted:
            runtime_session_manager.mark_interrupted(
                run_context,
                phase="awaiting_approval",
                summary="运行已暂停，等待老板批示",
                title="运行暂停等待审批",
            )
        elif not error_emitted:
            runtime_session_manager.mark_completed(
                run_context,
                phase="response_completed",
                summary="运行已完成",
                title="运行完成",
            )

        post_run_hooks = runtime_hook_manager.run_post_run_hooks(run_context, final_text=final_answer)
        if final_answer:
            workspace_manager.write_text_artifact(
                run_context,
                name="final_response.md",
                content=final_answer,
                category="response",
                media_type="text/markdown",
            )
        runtime_session_manager.attach_meta(
            run_context,
            post_run_hooks=[hook.to_dict() for hook in post_run_hooks],
            artifacts=workspace_manager.list_artifacts(run_context),
        )
        for hook in post_run_hooks:
            hook_payload = hook.to_dict()
            yield self._fmt_workflow_event(
                self._build_workflow_event(
                    session_id=session_id,
                    run_id=run_id,
                    phase="hook_finalized",
                    title=f"收尾检查: {hook_payload['name']}",
                    summary=hook_payload["summary"],
                    status=hook_payload["status"],
                    role="system",
                    meta={"hook": hook_payload},
                )
            )
        yield self._fmt_workflow_event(
            self._build_workflow_event(
                session_id=session_id,
                run_id=run_id,
                phase="artifacts_indexed",
                title="本轮卷宗归档完成",
                summary=f"共归档 {len(workspace_manager.list_artifacts(run_context))} 份运行材料",
                status="completed" if not error_emitted else "info",
                role="system",
                meta={"artifacts": workspace_manager.list_artifacts(run_context)},
            )
        )

        yield self._fmt_sse(SseEventType.RESPONSE_END.value, "")

    # ------------------------------------------------------------------ #
    #  事件处理子方法                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def _handle_live_stream_event(
            cls,
            payload: Any,
            live_streamed_agents: Set[str],
            session_id: str,
            run_id: str,
    ) -> List[str]:
        """
        处理子 Agent 实时正文流事件。

        记录已推流的 Agent 名称，供后续抑制 synthetic 重复输出。
        """
        if not isinstance(payload, dict):
            return []
        visible_content = str(payload.get("content") or "").strip()
        if not visible_content:
            return []
        source_agent = str(payload.get("agent_name") or "").strip()
        role = cls._workflow_role_for_agent(source_agent)
        phase = "direct_response_streaming" if role == "supervisor" else "worker_streaming"
        title = (
            f"{cls._workflow_display_name(source_agent)}正在直接回话"
            if role == "supervisor"
            else f"{cls._workflow_display_name(source_agent)}正在执行"
        )
        first_stream = bool(source_agent and source_agent not in live_streamed_agents)
        if source_agent:
            live_streamed_agents.add(source_agent)
        workflow_chunks = []
        if source_agent:
            workflow_chunks.append(
                cls._fmt_workflow_event(
                    cls._build_workflow_event(
                        session_id=session_id,
                        run_id=run_id,
                        phase=phase,
                        title=title,
                        summary=visible_content[:80],
                        status="active",
                        role=role,
                        agent_name=source_agent,
                        task_id=str(payload.get("task_id") or ""),
                        meta={
                            "preview": visible_content[:160],
                            "first_stream": first_stream,
                        },
                    )
                )
            )
        workflow_chunks.append(cls._fmt_sse(SseEventType.STREAM.value, visible_content))
        return workflow_chunks

    def _handle_message_chunk(
            self,
            event: Tuple[Any, Any],
            router_prefix_buffer: str,
            router_prefix_done: bool,
            session_id: str = "",
            run_id: str = "",
    ) -> Tuple[List[str], str, bool]:
        """
        处理 messages 模式下的流式 token chunk。

        过滤逻辑：
        - 静默内部路由/规划节点的文本输出，只允许最终节点往前端推流。
        - 过滤工具调用 chunk（工具调用通过 THINKING 事件单独展示）。
        - 跨 chunk 去掉路由 JSON 前缀，避免 `{"intent":...}` 泄漏到用户界面。
        """
        msg_chunk, metadata = event
        if not isinstance(msg_chunk, AIMessageChunk):
            return [], router_prefix_buffer, router_prefix_done

        node_name = metadata.get("langgraph_node", "")

        # 这些节点产生的文本是内部路由决策，不应该展示给用户
        _silenced_nodes = {
            "Domain_Router_Node",
            "Rule_Engine_Node",
            "Intent_Router_Node",
            "Parent_Planner_Node",
            "executor_node",
            "dispatcher_node",
            "worker_node",
            "reducer_node",
            "reflection_node",
        }

        if node_name in _silenced_nodes:
            return [], router_prefix_buffer, router_prefix_done

        # 工具调用 chunk 展示，同时下发 WORKFLOW_EVENT 供前端详情面板展示
        if getattr(msg_chunk, "tool_calls", None):
            chunks = []
            for tc in msg_chunk.tool_calls:
                tool_name = tc.get("name", "...")
                chunks.append(self._fmt_sse(SseEventType.THINKING.value, f"🔧 正在调度: {tool_name}"))
                chunks.append(self._fmt_workflow_event(
                    self._build_workflow_event(
                        session_id=session_id,
                        run_id=run_id,
                        phase="tool_called",
                        title=f"调用工具: {tool_name}",
                        summary=json.dumps(tc.get("args", {}), ensure_ascii=False),
                        status="active",
                        role=self._workflow_role_for_agent(node_name),
                        agent_name=node_name,
                        node_name=node_name,
                        meta={"args": tc.get("args", {}), "tool_call_id": tc.get("id")}
                    )
                ))
            if chunks:
                return chunks, router_prefix_buffer, router_prefix_done
            return [], router_prefix_buffer, router_prefix_done

        if not msg_chunk.content:
            return [], router_prefix_buffer, router_prefix_done

        # 只允许主路径与专业 Agent 的正文流推送
        from supervisor.registry import agent_classes as _agent_classes
        allowed_names = {None, "", "ChatAgent", "Aggregator", *_agent_classes.keys()}
        if getattr(msg_chunk, "name", None) not in allowed_names:
            return [], router_prefix_buffer, router_prefix_done

        # 单 chunk 路由 JSON 过滤
        visible = self._strip_router_json_prefix_single(msg_chunk.content)
        # 跨 chunk 路由 JSON 前缀过滤
        visible, router_prefix_buffer, router_prefix_done = self._strip_router_json_prefix_cross_chunk(
            visible, router_prefix_buffer, router_prefix_done
        )

        if not isinstance(visible, str) or not visible:
            return [], router_prefix_buffer, router_prefix_done

        return [self._fmt_sse(SseEventType.STREAM.value, visible)], router_prefix_buffer, router_prefix_done

    # ------------------------------------------------------------------ #
    #  SSE 格式化工具                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def _fmt_sse(cls, event_type: str, content: str) -> str:
        """
        将事件类型和内容格式化为标准 SSE 字符串。

        格式：
            event: <event_type>\n
            data: {"type": "<event_type>", "content": "<content>"}\n\n
        """
        visible_content = str(content or "")
        if event_type == SseEventType.STREAM.value:
            visible_content = cls._format_user_visible_text(visible_content)

        payload = json.dumps(
            {SsePayloadField.TYPE.value: event_type, SsePayloadField.CONTENT.value: visible_content},
            ensure_ascii=False,
        )
        return f"event: {event_type}\ndata: {payload}\n\n"

    @staticmethod
    def _fmt_workflow_event(payload: Dict[str, Any]) -> str:
        """将结构化流程负载格式化为独立 SSE 事件。"""
        body = json.dumps(
            {
                SsePayloadField.TYPE.value: SseEventType.WORKFLOW_EVENT.value,
                SsePayloadField.CONTENT.value: str(payload.get("summary") or payload.get("title") or ""),
                SsePayloadField.PAYLOAD.value: payload,
            },
            ensure_ascii=False,
        )
        return f"event: {SseEventType.WORKFLOW_EVENT.value}\ndata: {body}\n\n"

    @staticmethod
    def _normalize_graph_error_message(error: Any) -> str:
        """
        将内部异常归一为稳定的用户可见错误文案。

        优先读取节点显式提供的 `user_message`，否则按常见错误类型降级。
        """
        user_message = str(getattr(error, "user_message", "") or "").strip()
        if user_message:
            return user_message

        raw = str(error or "").strip()
        lower = raw.lower()
        timeout_markers = (
            "timeout",
            "timed out",
            "readtimeout",
            "chat_node.first_token_timeout",
            "chat_node.total_timeout",
            "超时",
        )
        connection_markers = (
            "connection",
            "connecterror",
            "connectionerror",
            "apiconnectionerror",
            "连接失败",
            "连接断开",
        )
        if isinstance(error, TimeoutError) or any(marker in lower for marker in timeout_markers):
            return SseMessage.ERROR_TIMEOUT
        if isinstance(error, ConnectionError) or any(marker in lower for marker in connection_markers):
            return SseMessage.ERROR_CONNECTION
        return SseMessage.ERROR_RUNTIME

    # ------------------------------------------------------------------ #
    #  路由 JSON 前缀过滤                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _strip_router_json_prefix_single(content: Any) -> str:
        """
        单 chunk 路由 JSON 前缀过滤。

        若整个 content 是一个 JSON 对象（路由决策输出），则静默丢弃；
        否则原样返回，保留正常正文。
        """
        if not isinstance(content, str):
            try:
                content = str(content or "")
            except Exception:
                return ""
        stripped = content.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                json.loads(stripped)
                return ""
            except Exception:
                pass
        return content

    @staticmethod
    def _strip_router_json_prefix_cross_chunk(
            content: str,
            buffer: str,
            done: bool,
    ) -> Tuple[str, str, bool]:
        """
        跨 chunk 路由 JSON 前缀过滤。

        若流式 token 的前几个 chunk 组合起来是 JSON 对象前缀
        （如 `{"intent":` 开头），则在确认之前暂缓输出；
        确认是 JSON 后静默丢弃，确认不是 JSON 后补发缓冲。

        Args:
            content: 当前 chunk 文本。
            buffer:  跨 chunk 累积缓冲区。
            done:    是否已确认非 JSON 前缀（不再需要过滤）。

        Returns:
            (visible_text, new_buffer, new_done)
        """
        if done:
            return content, buffer, done

        if not content:
            return "", buffer, done

        buffer += content

        # 已经确认不是 JSON 前缀，补发缓冲并标记完成
        if not buffer.lstrip().startswith("{"):
            flushed = buffer
            return flushed, "", True

        # 缓冲区积累超过阈值后，尝试确认是否为完整 JSON
        if len(buffer) > 512:
            try:
                json.loads(buffer.strip())
                # 整段是 JSON，丢弃
                return "", "", True
            except Exception:
                # 不是合法 JSON，补发缓冲
                flushed = buffer
                return flushed, "", True

        # 如果缓冲内已经出现了 JSON 结束符，尝试直接判定
        if buffer.rstrip().endswith("}"):
            try:
                json.loads(buffer.strip())
                return "", "", True
            except Exception:
                pass

        # 继续等待更多 chunk
        return "", buffer, False

    # ------------------------------------------------------------------ #
    #  Interrupt 注册与扫描                                                 #
    # ------------------------------------------------------------------ #

    def _format_interrupt_payload(
            self,
            session_id: str,
            payload: Any,
    ) -> Dict[str, Any]:
        """
        将原始 interrupt payload 规范化为前端期望的统一格式。

        保证字段完整性：message / allowed_decisions / action_requests / agent_name。
        """
        if not isinstance(payload, dict):
            payload = {"message": str(payload)}

        result: Dict[str, Any] = dict(payload)
        result.setdefault("message", DEFAULT_INTERRUPT_MESSAGE)
        result.setdefault("allowed_decisions", list(DEFAULT_ALLOWED_DECISIONS))
        result.setdefault("action_requests", [])
        result.setdefault("agent_name", "")
        result["session_id"] = session_id
        return result

    def _register_interrupts(
            self,
            session_id: str,
            interrupt_event: Dict[str, Any],
            effective_config: Dict[str, Any],
    ) -> None:
        """
        将 interrupt 事件写入 interrupt_service，供前端审批后 resume 使用。

        从 interrupt_event 中提取关键字段，调用 register_pending_approval 写库。
        """
        try:
            message_id = str(interrupt_event.get("message_id") or uuid.uuid4().hex)
            action_requests = interrupt_event.get("action_requests") or []
            # 取第一个 action_request 作为主要审批项
            first_action = action_requests[0] if action_requests else {}
            action_name = str(
                first_action.get("action_name")
                or interrupt_event.get("action_name")
                or "unknown_action"
            )
            action_args = dict(
                first_action.get("action_args")
                or interrupt_event.get("action_args")
                or {}
            )
            description = str(interrupt_event.get("message") or interrupt_event.get("description") or "")
            agent_name = str(interrupt_event.get("agent_name") or "")
            subgraph_thread_id = str(interrupt_event.get("subgraph_thread_id") or "")
            checkpoint_id = interrupt_event.get("checkpoint_id")
            checkpoint_ns = interrupt_event.get("checkpoint_ns")

            # 将 message_id 写回事件，保证前端 resume 时携带正确 ID
            interrupt_event["message_id"] = message_id

            interrupt_service.register_pending_approval(
                session_id=session_id,
                message_id=message_id,
                action_name=action_name,
                action_args=action_args,
                description=description,
                agent_name=agent_name or None,
                subgraph_thread_id=subgraph_thread_id or None,
                checkpoint_id=checkpoint_id,
                checkpoint_ns=checkpoint_ns,
            )
        except Exception as exc:
            log.warning(f"注册 interrupt 失败，已降级跳过: {exc}")

    def _scan_subgraph_interrupts(
            self,
            session_id: str,
            effective_config: Dict[str, Any],
            candidate_agent_names: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        扫描指定（或全量）子 Agent 的 checkpointer 快照，提取最新的 Interrupt 载荷。

        Args:
            session_id:            当前会话 ID。
            effective_config:      模型配置，用于构造 Agent 实例。
            candidate_agent_names: 候选 Agent 名称列表；None 时扫描全量已注册 Agent。

        Returns:
            找到 interrupt 时返回规范化的 payload 字典；未找到返回 None。
        """
        names_to_scan = candidate_agent_names or list(agent_classes.keys())
        if not names_to_scan:
            return None

        try:
            model, _ = create_model_from_config(**effective_config)
        except Exception as exc:
            log.warning(f"扫描子图中断时构造模型失败: {exc}")
            return None

        for name in names_to_scan:
            if name not in agent_classes:
                continue
            try:
                ag = agent_classes[name].cls(
                    AgentRequest(
                        user_input="",
                        model=model,
                        session_id=session_id,
                        subgraph_id=name,
                    )
                )
                snap = self._safe_get_agent_state(ag, name, "post-run 扫描")
                if snap and self._snapshot_has_interrupt(snap):
                    raw = self._extract_interrupt_from_snapshot(snap)
                    if raw:
                        raw["agent_name"] = raw.get("agent_name") or name
                        return self._format_interrupt_payload(session_id, raw)
            except Exception as exc:
                log.warning(f"扫描 Agent [{name}] 快照失败，跳过: {exc}")
                continue
        return None

    @staticmethod
    def _snapshot_has_interrupt(snapshot: Any) -> bool:
        """判断 LangGraph 快照中是否有未处理的 Interrupt。"""
        tasks = getattr(snapshot, "tasks", None) or []
        for task in tasks:
            interrupts = getattr(task, "interrupts", None) or []
            if interrupts:
                return True
        next_nodes = getattr(snapshot, "next", None) or []
        if next_nodes:
            return True
        return False

    @staticmethod
    def _extract_interrupt_from_snapshot(snapshot: Any) -> Optional[Dict[str, Any]]:
        """从 LangGraph 快照中提取第一个 Interrupt 的 payload。"""
        tasks = getattr(snapshot, "tasks", None) or []
        for task in tasks:
            interrupts = getattr(task, "interrupts", None) or []
            if not interrupts:
                continue
            first = interrupts[0]
            value = getattr(first, "value", first)
            if isinstance(value, dict):
                return dict(value)
            return {
                "message": str(value) if value is not None else DEFAULT_INTERRUPT_MESSAGE,
                "allowed_decisions": list(DEFAULT_ALLOWED_DECISIONS),
                "action_requests": [],
            }
        return None

    @staticmethod
    def _safe_get_agent_state(
            agent: Any,
            agent_name: str,
            context: str,
            *,
            thread_id: str = "",
            checkpoint_id: Any = None,
            checkpoint_ns: Any = None,
    ) -> Optional[Any]:
        """安全地获取 Agent 图快照，失败时返回 None 而不抛出异常。"""
        try:
            if thread_id or checkpoint_id or checkpoint_ns:
                config = {
                    "configurable": {
                        "thread_id": thread_id or f"{agent.session_id}_{agent.subgraph_id}"
                    }
                }
                if checkpoint_id:
                    config["configurable"]["checkpoint_id"] = checkpoint_id
                if checkpoint_ns:
                    config["configurable"]["checkpoint_ns"] = checkpoint_ns
                return agent.graph.get_state(config)
            return agent.get_state()
        except Exception as exc:
            log.debug(f"获取 Agent [{agent_name}] 快照失败 ({context}): {exc}")
            return None

    # ------------------------------------------------------------------ #
    #  最终答案回捞                                                         #
    # ------------------------------------------------------------------ #

    def _extract_final_answer_from_state(
            self,
            graph: Any,
            session_id: str,
    ) -> str:
        """
        从 Supervisor 图最终状态中回捞最后一条 AI 消息作为兜底答案。

        用于图执行正常结束但事件队列中没有任何正文输出的场景，
        避免前端白屏无响应。
        """
        try:
            config = {"configurable": {"thread_id": session_id}}
            state = graph.get_state(config)
            messages = (getattr(state, "values", None) or {}).get("messages", [])
            # 从后向前找最后一条有内容的 AI 消息
            for msg in reversed(messages):
                if not isinstance(msg, AIMessage):
                    continue
                # 跳过工具调用消息
                if getattr(msg, "tool_calls", None):
                    continue
                content = msg.content
                if isinstance(content, str) and content.strip():
                    return self._format_user_visible_text(content)
                if isinstance(content, list):
                    parts = [
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content
                        if item
                    ]
                    text = self._format_user_visible_text("".join(parts))
                    if text:
                        return text
        except Exception as exc:
            log.debug(f"回捞最终答案失败: {exc}")
        return ""

    def _handle_updates_event(
            self,
            event: Dict[str, Any],
            session_id: str,
            effective_config: Dict[str, Any],
            live_streamed_agents: Set[str],
            interrupt_emitted: bool,
            active_agent_candidates: Set[str],
            run_id: str = "",
    ) -> Generator[str, None, None]:
        """
        处理 updates 模式下的图状态更新事件。

        包含：
        - 各路由节点的决策日志（domain/intent/planner/worker 耗时等）。
        - in-band Interrupt payload 提取与注册。
        - Agent 节点的操作日志和 synthetic 消息推送。
        """
        from supervisor.registry import agent_classes as _agent_classes

        for node_name, node_val in event.items():
            # 记录本轮参与执行的 Agent 名称
            if node_name in _agent_classes:
                active_agent_candidates.add(node_name)

            # ── 路由节点日志 ────────────────────────────────────────────
            yield from self._emit_router_thinking(
                node_name=node_name,
                node_val=node_val,
                session_id=session_id,
                run_id=run_id,
            )

            # ── Interrupt 提取 ──────────────────────────────────────────
            if not interrupt_emitted:
                inband_interrupt = self._extract_interrupt_from_node_val(node_name, node_val, _agent_classes)
                if inband_interrupt:
                    source_node, payload = inband_interrupt
                    interrupt_event = self._format_interrupt_payload(session_id, payload)
                    interrupt_event.setdefault("subgraph_thread_id", f"{session_id}_{source_node}")
                    interrupt_event.setdefault("agent_name", source_node)
                    self._register_interrupts(session_id, interrupt_event, effective_config)
                    yield self._fmt_workflow_event(
                        self._build_workflow_event(
                            session_id=session_id,
                            run_id=run_id,
                            phase="worker_pending_approval",
                            title=f"{self._workflow_display_name(source_node)}等待老板批示",
                            summary=str(interrupt_event.get("message") or DEFAULT_INTERRUPT_MESSAGE),
                            status="waiting",
                            role=self._workflow_role_for_agent(source_node),
                            agent_name=str(interrupt_event.get("agent_name") or source_node),
                            task_id=str(interrupt_event.get("message_id") or ""),
                            node_name=source_node,
                            meta={"interrupt": interrupt_event},
                        )
                    )
                    yield self._fmt_sse(SseEventType.THINKING.value, SseMessage.NEED_MANUAL_APPROVAL)
                    yield self._fmt_sse(
                        SseEventType.INTERRUPT.value,
                        json.dumps(interrupt_event, ensure_ascii=False),
                    )

            # ── 节点显式错误 ────────────────────────────────────────────
            node_error_message = self._extract_node_error_message(node_val)
            if node_error_message:
                if node_name == "chat_node":
                    yield self._fmt_workflow_event(
                        self._build_workflow_event(
                            session_id=session_id,
                            run_id=run_id,
                            phase="direct_response_failed",
                            title="掌柜回禀失败",
                            summary=node_error_message,
                            status="error",
                            role="supervisor",
                            agent_name="ChatAgent",
                            node_name=node_name,
                        )
                    )
                elif node_name in _agent_classes:
                    yield self._fmt_workflow_event(
                        self._build_workflow_event(
                            session_id=session_id,
                            run_id=run_id,
                            phase="worker_failed",
                            title=f"{self._workflow_display_name(node_name)}执行受阻",
                            summary=node_error_message,
                            status="error",
                            role="worker",
                            agent_name=node_name,
                            node_name=node_name,
                        )
                    )
                yield self._fmt_sse(SseEventType.ERROR.value, node_error_message)
                continue

            # ── Agent 操作日志与 synthetic 消息 ────────────────────────
            if not isinstance(node_val, dict) or "messages" not in node_val:
                continue

            for msg in node_val.get("messages", []):
                from langchain_core.messages import ToolMessage
                if isinstance(msg, ToolMessage):
                    yield self._fmt_workflow_event(
                        self._build_workflow_event(
                            session_id=session_id,
                            run_id=run_id,
                            phase="tool_completed",
                            title=f"工具执行完毕: {msg.name}",
                            summary=str(msg.content)[:120],
                            status="completed",
                            role=self._workflow_role_for_agent(node_name),
                            agent_name=node_name,
                            node_name=node_name,
                            meta={"result": str(msg.content), "tool_call_id": getattr(msg, "tool_call_id", "")}
                        )
                    )
                    continue

                if not isinstance(msg, AIMessage):
                    continue
                metadata = getattr(msg, "response_metadata", {}) or {}
                # 从 response_metadata 提取 Agent 操作日志
                for log_entry in metadata.get("operation_logs", []):
                    yield self._fmt_sse(SseEventType.THINKING.value, log_entry)
                # synthetic 消息：由于未走正常 LLM stream，需直接推送给前端
                should_emit = metadata.get("synthetic") and bool(msg.content)
                source_agent_name = str(getattr(msg, "name", "") or node_name or "")
                visible = self._strip_router_json_prefix_single(msg.content)
                visible_text = visible if isinstance(visible, str) else ""

                if node_name == "aggregator_node" and visible_text.strip():
                    yield self._fmt_workflow_event(
                        self._build_workflow_event(
                            session_id=session_id,
                            run_id=run_id,
                            phase="aggregator_completed",
                            title="总管完成汇总",
                            summary=visible_text[:160],
                            status="completed",
                            role="supervisor",
                            agent_name="Aggregator",
                            node_name=node_name,
                            meta={"force_emit": bool(metadata.get("force_emit"))},
                        )
                    )
                    yield self._fmt_workflow_event(
                        self._build_workflow_event(
                            session_id=session_id,
                            run_id=run_id,
                            phase="final_report_delivered",
                            title="总管向老板回禀",
                            summary=visible_text[:160],
                            status="completed",
                            role="supervisor",
                            agent_name="Aggregator",
                            node_name=node_name,
                        )
                    )
                elif node_name == "chat_node" and visible_text.strip():
                    yield self._fmt_workflow_event(
                        self._build_workflow_event(
                            session_id=session_id,
                            run_id=run_id,
                            phase="direct_response_completed",
                            title="掌柜直接回复老板",
                            summary=visible_text[:160],
                            status="completed",
                            role="supervisor",
                            agent_name="ChatAgent",
                            node_name=node_name,
                        )
                    )
                elif node_name in _agent_classes and visible_text.strip():
                    yield self._fmt_workflow_event(
                        self._build_workflow_event(
                            session_id=session_id,
                            run_id=run_id,
                            phase="worker_completed",
                            title=f"{self._workflow_display_name(node_name)}提交结果",
                            summary=visible_text[:160],
                            status="completed",
                            role="worker",
                            agent_name=node_name,
                            node_name=node_name,
                            meta={"live_streamed": bool(metadata.get("live_streamed"))},
                        )
                    )
                # 仅对 ChatAgent 执行“live_stream 后抑制 synthetic”；
                # 工具型 Agent 可能先流出过渡句，再产出最终结果，若统一抑制会导致“有前奏、无结论”。
                if source_agent_name == "ChatAgent":
                    if metadata.get("live_streamed"):
                        should_emit = False
                    elif live_streamed_agents and source_agent_name in live_streamed_agents:
                        should_emit = False
                if should_emit and (node_name != "aggregator_node" or metadata.get("force_emit")):
                    if visible_text.strip():
                        yield self._fmt_sse(SseEventType.STREAM.value, visible_text)

    def _process_supervisor_event(
            self,
            event: Dict[str, Any],
            live_streamed_agents: Optional[Set[str]] = None,
    ) -> Generator[str, None, None]:
        """
        兼容旧接口：转发到新的 updates 事件处理器。

        旧单测与部分历史调用路径仍引用该方法名，保留适配层可降低重构成本。
        """
        yield from self._handle_updates_event(
            event=event or {},
            session_id="",
            run_id="",
            effective_config={},
            live_streamed_agents=live_streamed_agents or set(),
            interrupt_emitted=False,
            active_agent_candidates=set(),
        )

    @staticmethod
    def _extract_node_error_message(node_val: Any) -> Optional[str]:
        """从节点更新值中提取显式用户错误文案。"""
        if not isinstance(node_val, dict):
            return None
        error_message = str(node_val.get("error_message") or "").strip()
        if error_message:
            return error_message
        return None

    @staticmethod
    def _emit_router_thinking(
            node_name: str,
            node_val: Any,
            session_id: str,
            run_id: str,
    ) -> Generator[str, None, None]:
        """
        为各路由/规划节点生成可读的 thinking 日志事件。

        将原本只存在于服务端日志的路由决策信息（domain/intent/planner/worker 耗时）
        实时推送到前端 thinking 面板，大幅提升链路可观测性。
        """
        if not isinstance(node_val, dict):
            return

        # 数据域路由器日志
        if node_name == "Domain_Router_Node" and "data_domain" in node_val:
            elapsed = node_val.get("domain_elapsed_ms")
            elapsed_text = f"，耗时: {int(elapsed)}ms" if isinstance(elapsed, (int, float)) else ""
            yield GraphRunner._fmt_workflow_event(
                GraphRunner._build_workflow_event(
                    session_id=session_id,
                    run_id=run_id,
                    phase="domain_routed",
                    title="总管完成数据域判断",
                    summary=f"路由至 {node_val.get('data_domain')}",
                    status="completed",
                    role="supervisor",
                    agent_name="ChatAgent",
                    node_name=node_name,
                    meta={
                        "data_domain": node_val.get("data_domain"),
                        "confidence": float(node_val.get("domain_confidence", 0) or 0),
                        "source": node_val.get("domain_route_source", "unknown"),
                        "route_strategy": node_val.get("route_strategy", "single_domain"),
                    },
                )
            )
            yield GraphRunner._fmt_sse(
                SseEventType.THINKING.value,
                f"数据域路由: {node_val.get('data_domain')} "
                f"(置信度: {float(node_val.get('domain_confidence', 0)):.2f}, "
                f"来源: {node_val.get('domain_route_source', 'unknown')}, "
                f"策略: {node_val.get('route_strategy', 'single_domain')}{elapsed_text})",
            )
            return

        # 意图路由器日志
        if node_name == "Intent_Router_Node" and "intent" in node_val:
            elapsed = node_val.get("intent_elapsed_ms")
            elapsed_text = f"，耗时: {int(elapsed)}ms" if isinstance(elapsed, (int, float)) else ""
            intent_name = str(node_val.get("intent") or "CHAT")
            is_complex = bool(node_val.get("is_complex"))
            if is_complex:
                workflow_phase = "complex_route_escalated"
                workflow_title = "总管决定拆解多环节任务"
                workflow_summary = f"改走任务编排，目标意图 {intent_name}"
                workflow_status = "active"
            elif intent_name == "CHAT":
                workflow_phase = "direct_chat_selected"
                workflow_title = "总管决定亲自答复"
                workflow_summary = "无需分派员工，掌柜直接处理"
                workflow_status = "active"
            else:
                workflow_phase = "direct_agent_selected"
                workflow_title = f"总管指派 {GraphRunner._workflow_display_name(intent_name)}"
                workflow_summary = f"目标意图 {intent_name}"
                workflow_status = "active"
            yield GraphRunner._fmt_workflow_event(
                GraphRunner._build_workflow_event(
                    session_id=session_id,
                    run_id=run_id,
                    phase=workflow_phase,
                    title=workflow_title,
                    summary=workflow_summary,
                    status=workflow_status,
                    role="supervisor",
                    agent_name="ChatAgent" if intent_name == "CHAT" else intent_name,
                    node_name=node_name,
                    meta={
                        "intent": intent_name,
                        "confidence": float(node_val.get("intent_confidence", 0) or 0),
                        "is_complex": is_complex,
                    },
                )
            )
            yield GraphRunner._fmt_sse(
                SseEventType.THINKING.value,
                f"智能路由指派: {intent_name} "
                f"(置信度: {node_val.get('intent_confidence', 0):.2f}{elapsed_text})",
            )
            return

        # 规划器日志
        if node_name == "Parent_Planner_Node" and "task_list" in node_val:
            elapsed = node_val.get("planner_elapsed_ms")
            elapsed_text = f"，耗时: {int(elapsed)}ms" if isinstance(elapsed, (int, float)) else ""
            tasks = node_val.get("task_list") or []
            yield GraphRunner._fmt_workflow_event(
                GraphRunner._build_workflow_event(
                    session_id=session_id,
                    run_id=run_id,
                    phase="tasks_planned",
                    title="总管拆解任务",
                    summary=f"共拆解 {len(tasks)} 个子任务",
                    status="completed",
                    role="supervisor",
                    agent_name="ChatAgent",
                    node_name=node_name,
                    meta={
                        "planner_source": node_val.get("planner_source", "unknown"),
                        "tasks": tasks,
                    },
                )
            )
            yield GraphRunner._fmt_sse(
                SseEventType.THINKING.value,
                f"任务拆解完成: {len(tasks)} 个子任务 "
                f"(来源: {node_val.get('planner_source', 'unknown')}{elapsed_text})",
            )
            return

        # DAG 调度器日志
        if node_name == "dispatcher_node" and "active_tasks" in node_val:
            count = len(node_val.get("active_tasks", []))
            wave = node_val.get("current_wave", "?")
            if count > 0:
                for task in node_val.get("active_tasks", []):
                    if not isinstance(task, dict):
                        continue
                    agent_name = str(task.get("agent") or "")
                    yield GraphRunner._fmt_workflow_event(
                        GraphRunner._build_workflow_event(
                            session_id=session_id,
                            run_id=run_id,
                            phase="task_dispatched",
                            title=f"总管派发 {GraphRunner._workflow_display_name(agent_name)}",
                            summary=str(task.get("input") or "")[:120],
                            status="active",
                            role=GraphRunner._workflow_role_for_agent(agent_name),
                            agent_name=agent_name,
                            task_id=str(task.get("id") or ""),
                            node_name=node_name,
                            meta={
                                "wave": wave,
                                "depends_on": task.get("depends_on") or [],
                            },
                        )
                    )
                yield GraphRunner._fmt_sse(
                    SseEventType.THINKING.value,
                    f"🚀 DAG 第 {wave} 波次：派发 {count} 个并行子任务",
                )
            return

        # Worker 子任务耗时日志
        if node_name == "worker_node" and "worker_results" in node_val:
            for worker_item in (node_val.get("worker_results") or []):
                if not isinstance(worker_item, dict):
                    continue
                task_id = str(worker_item.get("task_id") or "")
                agent_name = str(worker_item.get("agent") or "")
                error_text = str(worker_item.get("error") or "").strip()
                result_text = str(worker_item.get("result") or "")
                if error_text:
                    phase = "worker_failed"
                    status = "error"
                    title = f"{GraphRunner._workflow_display_name(agent_name)}执行受阻"
                    summary = error_text
                elif result_text == WORKER_PENDING_APPROVAL_RESULT:
                    phase = "worker_pending_approval"
                    status = "waiting"
                    title = f"{GraphRunner._workflow_display_name(agent_name)}等待老板批示"
                    summary = "该子任务需要人工审批后继续"
                elif result_text == WORKER_CANCELLED_RESULT:
                    phase = "worker_cancelled"
                    status = "cancelled"
                    title = f"{GraphRunner._workflow_display_name(agent_name)}停止执行"
                    summary = "任务已被取消"
                else:
                    phase = "worker_completed"
                    status = "completed"
                    title = f"{GraphRunner._workflow_display_name(agent_name)}完成任务"
                    summary = result_text[:120]
                yield GraphRunner._fmt_workflow_event(
                    GraphRunner._build_workflow_event(
                        session_id=session_id,
                        run_id=run_id,
                        phase=phase,
                        title=title,
                        summary=summary,
                        status=status,
                        role=GraphRunner._workflow_role_for_agent(agent_name),
                        agent_name=agent_name,
                        task_id=task_id,
                        node_name=node_name,
                        meta={
                            "elapsed_ms": worker_item.get("elapsed_ms"),
                            "wave": worker_item.get("wave"),
                        },
                    )
                )
                elapsed = worker_item.get("elapsed_ms")
                if not isinstance(elapsed, (int, float)):
                    continue
                label = f"{task_id}:{agent_name}" if task_id and agent_name else task_id or agent_name or "worker"
                yield GraphRunner._fmt_sse(
                    SseEventType.THINKING.value,
                    f"⏱️ 子任务[{label}]执行耗时: {int(elapsed)}ms",
                )
            return

        if node_name == "reflection_node" and "reflection_source" in node_val:
            reflection_source = str(node_val.get("reflection_source") or "unknown")
            reflection_summary = str(node_val.get("reflection_summary") or "").strip() or reflection_source
            appended_count = 0
            if reflection_source == "llm_appended":
                appended_count = max(0, len(node_val.get("task_list") or []))
            phase = "tasks_reflected" if reflection_source == "llm_appended" else "reflection_converged"
            title = "总管自动复盘追加步骤" if reflection_source == "llm_appended" else "总管复盘后决定收敛"
            yield GraphRunner._fmt_workflow_event(
                GraphRunner._build_workflow_event(
                    session_id=session_id,
                    run_id=run_id,
                    phase=phase,
                    title=title,
                    summary=reflection_summary,
                    status="completed",
                    role="supervisor",
                    agent_name="ChatAgent",
                    node_name=node_name,
                    meta={
                        "reflection_source": reflection_source,
                        "reflection_round": node_val.get("reflection_round"),
                        "appended_count": appended_count,
                    },
                )
            )
            yield GraphRunner._fmt_sse(
                SseEventType.THINKING.value,
                f"执行复盘: {reflection_summary}",
            )

    @staticmethod
    def _extract_interrupt_from_node_val(
            node_name: str,
            node_val: Any,
            agent_classes_map: Dict[str, Any],
    ) -> Optional[Tuple[str, Any]]:
        """
        从节点更新值中提取 in-band interrupt payload。

        Returns:
            命中时返回 (source_node_name, payload)；未命中返回 None。
        """
        if not isinstance(node_val, dict):
            return None
        payload = node_val.get("interrupt_payload") or node_val.get("interrupt")
        if not payload:
            return None
        # 若 payload 中未标注 agent_name，且节点本身是已知 Agent，则自动补充
        if isinstance(payload, dict) and not payload.get("agent_name") and node_name in agent_classes_map:
            payload = dict(payload)
            payload["agent_name"] = node_name
        return node_name, payload

    # ------------------------------------------------------------------ #
    #  后置中断扫描                                                          #
    # ------------------------------------------------------------------ #

    def _try_post_run_interrupt_scan(
            self,
            session_id: str,
            effective_config: Dict[str, Any],
            candidate_names: List[str],
            run_id: str = "",
    ) -> Generator[str, None, None]:
        """流结束后定向扫描子图快照是否存在未处理的 Interrupt（默认关闭）。"""
        try:
            interrupt_event = self._scan_subgraph_interrupts(
                session_id,
                effective_config=effective_config,
                candidate_agent_names=candidate_names,
            )
        except Exception as scan_exc:
            log.warning(f"后置中断扫描失败，已降级跳过: {scan_exc}")
            return
        if interrupt_event:
            self._register_interrupts(session_id, interrupt_event, effective_config)
            agent_name = str(interrupt_event.get("agent_name") or "")
            yield self._fmt_workflow_event(
                self._build_workflow_event(
                    session_id=session_id,
                    run_id=run_id,
                    phase="worker_pending_approval",
                    title=f"{self._workflow_display_name(agent_name)}等待老板批示",
                    summary=str(interrupt_event.get("message") or DEFAULT_INTERRUPT_MESSAGE),
                    status="waiting",
                    role=self._workflow_role_for_agent(agent_name),
                    agent_name=agent_name,
                    task_id=str(interrupt_event.get("message_id") or ""),
                    meta={"interrupt": interrupt_event},
                )
            )
            yield self._fmt_sse(SseEventType.THINKING.value, SseMessage.NEED_MANUAL_APPROVAL)
            yield self._fmt_sse(
                SseEventType.INTERRUPT.value,
                json.dumps(interrupt_event, ensure_ascii=False),
            )

    # ------------------------------------------------------------------ #
    #  审批恢复流程                                                          #
    # ------------------------------------------------------------------ #

    async def _handle_resume_stream_async(
            self,
            run_context,
            graph: Any,
            effective_config: Dict[str, Any],
            emit_response_start: bool,
            resume_message_id: str = "",
    ) -> AsyncGenerator[str, None]:
        """统一处理 [RESUME] 审批恢复流程的 SSE 推送（异步版本）。"""
        session_id = run_context.session_id
        try:
            # 兼容旧单测和旧调用方：未指定 message_id 时仍保持单参数调用语义。
            if resume_message_id:
                resume_meta = self._check_pending_approval(session_id, resume_message_id=resume_message_id)
            else:
                resume_meta = self._check_pending_approval(session_id)
            if not resume_meta:
                runtime_session_manager.mark_failed(
                    run_context,
                    phase="resume_missing_approval",
                    summary=SseMessage.INVALID_RESUME_PARAM,
                    title="审批恢复失败",
                    error=SseMessage.INVALID_RESUME_PARAM,
                )
                yield self._fmt_sse(SseEventType.ERROR.value, SseMessage.INVALID_RESUME_PARAM)
                return
            if emit_response_start:
                yield self._fmt_sse(SseEventType.RESPONSE_START.value, "")
            yield self._fmt_workflow_event(
                self._build_workflow_event(
                    session_id=session_id,
                    run_id=run_context.run_id,
                    phase="resume_requested",
                    title="老板批示已送达",
                    summary=f"审批结果：{resume_meta.get('decision')}",
                    status="completed",
                    role="boss",
                    meta={"message_id": resume_meta.get("message_id")},
                )
            )
            yield self._fmt_sse(SseEventType.THINKING.value, SseMessage.RESUME_DETECTED)
            for chunk in self._handle_resume(
                    session_id=session_id,
                    resume_meta=resume_meta,
                    supervisor_graph=graph,
                    effective_config=effective_config,
                    run_id=run_context.run_id,
            ):
                runtime_session_manager.record_workflow_event_chunk(run_context, chunk)
                yield chunk
            runtime_session_manager.mark_completed(
                run_context,
                phase="resume_completed",
                summary="审批恢复流程执行完成",
                title="审批恢复完成",
            )
            yield self._fmt_sse(SseEventType.RESPONSE_END.value, "")
        finally:
            runtime_session_manager.cleanup_run(run_context)

    def _check_pending_approval(self, session_id: str, resume_message_id: str = "") -> Optional[Dict[str, Any]]:
        """
        查询当前会话是否有前端已提交的审批结果。

        Returns:
            有待恢复审批时返回恢复元信息字典；否则返回 None。
        """
        approval = (
            interrupt_service.fetch_resolved_approval_by_message(session_id, resume_message_id)
            if resume_message_id
            else interrupt_service.fetch_latest_resolved_approval(session_id)
        )
        if not approval:
            return None
        decision = (
            ApprovalDecision.APPROVE.value
            if approval.get("status") == ApprovalStatus.APPROVE.value
            else ApprovalDecision.REJECT.value
        )
        return {
            "message_id": approval.get("message_id"),
            "decision": decision,
            "agent_name": approval.get("agent_name"),
            "action_name": approval.get("action_name"),
            "action_args": approval.get("action_args", {}),
            "subgraph_thread_id": approval.get("subgraph_thread_id"),
            "checkpoint_id": approval.get("checkpoint_id"),
            "checkpoint_ns": approval.get("checkpoint_ns"),
            "command": Command(resume=decision),
        }

    def _handle_resume(
            self,
            session_id: str,
            resume_meta: Dict[str, Any],
            supervisor_graph: Any,
            effective_config: Dict[str, Any],
            run_id: str = "",
    ) -> Generator[str, None, None]:
        """
        恢复被 Interrupt 挂起的子 Agent 执行。

        流程：
        1. 优先按 resume_meta 中记录的 agent_name 直接定位目标子图。
        2. 若未找到，全量扫描所有 Agent 的挂起快照。
        3. 向目标子图发送 Command(resume=decision) 恢复执行。
        4. 将恢复结果回填到 Supervisor 状态中。
        """
        command = resume_meta["command"]
        preferred_agent = resume_meta.get("agent_name")
        message_id = resume_meta.get("message_id")
        decision = resume_meta.get("decision")
        preferred_thread_id = resume_meta.get("subgraph_thread_id")
        preferred_ckpt_id = resume_meta.get("checkpoint_id")
        preferred_ckpt_ns = resume_meta.get("checkpoint_ns")

        model, _ = create_model_from_config(**effective_config)
        target_agent = None

        # 优先按记录的 agent_name 定位目标子图
        if preferred_agent and preferred_agent in agent_classes:
            candidate = agent_classes[preferred_agent].cls(
                AgentRequest(user_input="", model=model, session_id=session_id, subgraph_id=preferred_agent)
            )
            if preferred_ckpt_id:
                target_agent = candidate
            else:
                snap = self._safe_get_agent_state(
                    candidate,
                    preferred_agent,
                    "恢复前定位",
                    thread_id=preferred_thread_id,
                    checkpoint_id=preferred_ckpt_id,
                    checkpoint_ns=preferred_ckpt_ns,
                )
                if snap and self._snapshot_has_interrupt(snap):
                    target_agent = candidate

        # 全量扫描挂起状态
        if not target_agent:
            for name, info in agent_classes.items():
                ag = info.cls(AgentRequest(user_input="", model=model, session_id=session_id, subgraph_id=name))
                snap = self._safe_get_agent_state(ag, name, "恢复前扫描")
                if snap and self._snapshot_has_interrupt(snap):
                    target_agent = ag
                    break

        if not target_agent:
            # 兜底：审批 SQL 执行时直接执行已审批 SQL
            if decision == ApprovalDecision.APPROVE.value and resume_meta.get(
                    "action_name") == SQL_APPROVAL_ACTION_NAME:
                approved_sql = (resume_meta.get("action_args") or {}).get("sql")
                if approved_sql:
                    try:
                        from tools.agent_tools.sql_tools import execute_sql, format_sql_result_for_user
                        result = execute_sql(approved_sql, domain="LOCAL_DB")
                        formatted = format_sql_result_for_user(approved_sql, result)
                        yield self._fmt_sse(SseEventType.STREAM.value, formatted)
                        if message_id:
                            interrupt_service.mark_approval_consumed(session_id, message_id)
                        return
                    except Exception as sql_exc:
                        yield self._fmt_sse(SseEventType.ERROR.value, f"未找到中断任务且 SQL 兜底执行失败: {sql_exc}")
                        return
            yield self._fmt_sse(SseEventType.ERROR.value, SseMessage.ERROR_NO_INTERRUPTED_TASK)
            return

        # 构造子图 Config 并发送 Command 恢复执行
        subgraph_thread_id = preferred_thread_id or f"{session_id}_{target_agent.subgraph_id}"
        subgraph_config: Dict[str, Any] = {"configurable": {"thread_id": subgraph_thread_id}}
        if preferred_ckpt_id:
            subgraph_config["configurable"]["checkpoint_id"] = preferred_ckpt_id
        if preferred_ckpt_ns:
            subgraph_config["configurable"]["checkpoint_ns"] = preferred_ckpt_ns

        log.info(
            f"Resume target={target_agent.subgraph_id}, thread={subgraph_thread_id}, "
            f"ckpt={preferred_ckpt_id}, decision={decision}"
        )
        yield self._fmt_workflow_event(
            self._build_workflow_event(
                session_id=session_id,
                run_id=run_id,
                phase="worker_resumed",
                title=f"{self._workflow_display_name(target_agent.subgraph_id)}恢复执行",
                summary=f"审批结果：{decision}",
                status="active",
                role=self._workflow_role_for_agent(target_agent.subgraph_id),
                agent_name=target_agent.subgraph_id,
                meta={"message_id": message_id},
            )
        )

        final_msgs: List[AIMessage] = []
        try:
            for event in target_agent.graph.stream(command, config=subgraph_config, stream_mode="updates"):
                for _, val in event.items():
                    if "messages" in val:
                        for msg in val["messages"]:
                            if isinstance(msg, AIMessage):
                                final_msgs.append(msg)
                                yield self._fmt_sse(SseEventType.STREAM.value, msg.content)
        except Exception as resume_exc:
            yield from self._handle_resume_exception(
                session_id=session_id, resume_meta=resume_meta, resume_exc=resume_exc,
                target_agent=target_agent, effective_config=effective_config,
                message_id=message_id, decision=decision,
            )
            return

        if not final_msgs:
            yield from self._handle_resume_empty_result(
                session_id=session_id, decision=decision, message_id=message_id,
                target_agent=target_agent, effective_config=effective_config,
            )
            return

        if message_id:
            interrupt_service.mark_approval_consumed(session_id, message_id)
        self._backfill_supervisor_state(supervisor_graph, session_id, final_msgs)
        final_text = next(
            (
                str(msg.content or "").strip()
                for msg in reversed(final_msgs)
                if isinstance(msg.content, str) and str(msg.content or "").strip()
            ),
            "",
        )
        if final_text:
            yield self._fmt_workflow_event(
                self._build_workflow_event(
                    session_id=session_id,
                    run_id=run_id,
                    phase="final_report_delivered",
                    title="恢复执行后已向老板回禀",
                    summary=final_text[:160],
                    status="completed",
                    role=self._workflow_role_for_agent(target_agent.subgraph_id),
                    agent_name=target_agent.subgraph_id,
                )
            )

    def _handle_resume_exception(
            self,
            session_id: str,
            resume_meta: Dict[str, Any],
            resume_exc: Exception,
            target_agent: Any,
            effective_config: Dict[str, Any],
            message_id: Optional[str],
            decision: Optional[str],
    ) -> Generator[str, None, None]:
        """处理恢复执行时抛出的异常，区分 Interrupt 挂起和真实错误。"""
        err_msg = str(resume_exc)
        if "Interrupt(" in err_msg or resume_exc.__class__.__name__ == "GraphInterrupt":
            interrupt_event = self._scan_subgraph_interrupts(
                session_id, effective_config=effective_config,
                candidate_agent_names=[target_agent.subgraph_id],
            )
            if interrupt_event:
                if message_id:
                    interrupt_service.mark_approval_consumed(session_id, message_id)
                self._register_interrupts(session_id, interrupt_event, effective_config)
                yield self._fmt_sse(SseEventType.THINKING.value, SseMessage.RESUME_NEED_MANUAL_APPROVAL)
                yield self._fmt_sse(SseEventType.INTERRUPT.value, json.dumps(interrupt_event, ensure_ascii=False))
            else:
                yield self._fmt_sse(SseEventType.ERROR.value, "恢复执行进入审批状态，但未获取到审批详情，请重试。")
            return

        log.exception(f"Resume non-interrupt failure: {err_msg}")
        # 兜底：部分模型在 resume 场景报错时，直接执行已审批 SQL
        if (
                "1214" in err_msg
                and resume_meta.get("action_name") == SQL_APPROVAL_ACTION_NAME
                and decision == ApprovalDecision.APPROVE.value
        ):
            approved_sql = (resume_meta.get("action_args") or {}).get("sql")
            if approved_sql:
                try:
                    from tools.agent_tools.sql_tools import execute_sql, format_sql_result_for_user
                    result = execute_sql(approved_sql, domain="LOCAL_DB")
                    formatted = format_sql_result_for_user(approved_sql, result)
                    yield self._fmt_sse(SseEventType.STREAM.value, formatted)
                    if message_id:
                        interrupt_service.mark_approval_consumed(session_id, message_id)
                    return
                except Exception as sql_exc:
                    yield self._fmt_sse(SseEventType.ERROR.value, f"任务恢复失败且 SQL 兜底执行失败: {sql_exc}")
                    return
        yield self._fmt_sse(SseEventType.ERROR.value, f"任务恢复失败: {err_msg}")

    def _handle_resume_empty_result(
            self,
            session_id: str,
            decision: Optional[str],
            message_id: Optional[str],
            target_agent: Any,
            effective_config: Dict[str, Any],
    ) -> Generator[str, None, None]:
        """处理恢复执行后无 AI 消息输出的情况。"""
        if decision == ApprovalDecision.REJECT.value:
            if message_id:
                interrupt_service.mark_approval_consumed(session_id, message_id)
            yield self._fmt_sse(SseEventType.STREAM.value, GRAPH_RUNNER_REJECTED_MESSAGE)
            return
        interrupt_event = self._scan_subgraph_interrupts(
            session_id, effective_config=effective_config,
            candidate_agent_names=[target_agent.subgraph_id],
        )
        if interrupt_event:
            if message_id:
                interrupt_service.mark_approval_consumed(session_id, message_id)
            self._register_interrupts(session_id, interrupt_event, effective_config)
            yield self._fmt_sse(SseEventType.THINKING.value, SseMessage.RESUME_NEED_MANUAL_APPROVAL)
            yield self._fmt_sse(SseEventType.INTERRUPT.value, json.dumps(interrupt_event, ensure_ascii=False))
        else:
            yield self._fmt_sse(SseEventType.ERROR.value, SseMessage.ERROR_RESUME_EMPTY_RESULT)

    def _backfill_supervisor_state(
            self,
            supervisor_graph: Any,
            session_id: str,
            final_msgs: List[AIMessage],
    ) -> None:
        """将子 Agent 恢复结果回填到 Supervisor 状态，保证会话连续性。"""
        if not final_msgs:
            return
        sup_config = {"configurable": {"thread_id": session_id}}
        try:
            sup_state = supervisor_graph.get_state(sup_config)
            if getattr(sup_state, "values", None):
                supervisor_graph.update_state(sup_config, {"messages": final_msgs})
        except Exception as exc:
            log.warning(f"恢复后回填 Supervisor 状态失败，已降级跳过: {exc}")
