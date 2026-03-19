# -*- coding: utf-8 -*-
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
from constants.workflow_constants import GRAPH_STREAM_MODES, GraphQueueItemType
from services.agent_stream_bus import agent_stream_bus
from services.interrupt_service import interrupt_service
from services.request_cancellation_service import request_cancellation_service
from utils.custom_logger import CustomLogger, get_logger
from utils.date_utils import get_agent_date_context
from utils.history_compressor import compress_history_messages

log = get_logger(__name__)


class GraphRunner:
    """
    图执行器：AI 对话链路的核心调度中枢。

    职责：
    1. 管理 Supervisor 编译图缓存，同一配置只编译一次。
    2. 将同步图执行包装为可 yield 的 SSE 流，供 FastAPI StreamingResponse 消费。
    3. 拦截前置规则命中、RAG 注入、Interrupt 审批恢复等横切关注点。
    """

    def __init__(self, model_config: Optional[Dict[str, Any]] = None) -> None:
        """初始化执行器，可选注入默认模型配置。"""
        self.model_config: Dict[str, Any] = model_config or {}
        # 按模型配置指纹缓存编译好的 Supervisor 图，避免重复编译带来的初始化开销
        self._supervisor_cache: Dict[str, Any] = {}

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

    def _get_or_create_supervisor(self, model_config: Dict[str, Any]) -> Any:
        """
        获取（或创建）Supervisor 编译图实例。

        命中缓存时直接返回，未命中时编译后写入缓存，实现懒加载。
        """
        cache_key = self._build_config_key(model_config)
        if cache_key not in self._supervisor_cache:
            log.info(f"首次编译 Supervisor 图，config_key={cache_key[:8]}...")
            self._supervisor_cache[cache_key] = create_supervisor_graph(model_config)
        return self._supervisor_cache[cache_key]

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
        核心流式执行器（异步生产者-消费者模型）。

        【设计说明】
        图的执行（阻塞同步）跑在后台 daemon 线程（生产者），asyncio 事件循环
        作为消费者通过 asyncio.Queue 异步取数据转为 SSE 下发给前端。
        全程不阻塞事件循环，支持真正的并发流式推送。

        Args:
            user_input:          用户当前输入文本，"[RESUME]" 表示审批恢复。
            session_id:          会话 ID，同时作为 LangGraph checkpointer 的 thread_id。
            model_config:        覆盖默认配置的模型参数字典。
            history_messages:    历史对话列表 [{user_content, model_content}, ...]。
            session_context:     结构化会话上下文（城市、用户画像等槽位）。
            emit_response_start: 是否在流开始时发送 response_start 事件。

        Yields:
            标准 SSE 格式字符串（"event: ...\ndata: ...\n\n"）。
        """
        history_messages = history_messages or []
        session_context = session_context or {}
        # 实例默认配置与本次请求配置合并，请求级配置优先级更高
        effective_config = {**self.model_config, **(model_config or {})}
        graph = self._get_or_create_supervisor(effective_config)

        # ── 分支一：审批恢复流程 ──────────────────────────────────────────
        if user_input == "[RESUME]":
            async for chunk in self._handle_resume_stream_async(
                session_id=session_id,
                graph=graph,
                effective_config=effective_config,
                emit_response_start=emit_response_start,
            ):
                yield chunk
            return

        # ── 基础输入校验 ──────────────────────────────────────────────────
        if not (user_input or "").strip():
            yield self._fmt_sse(SseEventType.ERROR.value, SseMessage.INVALID_RESUME_PARAM)
            return

        # ── 分支二：前置规则拦截（Zero-LLM、Zero-Graph） ────────────────────
        rule_result = self._try_rule_intercept(user_input)
        if rule_result is not None:
            thinking_hint, rule_content = rule_result
            if emit_response_start:
                yield self._fmt_sse(SseEventType.RESPONSE_START.value, "")
            yield self._fmt_sse(SseEventType.THINKING.value, thinking_hint)
            yield self._fmt_sse(SseEventType.STREAM.value, rule_content)
            yield self._fmt_sse(SseEventType.RESPONSE_END.value, "")
            return

        # ── 构造入图消息列表 ──────────────────────────────────────────────
        messages = self._build_input_messages(history_messages, user_input, session_context)

        # ── RAG 上下文注入（可选） ────────────────────────────────────────
        rag_thinking_text = self._inject_rag_context(messages, user_input, effective_config)

        # ── 构造图执行 Config ─────────────────────────────────────────────
        run_id = f"{session_id}:{uuid.uuid4().hex}"
        request_cancellation_service.register_request(run_id)
        graph_config = {"configurable": {"thread_id": session_id, "run_id": run_id}}
        graph_inputs = {
            "messages": messages,
            "session_id": session_id,
            "llm_config": effective_config,
            "context_slots": session_context.get("context_slots") or {},
            "context_summary": session_context.get("context_summary") or "",
        }

        # response_start 必须在任何内容之前发出，让前端进入流式接收模式
        if emit_response_start:
            yield self._fmt_sse(SseEventType.RESPONSE_START.value, "")

        if rag_thinking_text:
            yield self._fmt_sse(SseEventType.THINKING.value, rag_thinking_text)

        # ── 启动后台图执行并消费事件队列（异步） ──────────────────────────
        async for chunk in self._run_graph_stream_async(
            graph=graph,
            graph_inputs=graph_inputs,
            graph_config=graph_config,
            session_id=session_id,
            run_id=run_id,
            effective_config=effective_config,
        ):
            yield chunk

    # ------------------------------------------------------------------ #
    #  前置规则拦截引擎                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _try_rule_intercept(user_input: str) -> Optional[Tuple[str, str]]:
        """
        前置规则拦截器（Zero-LLM、Zero-Graph）。

        当用户输入命中预设规则，且输入长度在覆盖率估算范围内时，
        直接返回 (thinking_text, response_text)，完全不进入 LangGraph。

        Returns:
            命中时返回 (thinking提示文本, 回复正文)；未命中或放行时返回 None。
        """
        user_text_lower = user_input.lower().strip()
        max_scan_len = GRAPH_RUNNER_TUNING.rule_scan_max_len

        # 超长输入直接跳过规则扫描，避免误拦截包含复杂业务意图的长句
        if len(user_text_lower) > max_scan_len:
            return None

        sorted_rules = rule_registry.get_rules()
        matched_responses: List[str] = []
        matched_ids: List[str] = []

        for rule in sorted_rules:
            # 使用预编译正则，O(1) 快速匹配
            if any(p.search(user_text_lower) for p in rule._compiled_patterns):
                context_kwargs = handle_action(rule.action)
                try:
                    final_resp = rule.response_template.format(**context_kwargs)
                    matched_responses.append(final_resp)
                    matched_ids.append(rule.id)
                except Exception as fmt_err:
                    log.error(f"规则拦截器模板 [{rule.id}] 格式化失败: {fmt_err}")

        if not matched_responses:
            return None

        # 意图覆盖率防爆盾：文本长度远超规则能覆盖范围时，放行至大模型
        chars_per_intent = GRAPH_RUNNER_TUNING.chars_per_intent
        estimated_covered_len = len(matched_responses) * chars_per_intent
        if len(user_text_lower) > estimated_covered_len + 5:
            log.info(
                f"规则部分命中但句子较长（{len(user_text_lower)}字），"
                f"疑似复杂意图，放行至大模型: {user_text_lower[:80]}"
            )
            return None

        # 多规则命中时将回复合并
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
        session_id: str,
        run_id: str,
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
        loop = asyncio.get_event_loop()
        # asyncio.Queue：生产者线程通过 call_soon_threadsafe 写入，消费协程 await get()
        event_queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        # 追踪参与本次请求的线程 ID，用于日志拦截器过滤
        owner_thread_ids: Set[int] = {threading.get_ident()}
        # 记录本轮已实时推流的 Agent，用于抑制 synthetic 重复输出
        live_streamed_agents: Set[str] = set()
        # 跨 chunk 的路由 JSON 前缀过滤状态
        router_prefix_buffer = ""
        router_prefix_done = False

        def _safe_enqueue(item: Tuple[str, Any], *, drop_on_full: bool = False) -> None:
            """从后台线程安全地向 asyncio.Queue 写入事件。"""
            def _put():
                try:
                    event_queue.put_nowait(item)
                except asyncio.QueueFull:
                    if not drop_on_full:
                        # 满队列时阻塞式写入（在事件循环中调度）
                        asyncio.ensure_future(event_queue.put(item), loop=loop)
            loop.call_soon_threadsafe(_put)

        # 记录上一条日志 SSE，连续重复的日志直接丢弃，避免前端 thinking 刷屏
        last_log_sse: str = ""

        def _log_interceptor(sse_message: str) -> None:
            """拦截当前请求线程产生的日志并推入事件队列（跨会话隔离）。"""
            nonlocal last_log_sse
            if threading.get_ident() not in owner_thread_ids:
                return
            if sse_message == last_log_sse:
                return
            last_log_sse = sse_message
            _safe_enqueue((GraphQueueItemType.LOG.value, sse_message), drop_on_full=True)

        def _live_stream_interceptor(payload: Dict[str, Any]) -> None:
            """将子 Agent 实时正文流推入事件队列。"""
            _safe_enqueue((GraphQueueItemType.LIVE_STREAM.value, payload), drop_on_full=True)

        def _graph_worker() -> None:
            """
            后台生产者线程：驱动 Supervisor 图执行并将事件推入队列。

            遇到 Interrupt 异常（子图挂起等待审批）视为正常终止，
            其他未知异常推入 ERROR 事件由消费协程处理。
            """
            owner_thread_ids.add(threading.get_ident())
            with request_cancellation_service.bind_request(run_id):
                try:
                    for ev_type, ev in graph.stream(
                        graph_inputs,
                        config=graph_config,
                        stream_mode=list(GRAPH_STREAM_MODES),
                    ):
                        if request_cancellation_service.is_cancelled(run_id):
                            log.info(f"Graph Worker 收到取消信号，停止事件采集。run_id={run_id}")
                            break
                        _safe_enqueue((GraphQueueItemType.GRAPH.value, (ev_type, ev)))
                    _safe_enqueue((GraphQueueItemType.DONE.value, None))
                except Exception as worker_exc:
                    err_msg = str(worker_exc)
                    if "Interrupt(" in err_msg or worker_exc.__class__.__name__ == "GraphInterrupt":
                        log.info(f"Graph Worker 检测到 Interrupt 挂起: {err_msg[:200]}")
                        _safe_enqueue((GraphQueueItemType.DONE.value, None))
                    else:
                        log.exception(f"Graph Worker 执行异常: {worker_exc}")
                        _safe_enqueue((GraphQueueItemType.ERROR.value, err_msg))

        # 注册日志拦截回调
        CustomLogger.add_global_sse_callback(_log_interceptor)
        # 注册子 Agent 实时流回调（可配置开关）
        if AGENT_LIVE_STREAM_ENABLED:
            agent_stream_bus.register_callback(run_id, _live_stream_interceptor)

        # 启动后台图执行线程
        worker_thread = threading.Thread(target=_graph_worker, daemon=True, name=f"graph-worker-{run_id[:8]}")
        worker_thread.start()
        if worker_thread.ident is not None:
            owner_thread_ids.add(worker_thread.ident)

        try:
            async for chunk in self._consume_event_queue_async(
                event_queue=event_queue,
                worker_thread=worker_thread,
                run_id=run_id,
                session_id=session_id,
                effective_config=effective_config,
                graph=graph,
                live_streamed_agents=live_streamed_agents,
                router_prefix_buffer=router_prefix_buffer,
                router_prefix_done=router_prefix_done,
            ):
                yield chunk
        except GeneratorExit:
            request_cancellation_service.cancel_request(run_id)
            log.warning(f"客户端断开连接，已触发取消。session_id={session_id}, run_id={run_id}")
            raise
        finally:
            if AGENT_LIVE_STREAM_ENABLED:
                agent_stream_bus.unregister_callback(run_id)
            CustomLogger.remove_global_sse_callback(_log_interceptor)
            if worker_thread.is_alive():
                request_cancellation_service.cancel_request(run_id)
                worker_thread.join(timeout=1.0)
            if worker_thread.is_alive():
                log.warning("Graph Worker 线程未能在超时内退出，主流程继续。")
            request_cancellation_service.cleanup_request(run_id)

    # ------------------------------------------------------------------ #
    #  事件队列消费器                                                        #
    # ------------------------------------------------------------------ #

    async def _consume_event_queue_async(
        self,
        event_queue: asyncio.Queue,
        worker_thread: threading.Thread,
        run_id: str,
        session_id: str,
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
                yield self._fmt_sse(SseEventType.ERROR.value, item_data)
                break

            elif item_type == GraphQueueItemType.LOG.value:
                progress_emitted = True
                yield item_data

            elif item_type == GraphQueueItemType.LIVE_STREAM.value:
                sse_chunk = self._handle_live_stream_event(item_data, live_streamed_agents)
                if sse_chunk:
                    progress_emitted = True
                    stream_content_emitted = True
                    yield sse_chunk

            elif item_type == GraphQueueItemType.GRAPH.value:
                ev_type, ev = item_data

                if ev_type == "messages":
                    sse_chunk, router_prefix_buffer, router_prefix_done = self._handle_message_chunk(
                        ev, router_prefix_buffer, router_prefix_done
                    )
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
                    ):
                        progress_emitted = True
                        if sse_chunk.startswith(f"event: {SseEventType.INTERRUPT.value}"):
                            interrupt_emitted = True
                        if sse_chunk.startswith(f"event: {SseEventType.STREAM.value}"):
                            stream_content_emitted = True
                        yield sse_chunk

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
                yield self._fmt_sse(SseEventType.ERROR.value, "本轮未生成可展示的最终答复，请重试。")

        yield self._fmt_sse(SseEventType.RESPONSE_END.value, "")

    # ------------------------------------------------------------------ #
    #  事件处理子方法                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _handle_live_stream_event(
        payload: Any,
        live_streamed_agents: Set[str],
    ) -> str:
        """
        处理子 Agent 实时正文流事件。

        记录已推流的 Agent 名称，供后续抑制 synthetic 重复输出。
        """
        if not isinstance(payload, dict):
            return ""
        visible_content = str(payload.get("content") or "").strip()
        if not visible_content:
            return ""
        source_agent = str(payload.get("agent_name") or "").strip()
        if source_agent:
            live_streamed_agents.add(source_agent)
        return GraphRunner._fmt_sse(SseEventType.STREAM.value, visible_content)

    def _handle_message_chunk(
        self,
        event: Tuple[Any, Any],
        router_prefix_buffer: str,
        router_prefix_done: bool,
    ) -> Tuple[str, str, bool]:
        """
        处理 messages 模式下的流式 token chunk。

        过滤逻辑：
        - 静默内部路由/规划节点的文本输出，只允许最终节点往前端推流。
        - 过滤工具调用 chunk（工具调用通过 THINKING 事件单独展示）。
        - 跨 chunk 去掉路由 JSON 前缀，避免 `{"intent":...}` 泄漏到用户界面。
        """
        msg_chunk, metadata = event
        if not isinstance(msg_chunk, AIMessageChunk):
            return "", router_prefix_buffer, router_prefix_done

        node_name = metadata.get("langgraph_node", "")

        # 这些节点产生的文本是内部路由决策，不应该展示给用户
        _silenced_nodes = {
            "Domain_Router_Node",
            "Rule_Engine_Node",
            "Intent_Router_Node",
            "Parent_Planner_Node",
            "dispatcher_node",
            "worker_node",
            "reducer_node",
        }

        if node_name in _silenced_nodes:
            return "", router_prefix_buffer, router_prefix_done

        # 工具调用 chunk 只通过 THINKING 展示，不做文本推流
        if getattr(msg_chunk, "tool_calls", None):
            tool_thinking = ""
            for tc in msg_chunk.tool_calls:
                tool_thinking = f"🔧 正在调度: {tc.get('name', '...')}"
            if tool_thinking:
                return self._fmt_sse(SseEventType.THINKING.value, tool_thinking), router_prefix_buffer, router_prefix_done
            return "", router_prefix_buffer, router_prefix_done

        if not msg_chunk.content:
            return "", router_prefix_buffer, router_prefix_done

        # 只允许主路径与专业 Agent 的正文流推送
        from agent.registry import agent_classes as _agent_classes
        allowed_names = {None, "", "ChatAgent", "Aggregator", *_agent_classes.keys()}
        if getattr(msg_chunk, "name", None) not in allowed_names:
            return "", router_prefix_buffer, router_prefix_done

        # 单 chunk 路由 JSON 过滤
        visible = self._strip_router_json_prefix_single(msg_chunk.content)
        # 跨 chunk 路由 JSON 前缀过滤
        visible, router_prefix_buffer, router_prefix_done = self._strip_router_json_prefix_cross_chunk(
            visible, router_prefix_buffer, router_prefix_done
        )

        if not (isinstance(visible, str) and visible.strip()):
            return "", router_prefix_buffer, router_prefix_done

        return self._fmt_sse(SseEventType.STREAM.value, visible), router_prefix_buffer, router_prefix_done

    # ------------------------------------------------------------------ #
    #  SSE 格式化工具                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fmt_sse(event_type: str, content: str) -> str:
        """
        将事件类型和内容格式化为标准 SSE 字符串。

        格式：
            event: <event_type>\n
            data: {"type": "<event_type>", "content": "<content>"}\n\n
        """
        payload = json.dumps(
            {SsePayloadField.TYPE.value: event_type, SsePayloadField.CONTENT.value: content},
            ensure_ascii=False,
        )
        return f"event: {event_type}\ndata: {payload}\n\n"

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
    def _safe_get_agent_state(agent: Any, agent_name: str, context: str) -> Optional[Any]:
        """安全地获取 Agent 图快照，失败时返回 None 而不抛出异常。"""
        try:
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
                    return content.strip()
                if isinstance(content, list):
                    parts = [
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content
                        if item
                    ]
                    text = "".join(parts).strip()
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
    ) -> Generator[str, None, None]:
        """
        处理 updates 模式下的图状态更新事件。

        包含：
        - 各路由节点的决策日志（domain/intent/planner/worker 耗时等）。
        - in-band Interrupt payload 提取与注册。
        - Agent 节点的操作日志和 synthetic 消息推送。
        """
        from agent.registry import agent_classes as _agent_classes

        for node_name, node_val in event.items():
            # 记录本轮参与执行的 Agent 名称
            if node_name in _agent_classes:
                active_agent_candidates.add(node_name)

            # ── 路由节点日志 ────────────────────────────────────────────
            yield from self._emit_router_thinking(node_name, node_val)

            # ── Interrupt 提取 ──────────────────────────────────────────
            if not interrupt_emitted:
                inband_interrupt = self._extract_interrupt_from_node_val(node_name, node_val, _agent_classes)
                if inband_interrupt:
                    source_node, payload = inband_interrupt
                    interrupt_event = self._format_interrupt_payload(session_id, payload)
                    self._register_interrupts(session_id, interrupt_event, effective_config)
                    yield self._fmt_sse(SseEventType.THINKING.value, SseMessage.NEED_MANUAL_APPROVAL)
                    yield self._fmt_sse(
                        SseEventType.INTERRUPT.value,
                        json.dumps(interrupt_event, ensure_ascii=False),
                    )

            # ── Agent 操作日志与 synthetic 消息 ────────────────────────
            if not isinstance(node_val, dict) or "messages" not in node_val:
                continue

            for msg in node_val.get("messages", []):
                if not isinstance(msg, AIMessage):
                    continue
                metadata = getattr(msg, "response_metadata", {}) or {}
                # 从 response_metadata 提取 Agent 操作日志
                for log_entry in metadata.get("operation_logs", []):
                    yield self._fmt_sse(SseEventType.THINKING.value, log_entry)
                # synthetic 消息：由于未走正常 LLM stream，需直接推送给前端
                should_emit = metadata.get("synthetic") and bool(msg.content)
                source_agent_name = str(getattr(msg, "name", "") or node_name or "")
                if metadata.get("live_streamed"):
                    should_emit = False
                elif live_streamed_agents and source_agent_name in live_streamed_agents:
                    should_emit = False
                if should_emit and (node_name != "aggregator_node" or metadata.get("force_emit")):
                    visible = self._strip_router_json_prefix_single(msg.content)
                    if isinstance(visible, str) and visible.strip():
                        yield self._fmt_sse(SseEventType.STREAM.value, visible)

                
    @staticmethod
    def _emit_router_thinking(node_name: str, node_val: Any) -> Generator[str, None, None]:
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
            yield GraphRunner._fmt_sse(
                SseEventType.THINKING.value,
                f"智能路由指派: {node_val['intent']} "
                f"(置信度: {node_val.get('intent_confidence', 0):.2f}{elapsed_text})",
            )
            return

        # 规划器日志
        if node_name == "Parent_Planner_Node" and "task_list" in node_val:
            elapsed = node_val.get("planner_elapsed_ms")
            elapsed_text = f"，耗时: {int(elapsed)}ms" if isinstance(elapsed, (int, float)) else ""
            tasks = node_val.get("task_list") or []
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
                elapsed = worker_item.get("elapsed_ms")
                if not isinstance(elapsed, (int, float)):
                    continue
                task_id = str(worker_item.get("task_id") or "")
                agent_name = str(worker_item.get("agent") or "")
                label = f"{task_id}:{agent_name}" if task_id and agent_name else task_id or agent_name or "worker"
                yield GraphRunner._fmt_sse(
                    SseEventType.THINKING.value,
                    f"⏱️ 子任务[{label}]执行耗时: {int(elapsed)}ms",
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
        session_id: str,
        graph: Any,
        effective_config: Dict[str, Any],
        emit_response_start: bool,
    ) -> AsyncGenerator[str, None]:
        """统一处理 [RESUME] 审批恢复流程的 SSE 推送（异步版本）。"""
        resume_meta = self._check_pending_approval(session_id)
        if not resume_meta:
            yield self._fmt_sse(SseEventType.ERROR.value, SseMessage.INVALID_RESUME_PARAM)
            return
        if emit_response_start:
            yield self._fmt_sse(SseEventType.RESPONSE_START.value, "")
        yield self._fmt_sse(SseEventType.THINKING.value, SseMessage.RESUME_DETECTED)
        for chunk in self._handle_resume(
            session_id=session_id,
            resume_meta=resume_meta,
            supervisor_graph=graph,
            effective_config=effective_config,
        ):
            yield chunk
        yield self._fmt_sse(SseEventType.RESPONSE_END.value, "")

    def _check_pending_approval(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        查询当前会话是否有前端已提交的审批结果。

        Returns:
            有待恢复审批时返回恢复元信息字典；否则返回 None。
        """
        approval = interrupt_service.fetch_latest_resolved_approval(session_id)
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
                snap = self._safe_get_agent_state(candidate, preferred_agent, "恢复前定位")
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
            if decision == ApprovalDecision.APPROVE.value and resume_meta.get("action_name") == SQL_APPROVAL_ACTION_NAME:
                approved_sql = (resume_meta.get("action_args") or {}).get("sql")
                if approved_sql:
                    try:
                        from agent.tools.sql_tools import execute_sql, format_sql_result_for_user
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
                    from agent.tools.sql_tools import execute_sql, format_sql_result_for_user
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

    def _handle_resume_exception(self, session_id, resume_meta, resume_exc, target_agent, effective_config, message_id, decision):
        """处理恢复执行时抛出的异常，区分 Interrupt 挂起和真实错误。"""
        err_msg = str(resume_exc)
        if "Interrupt(" in err_msg or resume_exc.__class__.__name__ == "GraphInterrupt":
            interrupt_event = self._scan_subgraph_interrupts(session_id, effective_config=effective_config, candidate_agent_names=[target_agent.subgraph_id])
            if interrupt_event:
                if message_id: interrupt_service.mark_approval_consumed(session_id, message_id)
                self._register_interrupts(session_id, interrupt_event, effective_config)
                yield self._fmt_sse(SseEventType.THINKING.value, SseMessage.RESUME_NEED_MANUAL_APPROVAL)
                yield self._fmt_sse(SseEventType.INTERRUPT.value, json.dumps(interrupt_event, ensure_ascii=False))
            else:
                yield self._fmt_sse(SseEventType.ERROR.value, "恢复执行进入审批状态，但未获取到审批详情，请重试。")
            return
        log.exception(f"Resume non-interrupt failure: {err_msg}")
        if "1214" in err_msg and resume_meta.get("action_name") == SQL_APPROVAL_ACTION_NAME and decision == ApprovalDecision.APPROVE.value:
            approved_sql = (resume_meta.get("action_args") or {}).get("sql")
            if approved_sql:
                try:
                    from agent.tools.sql_tools import execute_sql, format_sql_result_for_user
                    result = execute_sql(approved_sql, domain="LOCAL_DB")
                    yield self._fmt_sse(SseEventType.STREAM.value, format_sql_result_for_user(approved_sql, result))
                    if message_id: interrupt_service.mark_approval_consumed(session_id, message_id)
                    return
                except Exception as sql_exc:
                    yield self._fmt_sse(SseEventType.ERROR.value, f"任务恢复失败且SQL兜底执行失败: {sql_exc}")
                    return
        yield self._fmt_sse(SseEventType.ERROR.value, f"任务恢复失败: {err_msg}")

    def _handle_resume_empty_result(self, session_id, decision, message_id, target_agent, effective_config):
        """处理恢复执行后无 AI 消息输出的情况。"""
        if decision == ApprovalDecision.REJECT.value:
            if message_id: interrupt_service.mark_approval_consumed(session_id, message_id)
            yield self._fmt_sse(SseEventType.STREAM.value, GRAPH_RUNNER_REJECTED_MESSAGE)
            return
        interrupt_event = self._scan_subgraph_interrupts(session_id, effective_config=effective_config, candidate_agent_names=[target_agent.subgraph_id])
        if interrupt_event:
            if message_id: interrupt_service.mark_approval_consumed(session_id, message_id)
            self._register_interrupts(session_id, interrupt_event, effective_config)
            yield self._fmt_sse(SseEventType.THINKING.value, SseMessage.RESUME_NEED_MANUAL_APPROVAL)
            yield self._fmt_sse(SseEventType.INTERRUPT.value, json.dumps(interrupt_event, ensure_ascii=False))
        else:
            yield self._fmt_sse(SseEventType.ERROR.value, SseMessage.ERROR_RESUME_EMPTY_RESULT)

    def _backfill_supervisor_state(self, supervisor_graph, session_id, final_msgs):
        """将子 Agent 恢复结果回填到 Supervisor 状态，保证会话连续性。"""
        if not final_msgs: return
        sup_config = {"configurable": {"thread_id": session_id}}
        try:
            sup_state = supervisor_graph.get_state(sup_config)
            if getattr(sup_state, "values", None):
                supervisor_graph.update_state(sup_config, {"messages": final_msgs})
        except Exception as exc:
            log.warning(f"恢复后回填 Supervisor 状态失败，已降级跳过: {exc}")
