import hashlib
import json
import time
import uuid
from typing import Any, Dict, Generator, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, AIMessageChunk
from langgraph.types import Command

from agent.graph_state import AgentRequest
from agent.graphs.supervisor import create_graph as create_supervisor_graph
from agent.llm.unified_loader import create_model_from_config
from agent.rag.vector_store import vector_store_service
from agent.registry import agent_classes
from constants.approval_constants import (
    ApprovalDecision,
    ApprovalStatus,
    DEFAULT_ALLOWED_DECISIONS,
    DEFAULT_INTERRUPT_MESSAGE,
    SQL_APPROVAL_ACTION_NAME,
)
from constants.agent_messages import GRAPH_RUNNER_REJECTED_MESSAGE
from services.interrupt_service import interrupt_service
from services.request_cancellation_service import request_cancellation_service
from constants.sse_constants import SseEventType, SseMessage, SsePayloadField
from constants.workflow_constants import GRAPH_STREAM_MODES, GraphQueueItemType
from utils.custom_logger import get_logger
from agent.rules.registry import rule_registry
from agent.rules.actions import handle_action
from config.runtime_settings import GRAPH_RUNNER_TUNING
from utils.date_utils import get_agent_date_context
from config.runtime_settings import AGENT_LOOP_CONFIG
from utils.history_compressor import compress_history_messages

log = get_logger(__name__)

"""
【模块说明】
API 接口的直接消费者。
负责拉起 Supervisor，将执行过程包装为 SSE 事件推给前端。
同时负责拦截子图 Bubble Up 的 Interrupt（挂起）事件并进行下发。
"""


class GraphRunner:
    """图执行器：AI对话链路的核心调度中枢，负责图的创建、执行和SSE推送"""

    def __init__(self, model_config: Optional[dict] = None):
        """初始化图执行器"""
        self.model_config = model_config or {}
        # 缓存编译好的图实例，避免重复编译
        self._supervisor_cache: Dict[str, Any] = {}

    @staticmethod
    def _config_key(model_config: dict) -> str:
        """将模型配置序列化为 MD5 摘要，用作 Supervisor 缓存的键。"""
        stable = json.dumps(model_config or {}, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(stable.encode("utf-8")).hexdigest()

    def _get_supervisor(self, model_config: dict):
        """获取或创建 Supervisor 编译图实例（带内存缓存）。"""
        cache_key = self._config_key(model_config)
        if cache_key not in self._supervisor_cache:
            self._supervisor_cache[cache_key] = create_supervisor_graph(model_config)
        return self._supervisor_cache[cache_key]

    def stream_run(
            self, user_input: str, session_id: str,
            model_config: Optional[dict] = None,
            history_messages: Optional[list] = None,
            session_context: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """
        核心流式执行器（使用后台线程实现真正的实时流）。
        由于 _agent_node 是同步阻塞执行的，如果不使用多线程，期间产生的底层日志
        (log.info等) 会被缓存在队列中直到节点结束才能下发。
        采用生产者-消费者模型，将图的执行放入后台线程，主线程只负责从队列取数据下发 SSE。
        """
        history_messages = history_messages or []
        # 会话结构化上下文（城市/画像）由 ChatService 注入
        session_context = session_context or {}
        effective_config = {**self.model_config, **(model_config or {})}
        graph = self._get_supervisor(effective_config)

        # 1. 检测前端是否发起了审批恢复操作
        resume_meta = self._check_pending_approval(session_id)
        if resume_meta:
            yield self._format_sse(SseEventType.RESPONSE_START.value, "")
            yield self._format_sse(SseEventType.THINKING.value, SseMessage.RESUME_DETECTED)
            yield from self._handle_resume(session_id, resume_meta, graph, effective_config)
            yield self._format_sse(SseEventType.RESPONSE_END.value, "")
            return

        if user_input == "[RESUME]" or not user_input:
            yield self._format_sse(SseEventType.ERROR.value, SseMessage.INVALID_RESUME_PARAM)
            return

        # ==========================================
        # Phase 2: 第一道防线 - 前置规则拦截 (Zero-LLM, Zero-Graph)
        # ==========================================

        import re

        user_text_lower = user_input.lower().strip()
        # 规则引擎扫描长度与覆盖率估算参数（统一走配置中心）
        _MAX_RULE_SCAN_LEN = GRAPH_RUNNER_TUNING.rule_scan_max_len
        _CHARS_PER_INTENT = GRAPH_RUNNER_TUNING.chars_per_intent
        if len(user_text_lower) <= _MAX_RULE_SCAN_LEN:
            sorted_rules = rule_registry.get_rules()
            matched_responses = []
            matched_ids = []
            # 1. 遍历所有规则，支持“复合意图”全量扫描
            for rule in sorted_rules:
                # 使用预编译配置以达到 O(1) 的匹配速度
                if any(p.search(user_text_lower) for p in rule._compiled_patterns):
                    # 命中规则，执行并组装模板
                    context_kwargs = handle_action(rule.action)
                    try:
                        final_resp = rule.response_template.format(**context_kwargs)
                        matched_responses.append(final_resp)
                        matched_ids.append(rule.id)
                    except Exception as e:
                        log.error(f"前置拦截器模板 {rule.id} 组装失败: {e}")

            if matched_responses:
                # 2. 【核心优化：意图覆盖率防爆盾】
                # 防止“局部匹配灾难”：句长远超命中规则能覆盖的程度，说明有未处理的复杂意图，必须放行
                estimated_covered_length = len(matched_responses) * _CHARS_PER_INTENT

                if len(user_text_lower) > estimated_covered_length + 5:
                    log.info(
                        f"触发部分拦截，但句子较长({len(user_text_lower)}字)，疑似包含复杂意图，放行至大模型: {user_text_lower}")
                    # ⚠️ 这里不 return，直接跳过拦截，让 LangGraph 和大模型接管整个长句
                else:
                    # 3. 完美覆盖！组合多个命中规则的回复
                    combined_resp = "\n\n".join(matched_responses)  # 多个回答用换行隔开
                    ids_str = ", ".join(matched_ids)

                    # 包装为标准的 SSE 流直接送给前端，全过程零 LLM 零图初始化！
                    yield self._format_sse(SseEventType.RESPONSE_START.value, "")
                    yield self._format_sse(SseEventType.THINKING.value, f"⚡ 极速拦截复合意图：命中 [{ids_str}]")
                    yield self._format_sse(SseEventType.STREAM.value, combined_resp)
                    yield self._format_sse(SseEventType.RESPONSE_END.value, "")
                    log.info(f"Pre-Graph 复合拦截生效: [{ids_str}] - {user_text_lower}")
                    return  # 彻底中断！不进入 LangGraph

        messages = self._build_initial_messages(history_messages, user_input, session_context)

        # 2. RAG 上下文注入
        rag_enabled = effective_config.get("rag_enabled", False)
        if isinstance(rag_enabled, str):
            rag_enabled = rag_enabled.lower() == "true"

        if rag_enabled:
            rag_context, rag_sources = self._retrieve_rag_context(user_input, effective_config)
            if rag_context:
                insert_idx = len(messages) - 1 if isinstance(messages[-1], HumanMessage) else len(messages)
                messages.insert(insert_idx, SystemMessage(content=f"参考以下知识库回答:\n{rag_context}"))
                yield self._format_sse(SseEventType.THINKING.value, f"RAG 命中 {len(rag_sources)} 个来源")

        # 3. 构造顶级调用 Config
        run_id = f"{session_id}:{uuid.uuid4().hex}"
        request_cancellation_service.register_request(run_id)
        config = {
            "configurable": {"thread_id": session_id, "run_id": run_id},
            "run_name": f"Supervisor_Run_{session_id}"  # 给 LangSmith 打标签
        }
        # 将结构化上下文一并透传给 GraphState，供 supervisor / agent 复用
        inputs = {
            "messages": messages,
            "session_id": session_id,
            "llm_config": effective_config,
            "context_slots": session_context.get("context_slots") or {},
            "context_summary": session_context.get("context_summary") or "",
        }

        import queue
        import threading
        # 统一事件队列，容纳底层的 python text log 和上层的 graph event
        event_queue = queue.Queue(maxsize=2000)
        owner_thread_ids = {threading.get_ident()}

        def _safe_put(item: Tuple[str, Any], *, drop_if_full: bool = False):
            try:
                event_queue.put_nowait(item)
            except queue.Full:
                if drop_if_full:
                    return
                event_queue.put(item)

        # 记录最近一条日志 SSE，避免重复回调导致前端 thinking 重复刷屏。
        last_log_sse = ""

        def log_interceptor(sse_message: str):
            """拦截日志并推送到队列"""
            nonlocal last_log_sse
            # 仅接受当前请求主线程/worker线程日志，避免跨会话串流。
            if threading.get_ident() not in owner_thread_ids:
                return
            # 连续重复日志直接丢弃，避免 UI 出现同一条 thinking 连续多次。
            if sse_message == last_log_sse:
                return
            last_log_sse = sse_message
            _safe_put((GraphQueueItemType.LOG.value, sse_message), drop_if_full=True)

        from utils.custom_logger import CustomLogger
        CustomLogger.add_global_sse_callback(log_interceptor)

        def graph_worker():
            """后台线程：执行图并将事件推入队列"""
            owner_thread_ids.add(threading.get_ident())
            with request_cancellation_service.bind_request(run_id):
                try:
                    for ev_type, ev in graph.stream(inputs, config=config, stream_mode=list(GRAPH_STREAM_MODES)):
                        if request_cancellation_service.is_cancelled(run_id):
                            log.info(f"Graph worker 检测到取消信号，终止事件采集。run_id={run_id}")
                            break
                        _safe_put((GraphQueueItemType.GRAPH.value, (ev_type, ev)))
                    _safe_put((GraphQueueItemType.DONE.value, None))
                except Exception as e:
                    # 兼容最新 LangGraph 版本：interrupt() 会抛出特俗异常提前终止流
                    err_msg = str(e)
                    if "Interrupt(" in err_msg or e.__class__.__name__ == "GraphInterrupt":
                        log.info(f"Graph 节点抛出执行挂起 (Interrupt): {err_msg}")
                        _safe_put((GraphQueueItemType.DONE.value, None))
                    else:
                        log.exception(f"执行异常: {e}")
                        _safe_put((GraphQueueItemType.ERROR.value, err_msg))

        # 启动后台图执行线程
        worker_thread = threading.Thread(target=graph_worker, daemon=True)
        worker_thread.start()
        if worker_thread.ident is not None:
            owner_thread_ids.add(worker_thread.ident)

        yield self._format_sse(SseEventType.RESPONSE_START.value, "")

        try:
            start_ts = time.time()
            last_event_ts = start_ts
            last_heartbeat_ts = 0.0
            interrupt_emitted = False
            error_emitted = False
            # 是否已经向前端输出过正文 stream（用于最终兜底判断）
            stream_content_emitted = False
            # 记录本轮真正触发过的子 Agent，供必要时做“定向中断扫描”
            active_agent_candidates: set[str] = set()
            progress_emitted = False
            heartbeat_emitted = False
            idle_heartbeat_sec = GRAPH_RUNNER_TUNING.idle_heartbeat_sec
            idle_timeout_sec = GRAPH_RUNNER_TUNING.idle_timeout_sec
            idle_timeout_enabled = GRAPH_RUNNER_TUNING.idle_timeout_enabled
            hard_timeout_sec = GRAPH_RUNNER_TUNING.hard_timeout_sec
            # 流式前缀过滤器状态：用于跨 chunk 去掉路由 JSON 前缀。
            router_prefix_buffer = ""
            router_prefix_done = False

            # 在主线程中阻塞监听队列并下发
            while True:
                now = time.time()

                # 总执行超时保护：避免模型或外部网络异常导致永不返回
                if (now - start_ts > hard_timeout_sec) and (now - last_event_ts >= idle_timeout_sec):
                    error_emitted = True
                    request_cancellation_service.cancel_request(run_id)
                    yield self._format_sse(SseEventType.ERROR.value, SseMessage.ERROR_TIMEOUT)
                    break

                try:
                    item_type, item_data = event_queue.get(timeout=GRAPH_RUNNER_TUNING.queue_poll_timeout_sec)
                except queue.Empty:
                    now = time.time()
                    # Worker 还活着：定时回传心跳，避免前端“卡住无反馈”
                    if worker_thread.is_alive():
                        # 注意：空闲超时在复杂工具调用场景容易误伤，这里默认仅发心跳，不做 idle fail-fast。
                        # 若需启用可在后续按路由/工具粒度再细化。
                        if idle_timeout_enabled and (now - last_event_ts >= idle_timeout_sec):
                            error_emitted = True
                            request_cancellation_service.cancel_request(run_id)
                            yield self._format_sse(SseEventType.ERROR.value, SseMessage.ERROR_IDLE_TIMEOUT)
                            break
                        # 仅在“尚无任何流程进展输出”时发送一次心跳，避免前端刷屏。
                        if (not progress_emitted) and (not heartbeat_emitted) and (now - last_heartbeat_ts >= idle_heartbeat_sec):
                            yield self._format_sse(
                                SseEventType.THINKING.value,
                                f"⏱️ 正在等待模型首包返回（已等待 {int(now - start_ts)}s）",
                            )
                            last_heartbeat_ts = now
                            heartbeat_emitted = True
                        continue

                    # Worker 已退出但未发 done/error，返回兜底错误避免静默
                    error_emitted = True
                    yield self._format_sse(SseEventType.ERROR.value, SseMessage.ERROR_TASK_INTERRUPTED)
                    break

                last_event_ts = time.time()

                if item_type == GraphQueueItemType.DONE.value:
                    break

                elif item_type == GraphQueueItemType.ERROR.value:
                    error_emitted = True
                    request_cancellation_service.cancel_request(run_id)
                    yield self._format_sse(SseEventType.ERROR.value, item_data)
                    break

                elif item_type == GraphQueueItemType.LOG.value:
                    # 直接透传底层格式化好的 SSE payload
                    progress_emitted = True
                    yield item_data

                elif item_type == GraphQueueItemType.GRAPH.value:
                    event_type, event = item_data
                    if event_type == "messages":
                        msg_chunk, metadata = event
                        if isinstance(msg_chunk, AIMessageChunk):
                            node_name = metadata.get("langgraph_node", "")

                            # 纯文本推送：拦截内部路由与中间节点，只允许最终节点或单兵节点往 UI 推流
                            _silenced_text_nodes = {
                                "Domain_Router_Node",
                                "Rule_Engine_Node", "Intent_Router_Node",
                                "Parent_Planner_Node", "dispatcher_node",
                                "worker_node", "reducer_node"
                            }

                            if node_name not in _silenced_text_nodes:
                                if msg_chunk.content and not getattr(msg_chunk, "tool_calls", None):
                                    # 允许主路径与各专业 Agent 的正文流，避免“有思考无最终答复”
                                    allowed_names = {None, "", "ChatAgent", "Aggregator", *agent_classes.keys()}
                                    if getattr(msg_chunk, "name", None) in allowed_names:
                                        visible_content = self._strip_internal_router_json_prefix(msg_chunk.content)
                                        visible_content, router_prefix_buffer, router_prefix_done = (
                                            self._strip_router_json_prefix_across_chunks(
                                                visible_content,
                                                router_prefix_buffer,
                                                router_prefix_done,
                                            )
                                        )
                                        if isinstance(visible_content, str) and visible_content.strip():
                                            progress_emitted = True
                                            stream_content_emitted = True
                                            yield self._format_sse(SseEventType.STREAM.value, visible_content)

                            # 工具调用推送：无论哪个节点，只要拉起工具，必然展示给用户
                            if getattr(msg_chunk, "tool_calls", None):
                                for tc in msg_chunk.tool_calls:
                                    progress_emitted = True
                                    yield self._format_sse(SseEventType.THINKING.value, f"🔧 正在调度: {tc.get('name', '...')}")
                    elif event_type == "updates":
                        # 记录本轮参与执行的 Agent 节点，避免后续全量扫描所有 Agent。
                        for maybe_agent_name in event.keys():
                            if maybe_agent_name in agent_classes:
                                active_agent_candidates.add(maybe_agent_name)
                        inband_interrupt = self._extract_interrupt_from_updates(event)
                        if inband_interrupt and not interrupt_emitted:
                            source_node, inband_payload = inband_interrupt
                            payload = dict(inband_payload) if isinstance(inband_payload, dict) else inband_payload
                            if isinstance(payload, dict) and not payload.get("agent_name") and source_node in agent_classes:
                                payload["agent_name"] = source_node
                            interrupt_event = self._format_interrupt_payload(session_id, payload)
                            self._register_interrupts(session_id, interrupt_event, effective_config)
                            progress_emitted = True
                            yield self._format_sse(SseEventType.THINKING.value, SseMessage.NEED_MANUAL_APPROVAL)
                            yield self._format_sse(SseEventType.INTERRUPT.value, json.dumps(interrupt_event, ensure_ascii=False))
                            interrupt_emitted = True
                        for sse_chunk in self._process_supervisor_event(event):
                            progress_emitted = True
                            if sse_chunk.startswith(f"event: {SseEventType.STREAM.value}"):
                                stream_content_emitted = True
                            yield sse_chunk

            # 4. 可选兜底：仅在开启配置时做“定向中断扫描”。
            # 默认关闭，避免每轮全量实例化 Agent 造成性能抖动和额外连接风险。
            if (
                (not interrupt_emitted)
                and GRAPH_RUNNER_TUNING.post_run_interrupt_scan_enabled
                and active_agent_candidates
            ):
                try:
                    interrupt_event = self._scan_subgraph_interrupts(
                        session_id,
                        effective_config=effective_config,
                        candidate_agent_names=list(active_agent_candidates),
                    )
                except Exception as scan_exc:
                    log.warning(f"后置中断扫描失败，已降级跳过: {scan_exc}")
                    interrupt_event = None
                if interrupt_event:
                    self._register_interrupts(session_id, interrupt_event, effective_config)
                    yield self._format_sse(SseEventType.THINKING.value, SseMessage.NEED_MANUAL_APPROVAL)
                    yield self._format_sse(SseEventType.INTERRUPT.value, json.dumps(interrupt_event, ensure_ascii=False))

            # 5. 最终兜底：若图执行结束但没有任何正文输出，尝试从最终状态回捞答案，避免“有思考无答复”。
            if (not interrupt_emitted) and (not stream_content_emitted) and (not error_emitted):
                fallback_text = self._extract_final_answer_from_supervisor_state(graph, session_id)
                if fallback_text:
                    stream_content_emitted = True
                    yield self._format_sse(SseEventType.STREAM.value, fallback_text)
                else:
                    error_emitted = True
                    yield self._format_sse(SseEventType.ERROR.value, "本轮未生成可展示的最终答复，请重试。")

            yield self._format_sse(SseEventType.RESPONSE_END.value, "")

        except GeneratorExit:
            # 客户端主动停止生成：立即广播取消，尽量中断下游工具重试和后续 DAG 计算。
            request_cancellation_service.cancel_request(run_id)
            log.warning(f"GraphRunner 检测到客户端断连，已触发取消。session_id={session_id}, run_id={run_id}")
            raise
        finally:
            CustomLogger.remove_global_sse_callback(log_interceptor)
            if worker_thread.is_alive():
                request_cancellation_service.cancel_request(run_id)
                worker_thread.join(timeout=1.0)
            if worker_thread.is_alive():
                log.warning("Graph worker 线程仍在运行，已由主流程提前结束返回。")
            request_cancellation_service.cleanup_request(run_id)

    @staticmethod
    def _build_initial_messages(
        history: list,
        user_input: str,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> List[BaseMessage]:
        """
        将前端传入的历史消息列表转换为 LangChain 的 BaseMessage 列表。

        Args:
            history: MongoDB 中存储的历史消息（字典列表）
            user_input: 当前用户输入

        Returns:
            LangChain 消息对象列表，末尾为当前用户输入
        """
        # 统一时间基准系统消息，避免相对日期理解偏差
        messages: List[BaseMessage] = [SystemMessage(content=get_agent_date_context())]
        # 结构化会话上下文消息（简洁版），减少重复追问城市/用户画像
        context_msg = GraphRunner._build_session_context_message(session_context or {})
        if context_msg:
            messages.append(SystemMessage(content=context_msg))
        history_pairs: List[BaseMessage] = []
        for msg in history:
            if msg.get("user_content"):
                history_pairs.append(HumanMessage(content=msg["user_content"]))
            if msg.get("model_content"):
                history_pairs.append(AIMessage(content=msg["model_content"], name=msg.get("name")))
        history_pairs = compress_history_messages(
            history_pairs,
            model=None,
            max_tokens=AGENT_LOOP_CONFIG.context_compress_max_tokens,
            max_chars=AGENT_LOOP_CONFIG.context_compress_max_chars,
        )
        messages.extend(history_pairs)
        messages.append(HumanMessage(content=user_input))
        return messages

    @staticmethod
    def _build_session_context_message(session_context: Dict[str, Any]) -> str:
        """根据会话上下文生成简洁系统提示，避免把原始 JSON 暴露给用户。"""
        # 摘要优先：由 session_state_service 生成，长度短、语义稳定
        summary_text = (session_context.get("context_summary") or "").strip()
        if summary_text:
            return summary_text

        # 摘要不存在时，尝试从槽位构造最小提示
        slots = session_context.get("context_slots") or {}
        if not isinstance(slots, dict):
            return ""

        city_value = (str(slots.get("city", "") or "")).strip()
        name_value = (str(slots.get("name", "") or "")).strip()
        age_value = slots.get("age")
        gender_value = (str(slots.get("gender", "") or "")).strip()
        height_value = slots.get("height_cm")
        weight_value = slots.get("weight_kg")
        fragments: List[str] = []
        if city_value:
            fragments.append(f"当前城市: {city_value}")
        if name_value:
            fragments.append(f"用户姓名: {name_value}")
        if isinstance(age_value, int):
            fragments.append(f"年龄: {age_value}岁")
        if gender_value:
            gender_label = "男" if gender_value.lower() == "male" else "女" if gender_value.lower() == "female" else gender_value
            fragments.append(f"性别: {gender_label}")
        if isinstance(height_value, int):
            fragments.append(f"身高: {height_value}cm")
        if isinstance(weight_value, (int, float)):
            fragments.append(f"体重: {float(weight_value):.1f}kg")
        if not fragments:
            return ""
        return "【会话关键上下文】\n- " + "\n- ".join(fragments)

    def _retrieve_rag_context(self, user_input: str, model_config: dict) -> Tuple[str, List[str]]:
        """检索 RAG 知识库，返回拼接后的上下文文本和来源列表。失败时静默降级为空。"""
        try:
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
        except Exception:
            return "", []

    def _check_pending_approval(self, session_id: str) -> Optional[dict]:
        """检查当前会话是否有前端已提交的审批结果，若有则构造恢复元信息。"""
        approval = interrupt_service.fetch_latest_resolved_approval(session_id)
        if approval:
            action = (
                ApprovalDecision.APPROVE.value
                if approval.get("status") == ApprovalStatus.APPROVE.value
                else ApprovalDecision.REJECT.value
            )
            return {
                "message_id": approval.get("message_id"),
                "decision": action,
                "agent_name": approval.get("agent_name"),
                "action_name": approval.get("action_name"),
                "action_args": approval.get("action_args", {}),
                "subgraph_thread_id": approval.get("subgraph_thread_id"),
                "checkpoint_id": approval.get("checkpoint_id"),
                "checkpoint_ns": approval.get("checkpoint_ns"),
                "command": Command(resume=action),
            }
        return None

    @staticmethod
    def _snapshot_has_interrupt(snapshot: Any) -> bool:
        """兼容不同 LangGraph 版本的快照中断字段。"""
        top_interrupts = getattr(snapshot, "interrupts", None) or []
        if top_interrupts:
            return True
        next_nodes = getattr(snapshot, "next", None) or []
        if next_nodes:
            # 部分版本在挂起时不暴露 interrupts，但会保留下一个待执行节点
            return True
        tasks = getattr(snapshot, "tasks", None) or []
        for t in tasks:
            if getattr(t, "interrupts", None):
                return True
        return False

    @staticmethod
    def _is_connection_closed_error(exc: Exception) -> bool:
        """判断异常是否属于数据库连接已关闭场景。"""
        msg = str(exc or "").lower()
        keywords = (
            "connection is closed",
            "connection not open",
            "server closed the connection",
            "terminating connection",
        )
        return any(k in msg for k in keywords)

    def _safe_get_agent_state(self, agent: Any, agent_name: str, scene: str) -> Optional[Any]:
        """
        安全读取子 Agent 快照。

        目的：
        - 单个 Agent 状态读取失败时不拖垮主流程；
        - 记录明确日志，便于后续排查连接/状态问题。
        """
        try:
            return agent.get_state()
        except Exception as exc:
            if self._is_connection_closed_error(exc):
                log.warning(f"{scene}: 读取 Agent[{agent_name}] 快照失败（连接已关闭），已跳过该 Agent。")
            else:
                log.warning(f"{scene}: 读取 Agent[{agent_name}] 快照失败，已跳过。原因: {exc}")
            return None

    def _handle_resume(
        self,
        session_id: str,
        resume_meta: dict,
        supervisor_graph,
        effective_config: Dict[str, Any],
    ) -> Generator[str, None, None]:
        """
        恢复被 Interrupt 挂起的子 Agent 执行。

        流程：遍历注册表中的所有子 Agent，找到处于中断状态的那个，
        向其发送 Command 恢复执行，并将结果回填到 Supervisor 状态中。
        """
        command = resume_meta["command"]
        preferred_agent_name = resume_meta.get("agent_name")
        message_id = resume_meta.get("message_id")
        decision = resume_meta.get("decision")
        target_agent = None
        model, _ = create_model_from_config(**effective_config)
        preferred_thread_id = resume_meta.get("subgraph_thread_id")
        preferred_checkpoint_id = resume_meta.get("checkpoint_id")
        preferred_checkpoint_ns = resume_meta.get("checkpoint_ns")

        if preferred_agent_name and preferred_agent_name in agent_classes:
            candidate = agent_classes[preferred_agent_name].cls(
                AgentRequest(user_input="", model=model, session_id=session_id, subgraph_id=preferred_agent_name)
            )
            if preferred_checkpoint_id:
                target_agent = candidate
            else:
                snapshot = self._safe_get_agent_state(candidate, preferred_agent_name, "恢复前检查目标 Agent")
                if snapshot and self._snapshot_has_interrupt(snapshot):
                    target_agent = candidate

        # 寻找挂起状态的子 Agent
        if not target_agent:
            for name, info in agent_classes.items():
                agent = info.cls(AgentRequest(user_input="", model=model, session_id=session_id, subgraph_id=name))
                snapshot = self._safe_get_agent_state(agent, name, "恢复前扫描挂起 Agent")
                if snapshot and self._snapshot_has_interrupt(snapshot):
                    target_agent = agent
                    break

        if not target_agent:
            # 兜底：若找不到挂起子图，但审批的是 SQL 执行，直接执行已审批 SQL，避免前端空转报错。
            if (
                decision == ApprovalDecision.APPROVE.value
                and resume_meta.get("action_name") == SQL_APPROVAL_ACTION_NAME
            ):
                sql_args = resume_meta.get("action_args") or {}
                approved_sql = sql_args.get("sql")
                if approved_sql:
                    try:
                        from agent.tools.sql_tools import execute_sql, format_sql_result_for_user
                        fallback_result = execute_sql(approved_sql, domain="LOCAL_DB")
                        formatted_result = format_sql_result_for_user(approved_sql, fallback_result)
                        yield self._format_sse(SseEventType.STREAM.value, formatted_result)
                        if message_id:
                            interrupt_service.mark_approval_consumed(session_id, message_id)
                        return
                    except Exception as sql_exc:
                        yield self._format_sse(SseEventType.ERROR.value, f"未找到中断任务且 SQL 兜底执行失败: {sql_exc}")
                        return
            yield self._format_sse(SseEventType.ERROR.value, SseMessage.ERROR_NO_INTERRUPTED_TASK)
            return

        # 携带 Config 恢复执行
        # 子图恢复必须命中 BaseAgent.run 使用的 thread_id 规则: {session_id}_{subgraph_id}
        subgraph_thread_id = preferred_thread_id or f"{session_id}_{target_agent.subgraph_id}"
        subgraph_config = {"configurable": {"thread_id": subgraph_thread_id}}
        if preferred_checkpoint_id:
            subgraph_config["configurable"]["checkpoint_id"] = preferred_checkpoint_id
        if preferred_checkpoint_ns:
            subgraph_config["configurable"]["checkpoint_ns"] = preferred_checkpoint_ns
        log.info(
            f"Resume target={target_agent.subgraph_id}, thread_id={subgraph_thread_id}, "
            f"checkpoint_id={preferred_checkpoint_id}, checkpoint_ns={preferred_checkpoint_ns}, decision={decision}"
        )
        final_msgs = []
        try:
            # 原生 interrupt 直接用 Command(resume=...) 恢复
            for event in target_agent.graph.stream(command, config=subgraph_config, stream_mode="updates"):
                for _, val in event.items():
                    if "messages" in val:
                        for msg in val["messages"]:
                            if isinstance(msg, AIMessage):
                                final_msgs.append(msg)
                                yield self._format_sse(SseEventType.STREAM.value, msg.content)
        except Exception as exc:
            err_msg = str(exc)
            if "Interrupt(" in err_msg or exc.__class__.__name__ == "GraphInterrupt":
                interrupt_event = self._scan_subgraph_interrupts(
                    session_id,
                    effective_config=effective_config,
                    candidate_agent_names=[target_agent.subgraph_id],
                )
                if interrupt_event:
                    if message_id:
                        interrupt_service.mark_approval_consumed(session_id, message_id)
                    self._register_interrupts(session_id, interrupt_event, effective_config)
                    yield self._format_sse(SseEventType.THINKING.value, SseMessage.RESUME_NEED_MANUAL_APPROVAL)
                    yield self._format_sse(SseEventType.INTERRUPT.value, json.dumps(interrupt_event, ensure_ascii=False))
                else:
                    yield self._format_sse(SseEventType.ERROR.value, "恢复执行进入审批状态，但未获取到审批详情，请重试。")
            else:
                log.exception(f"Resume non-interrupt failure: {err_msg}")
                # 兜底策略：部分模型服务在 resume 场景会报 messages 参数非法(1214)。
                # 若审批的是 execute_sql，直接执行已审批 SQL，避免链路中断。
                if (
                    "1214" in err_msg
                    and resume_meta.get("action_name") == SQL_APPROVAL_ACTION_NAME
                    and decision == ApprovalDecision.APPROVE.value
                ):
                    sql_args = resume_meta.get("action_args") or {}
                    approved_sql = sql_args.get("sql")
                    if approved_sql:
                        try:
                            from agent.tools.sql_tools import execute_sql, format_sql_result_for_user
                            fallback_result = execute_sql(approved_sql, domain="LOCAL_DB")
                            formatted_result = format_sql_result_for_user(approved_sql, fallback_result)
                            yield self._format_sse(SseEventType.STREAM.value, formatted_result)
                            if message_id:
                                interrupt_service.mark_approval_consumed(session_id, message_id)
                            return
                        except Exception as sql_exc:
                            yield self._format_sse(SseEventType.ERROR.value, f"任务恢复失败且 SQL 兜底执行失败: {sql_exc}")
                            return
                yield self._format_sse(SseEventType.ERROR.value, f"任务恢复失败: {err_msg}")
            return

        if not final_msgs:
            if decision == ApprovalDecision.REJECT.value:
                if message_id:
                    interrupt_service.mark_approval_consumed(session_id, message_id)
                yield self._format_sse(SseEventType.STREAM.value, GRAPH_RUNNER_REJECTED_MESSAGE)
                return
            interrupt_event = self._scan_subgraph_interrupts(
                session_id,
                effective_config=effective_config,
                candidate_agent_names=[target_agent.subgraph_id],
            )
            if interrupt_event:
                if message_id:
                    interrupt_service.mark_approval_consumed(session_id, message_id)
                self._register_interrupts(session_id, interrupt_event, effective_config)
                yield self._format_sse(SseEventType.THINKING.value, SseMessage.RESUME_NEED_MANUAL_APPROVAL)
                yield self._format_sse(SseEventType.INTERRUPT.value, json.dumps(interrupt_event, ensure_ascii=False))
            else:
                yield self._format_sse(SseEventType.ERROR.value, SseMessage.ERROR_RESUME_EMPTY_RESULT)
            return

        if message_id:
            interrupt_service.mark_approval_consumed(session_id, message_id)

        # 状态回填 Supervisor
        if final_msgs:
            sup_config = {"configurable": {"thread_id": session_id}}
            try:
                sup_state = supervisor_graph.get_state(sup_config)
                if getattr(sup_state, "values", None):
                    supervisor_graph.update_state(sup_config, {"messages": final_msgs})
            except Exception as exc:
                # 不影响主结果返回，只做日志告警
                log.warning(f"恢复后回填 Supervisor 状态失败，已降级跳过: {exc}")

    def _process_supervisor_event(self, event: dict) -> Generator[str, None, None]:
        """
        处理 Supervisor 图的 updates 事件。

        提取以下信息作为 thinking 事件推送给前端：
        - Supervisor 路由决策
        - Agent 操作日志（工具调用、结果、错误等，由 _agent_node 收集并嵌入 response_metadata）
        """
        for node_name, node_val in event.items():
            # 0. 数据域路由器日志
            if node_name == "Domain_Router_Node" and "data_domain" in node_val:
                route_strategy = node_val.get("route_strategy") or "single_domain"
                elapsed_ms = node_val.get("domain_elapsed_ms")
                elapsed_text = f"，耗时: {int(elapsed_ms)}ms" if isinstance(elapsed_ms, (int, float)) else ""
                yield self._format_sse(
                    SseEventType.THINKING.value,
                    f"数据域路由: {node_val.get('data_domain')} "
                    f"(置信度: {float(node_val.get('domain_confidence', 0.0)):.2f}, 来源: {node_val.get('domain_route_source', 'unknown')}, 策略: {route_strategy}{elapsed_text})"
                )
                continue

            # 1. 意图路由器的决策日志
            if node_name == "Intent_Router_Node" and "intent" in node_val:
                elapsed_ms = node_val.get("intent_elapsed_ms")
                elapsed_text = f"，耗时: {int(elapsed_ms)}ms" if isinstance(elapsed_ms, (int, float)) else ""
                yield self._format_sse(
                    SseEventType.THINKING.value,
                    f"智能路由指派: {node_val['intent']} "
                    f"(置信度: {node_val.get('intent_confidence', 0):.2f}{elapsed_text})"
                )
                continue

            # 2. 规划器日志
            if node_name == "Parent_Planner_Node" and "task_list" in node_val:
                planner_source = node_val.get("planner_source") or "unknown"
                planner_elapsed = node_val.get("planner_elapsed_ms")
                elapsed_text = f"，耗时: {int(planner_elapsed)}ms" if isinstance(planner_elapsed, (int, float)) else ""
                tasks = node_val.get("task_list") or []
                yield self._format_sse(
                    SseEventType.THINKING.value,
                    f"任务拆解完成: {len(tasks)} 个子任务 (来源: {planner_source}{elapsed_text})"
                )
                continue

            # DAG 调度器决策日志
            if node_name == "dispatcher_node" and "active_tasks" in node_val:
                count = len(node_val.get("active_tasks", []))
                wave = node_val.get("current_wave", "?")
                if count > 0:
                    yield self._format_sse(SseEventType.THINKING.value, f"🚀 DAG 第 {wave} 波次：派发 {count} 个并行子任务")
                continue

            # Worker 执行耗时日志：让用户在“思考过程”里直接看到每个子任务的耗时。
            if node_name == "worker_node" and "worker_results" in node_val:
                worker_results = node_val.get("worker_results") or []
                for worker_item in worker_results:
                    if not isinstance(worker_item, dict):
                        continue
                    task_id = str(worker_item.get("task_id") or "")
                    agent_name = str(worker_item.get("agent") or "")
                    elapsed_ms = worker_item.get("elapsed_ms")
                    if isinstance(elapsed_ms, (int, float)):
                        label = f"{task_id}:{agent_name}" if task_id and agent_name else (task_id or agent_name or "worker")
                        yield self._format_sse(
                            SseEventType.THINKING.value,
                            f"⏱️ 子任务[{label}]执行耗时: {int(elapsed_ms)}ms",
                        )
                continue

            # 3. Agent 节点返回的消息 — 提取操作日志
            if not isinstance(node_val, dict) or "messages" not in node_val:
                continue

            for msg in node_val.get("messages", []):
                if not isinstance(msg, AIMessage):
                    continue
                # 从 response_metadata 中提取 _agent_node 收集的操作日志
                metadata = getattr(msg, "response_metadata", {}) or {}
                operation_logs = metadata.get("operation_logs", [])
                for log_entry in operation_logs:
                    yield self._format_sse(SseEventType.THINKING.value, log_entry)

                # 如果这是一个由于出错等原因人工拼凑的兜底消息（未经历正常 LLM stream 过程）
                # 需将其直接作为 stream 推送给主面板，避免用户界面空白。
                # 注意：aggregator_node 在“确定性聚合”模式下不会走 messages 流，因此允许 force_emit 输出。
                should_emit_synthetic = metadata.get("synthetic") and bool(msg.content)
                if should_emit_synthetic and (node_name != "aggregator_node" or metadata.get("force_emit")):
                    visible_content = self._strip_internal_router_json_prefix(msg.content)
                    if isinstance(visible_content, str) and visible_content.strip():
                        yield self._format_sse(SseEventType.STREAM.value, visible_content)

    def _extract_final_answer_from_supervisor_state(self, supervisor_graph: Any, session_id: str) -> str:
        """从 Supervisor 最终状态提取最后一条可展示回答，作为流式兜底。"""
        state_config = {"configurable": {"thread_id": session_id}}
        try:
            snapshot = supervisor_graph.get_state(state_config)
        except Exception as exc:
            log.warning(f"最终状态兜底读取失败，已跳过: {exc}")
            return ""

        values = getattr(snapshot, "values", {}) or {}
        direct_answer = self._normalize_visible_answer_text(values.get("direct_answer"))
        if direct_answer:
            return direct_answer

        messages = values.get("messages") or []
        for msg in reversed(messages):
            if not isinstance(msg, AIMessage):
                continue
            visible_text = self._normalize_visible_answer_text(getattr(msg, "content", ""))
            if visible_text:
                return visible_text
        return ""

    def _normalize_visible_answer_text(self, content: Any) -> str:
        """将任意 message content 清洗为最终可展示文本。"""
        text = self._message_content_to_text(content).strip()
        if not text:
            return ""
        if self._is_internal_router_json(text):
            return ""
        return self._strip_internal_router_json_prefix(text).strip()

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        """将 message.content 的不同结构统一转换为文本。"""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part)
        if isinstance(content, dict):
            text = content.get("text") or content.get("content")
            if isinstance(text, str):
                return text
            return json.dumps(content, ensure_ascii=False)
        return str(content)

    @staticmethod
    def _extract_interrupt_from_updates(event: dict) -> Optional[Tuple[str, dict]]:
        """从 Supervisor updates 事件中提取子节点透传的 interrupt payload。"""
        for node_name, node_val in event.items():
            if not isinstance(node_val, dict):
                continue
            payload = node_val.get("interrupt_payload") or node_val.get("interrupt")
            if payload:
                return node_name, payload
        return None

    def _scan_subgraph_interrupts(
        self,
        session_id: str,
        *,
        effective_config: Dict[str, Any],
        candidate_agent_names: Optional[List[str]] = None,
    ) -> Optional[dict]:
        """
        读取子图快照并检查是否存在未处理的 interrupt。

        candidate_agent_names:
        - 传入时仅扫描指定 Agent，避免全量实例化；
        - 为空时回退到全量扫描（兼容旧逻辑）。
        """
        model, _ = create_model_from_config(**effective_config)
        names = candidate_agent_names or list(agent_classes.keys())
        for name in names:
            info = agent_classes.get(name)
            if not info:
                continue
            agent = info.cls(AgentRequest(user_input="", model=model, session_id=session_id, subgraph_id=name))
            snapshot = self._safe_get_agent_state(agent, name, "中断扫描")
            if not snapshot:
                continue
            top_interrupts = getattr(snapshot, "interrupts", None) or []
            if top_interrupts:
                payload = getattr(top_interrupts[0], "value", top_interrupts[0])
                formatted = self._format_interrupt_payload(session_id, payload)
                formatted["agent_name"] = formatted.get("agent_name") or name
                return formatted
            tasks = getattr(snapshot, "tasks", None) or []
            for t in tasks:
                interrupts = getattr(t, "interrupts", None) or []
                if not interrupts:
                    continue
                # 取第一个未处理中断
                first_interrupt = interrupts[0]
                payload = getattr(first_interrupt, "value", first_interrupt)
                formatted = self._format_interrupt_payload(session_id, payload)
                formatted["agent_name"] = formatted.get("agent_name") or name
                return formatted
        return None

    def _register_interrupts(self, session_id: str, payload: dict, effective_config: Dict[str, Any]):
        """将需要人工审核的 Interrupt 事件注册到 InterruptService，等待前端提交审批结果。"""
        payload_agent = payload.get("agent_name")
        checkpoint_meta: Dict[str, Any] = {}
        if payload_agent and payload_agent in agent_classes:
            try:
                model, _ = create_model_from_config(**effective_config)
                agent = agent_classes[payload_agent].cls(
                    AgentRequest(user_input="", model=model, session_id=session_id, subgraph_id=payload_agent)
                )
                snapshot = self._safe_get_agent_state(agent, payload_agent, "注册审批元数据")
                if snapshot:
                    snap_cfg = getattr(snapshot, "config", {}) or {}
                    if isinstance(snap_cfg, dict):
                        conf = snap_cfg.get("configurable", {}) or {}
                        if isinstance(conf, dict):
                            checkpoint_meta = {
                                "checkpoint_id": conf.get("checkpoint_id"),
                                "checkpoint_ns": conf.get("checkpoint_ns"),
                            }
            except Exception as e:
                log.warning(f"读取中断快照元数据失败: {e}")

        for index, req in enumerate(payload.get("action_requests", [])):
            if not req.get("name"):
                continue
            msg_id = req.get("id") or self._approval_message_id(session_id, req, index)
            interrupt_service.register_pending_approval(
                session_id=session_id,
                message_id=msg_id,
                action_name=req["name"],
                action_args=req.get("args", {}),
                description=req.get("description", ""),
                agent_name=payload_agent,
                subgraph_thread_id=f"{session_id}_{payload_agent}" if payload_agent else None,
                checkpoint_id=checkpoint_meta.get("checkpoint_id"),
                checkpoint_ns=checkpoint_meta.get("checkpoint_ns"),
            )

    @staticmethod
    def _approval_message_id(session_id: str, req: Dict[str, Any], index: int = 0) -> str:
        """为缺失 id 的审批请求生成稳定 message_id，保证前后端可对齐。"""
        stable = json.dumps(
            {"name": req.get("name"), "args": req.get("args", {}), "index": index},
            sort_keys=True,
            ensure_ascii=False,
        )
        digest = hashlib.md5(stable.encode("utf-8")).hexdigest()[:10]
        return f"{session_id}_{req.get('name', 'approval')}_{digest}"

    def _format_interrupt_payload(self, session_id: str, val: Any) -> dict:
        """将 LangGraph 的 Interrupt 值标准化为前端可消费的字典格式。"""
        if isinstance(val, dict):
            v = val
        elif hasattr(val, "__dict__"):
            v = val.__dict__
        else:
            v = {"message": str(val), "allowed_decisions": list(DEFAULT_ALLOWED_DECISIONS), "action_requests": []}
        action_requests = []
        for idx, req in enumerate(v.get("action_requests", [])):
            normalized = dict(req)
            if not normalized.get("id"):
                normalized["id"] = self._approval_message_id(session_id, normalized, idx)
            action_requests.append(normalized)
        return {
            "message": v.get("message", DEFAULT_INTERRUPT_MESSAGE),
            "allowed_decisions": v.get("allowed_decisions", list(DEFAULT_ALLOWED_DECISIONS)),
            "action_requests": action_requests,
            "agent_name": v.get("agent_name"),
        }

    @staticmethod
    def _format_sse(type_: str, content: str) -> str:
        """将事件类型和内容格式化为标准的 SSE（Server-Sent Events）字符串。"""
        payload = {
            SsePayloadField.TYPE.value: type_,
            SsePayloadField.CONTENT.value: content,
        }
        return f"event: {type_}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @staticmethod
    def _strip_internal_router_json_prefix(content: Any) -> str:
        """
        去除回复开头的内部路由 JSON 前缀，避免污染用户可见正文。
        兼容“纯 JSON”与“JSON + 正文”的场景。
        """
        if not isinstance(content, str):
            return ""
        raw_text = content
        trimmed = raw_text.lstrip()
        if not trimmed.startswith("{"):
            return raw_text
        try:
            decoder = json.JSONDecoder()
            parsed, end_idx = decoder.raw_decode(trimmed)
        except Exception:
            return raw_text
        if not isinstance(parsed, dict):
            return raw_text
        is_domain_router = "data_domain" in parsed and "confidence" in parsed
        is_intent_router = "intent" in parsed and "confidence" in parsed
        if not (is_domain_router or is_intent_router):
            return raw_text
        return trimmed[end_idx:].lstrip()

    @staticmethod
    def _strip_router_json_prefix_across_chunks(
        content: Any,
        prefix_buffer: str,
        prefix_done: bool,
    ) -> Tuple[str, str, bool]:
        """
        跨 chunk 去掉“路由 JSON + 正文”里的 JSON 前缀。

        背景：
        - 某些模型会先流出 `{"data_domain":...}` 或 `{"intent":...}`，再输出正文；
        - 若 JSON 被拆成多个 chunk，单次解析会失败，导致前缀泄漏到用户界面。
        """
        if prefix_done:
            return (content if isinstance(content, str) else "", prefix_buffer, prefix_done)

        text = content if isinstance(content, str) else ""
        if not text and not prefix_buffer:
            return "", prefix_buffer, prefix_done

        merged = (prefix_buffer or "") + text
        trimmed = merged.lstrip()

        # 非 JSON 起始：说明不存在路由前缀，后续可直接透传。
        if not trimmed.startswith("{"):
            return merged, "", True

        try:
            decoder = json.JSONDecoder()
            parsed, end_idx = decoder.raw_decode(trimmed)
            if isinstance(parsed, dict) and (
                ("data_domain" in parsed and "confidence" in parsed)
                or ("intent" in parsed and "confidence" in parsed)
            ):
                remainder = trimmed[end_idx:].lstrip()
                return remainder, "", True
            # 是普通 JSON，不做拦截。
            return merged, "", True
        except Exception:
            # 可能是被拆分的 JSON，先缓存等待下一个 chunk；避免无限缓存。
            if len(trimmed) <= 512:
                return "", trimmed, False
            return merged, "", True

    @staticmethod
    def _is_internal_router_json(content: Any) -> bool:
        """
        过滤路由器中间 JSON，避免直接展示给用户主回答区。
        典型格式：
        {"data_domain":"GENERAL","confidence":0.95}
        {"intent":"CHAT","confidence":0.94,"is_complex":false,...}
        """
        if not isinstance(content, str):
            return False
        text = content.strip()
        if not text.startswith("{") or not text.endswith("}"):
            return False
        try:
            decoder = json.JSONDecoder()
            data, end_idx = decoder.raw_decode(text)
        except Exception:
            return False
        if end_idx != len(text):
            return False
        if not isinstance(data, dict):
            return False
        if "data_domain" in data and "confidence" in data:
            return True
        if "intent" in data and "confidence" in data:
            return True
        return False
