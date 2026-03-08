import hashlib
import json
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, AIMessageChunk
from langgraph.types import Command

from agent.graph_state import AgentRequest
from agent.graphs.supervisor import create_graph as create_supervisor_graph
from agent.llm.unified_loader import create_model_from_config
from agent.rag.vector_store import vector_store_service
from agent.registry import agent_classes
from services.interrupt_service import interrupt_service
from utils.custom_logger import get_logger
from agent.rules.registry import rule_registry
from agent.rules.actions import handle_action
from utils.date_utils import get_agent_date_context

log = get_logger(__name__)

"""
【模块说明】
API 接口的直接消费者。
负责拉起 Supervisor，将执行过程包装为 SSE 事件推给前端。
同时负责拦截子图 Bubble Up 的 Interrupt（挂起）事件并进行下发。
"""


class GraphRunner:
    """
    图执行器（Graph Runner）—— 整个 AI 对话链路的核心调度中枢。

    职责：
    1. 从 ChatService（服务层）接收用户消息与模型配置
    2. 创建/缓存 Supervisor 图实例，避免重复编译
    3. 组装对话消息历史，注入 RAG 上下文
    4. 以 SSE（Server-Sent Events）格式将图的执行结果实时推送给前端
    5. 拦截子图冒泡的 Interrupt（中断审批）事件并下发给前端

    核心流程: ChatService → GraphRunner.stream_run() → Supervisor Graph → Sub-Agent
    """

    def __init__(self, model_config: Optional[dict] = None):
        """
        初始化图执行器。

        Args:
            model_config: 默认模型配置字典，包含 model、model_service、model_key 等字段。
        """
        self.model_config = model_config or {}
        # Supervisor 图的缓存，以配置哈希为 key，避免每次请求都重新编译图
        self._supervisor_cache: Dict[str, Any] = {}
        # 当前生效的模型配置（合并默认配置与请求级配置后的结果）
        self._effective_config: Dict[str, Any] = {}

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
            history_messages: Optional[list] = None
    ) -> Generator[str, None, None]:
        """
        核心流式执行器（使用后台线程实现真正的实时流）。
        由于 _agent_node 是同步阻塞执行的，如果不使用多线程，期间产生的底层日志
        (log.info等) 会被缓存在队列中直到节点结束才能下发。
        采用生产者-消费者模型，将图的执行放入后台线程，主线程只负责从队列取数据下发 SSE。
        """
        history_messages = history_messages or []
        effective_config = {**self.model_config, **(model_config or {})}
        self._effective_config = effective_config
        graph = self._get_supervisor(effective_config)

        # 1. 检测前端是否发起了审批恢复操作
        resume_meta = self._check_pending_approval(session_id)
        if resume_meta:
            yield self._format_sse("response_start", "")
            yield self._format_sse("thinking", "检测到审批结果，正在恢复执行...")
            yield from self._handle_resume(session_id, resume_meta, graph)
            yield self._format_sse("response_end", "")
            return

        if user_input == "[RESUME]" or not user_input:
            yield self._format_sse("error", "参数无效或无待恢复任务")
            return

        # ==========================================
        # Phase 2: 第一道防线 - 前置规则拦截 (Zero-LLM, Zero-Graph)
        # ==========================================

        import re

        user_text_lower = user_input.lower().strip()
        # 规则引擎最大扫描句长（超出此长度的句子直接交给 LLM）
        _MAX_RULE_SCAN_LEN = 60
        # 覆盖率估算：平均每个简单意图约占 N 个字符
        _CHARS_PER_INTENT = 15
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
                    yield self._format_sse("response_start", "")
                    yield self._format_sse("thinking", f"⚡ 极速拦截复合意图：命中 [{ids_str}]")
                    yield self._format_sse("stream", combined_resp)
                    yield self._format_sse("response_end", "")
                    log.info(f"Pre-Graph 复合拦截生效: [{ids_str}] - {user_text_lower}")
                    return  # 彻底中断！不进入 LangGraph

        messages = self._build_initial_messages(history_messages, user_input)

        # 2. RAG 上下文注入
        rag_enabled = effective_config.get("rag_enabled", False)
        if isinstance(rag_enabled, str):
            rag_enabled = rag_enabled.lower() == "true"

        if rag_enabled:
            rag_context, rag_sources = self._retrieve_rag_context(user_input, effective_config)
            if rag_context:
                insert_idx = len(messages) - 1 if isinstance(messages[-1], HumanMessage) else len(messages)
                messages.insert(insert_idx, SystemMessage(content=f"参考以下知识库回答:\n{rag_context}"))
                yield self._format_sse("thinking", f"RAG 命中 {len(rag_sources)} 个来源")

        # 3. 构造顶级调用 Config
        config = {
            "configurable": {"thread_id": session_id},
            "run_name": f"Supervisor_Run_{session_id}"  # 给 LangSmith 打标签
        }
        inputs = {"messages": messages, "session_id": session_id, "llm_config": effective_config}

        import queue
        import threading
        # 统一事件队列，容纳底层的 python text log 和上层的 graph event
        event_queue = queue.Queue()

        def log_interceptor(sse_message: str):
            event_queue.put(("log", sse_message))

        from utils.custom_logger import CustomLogger
        CustomLogger.add_global_sse_callback(log_interceptor)

        def graph_worker():
            try:
                for ev_type, ev in graph.stream(inputs, config=config, stream_mode=["updates", "messages"]):
                    event_queue.put(("graph", (ev_type, ev)))
                event_queue.put(("done", None))
            except Exception as e:
                # 兼容最新 LangGraph 版本：interrupt() 会抛出特俗异常提前终止流
                err_msg = str(e)
                if "Interrupt(" in err_msg or e.__class__.__name__ == "GraphInterrupt":
                    log.info(f"Graph 节点抛出执行挂起 (Interrupt): {err_msg}")
                    event_queue.put(("done", None))
                else:
                    log.exception(f"执行异常: {e}")
                    event_queue.put(("error", err_msg))

        # 启动后台图执行线程
        worker_thread = threading.Thread(target=graph_worker, daemon=True)
        worker_thread.start()

        yield self._format_sse("response_start", "")

        try:
            start_ts = time.time()
            last_event_ts = start_ts
            last_heartbeat_ts = 0.0
            interrupt_emitted = False
            idle_heartbeat_sec = 10.0
            idle_timeout_sec = 45.0
            hard_timeout_sec = 240.0

            # 在主线程中阻塞监听队列并下发
            while True:
                now = time.time()

                # 总执行超时保护：避免模型或外部网络异常导致永不返回
                if now - start_ts > hard_timeout_sec:
                    yield self._format_sse("error", "处理超时，请稍后重试。")
                    break

                try:
                    item_type, item_data = event_queue.get(timeout=1.0)
                except queue.Empty:
                    now = time.time()
                    # Worker 还活着：定时回传心跳，避免前端“卡住无反馈”
                    if worker_thread.is_alive():
                        if now - last_event_ts >= idle_timeout_sec:
                            yield self._format_sse("error", "长时间未收到模型响应，请重试。")
                            break
                        if now - last_heartbeat_ts >= idle_heartbeat_sec:
                            yield self._format_sse("thinking", "系统正在处理中，请稍候...")
                            last_heartbeat_ts = now
                        continue

                    # Worker 已退出但未发 done/error，返回兜底错误避免静默
                    yield self._format_sse("error", "任务已中断，请重试。")
                    break

                last_event_ts = time.time()

                if item_type == "done":
                    break

                elif item_type == "error":
                    yield self._format_sse("error", item_data)
                    break

                elif item_type == "log":
                    # 直接透传底层格式化好的 SSE payload
                    yield item_data

                elif item_type == "graph":
                    event_type, event = item_data
                    if event_type == "messages":
                        msg_chunk, metadata = event
                        if isinstance(msg_chunk, AIMessageChunk):
                            node_name = metadata.get("langgraph_node", "")

                            # 纯文本推送：拦截内部路由与中间节点，只允许最终节点或单兵节点往 UI 推流
                            _silenced_text_nodes = {
                                "Rule_Engine_Node", "Intent_Router_Node",
                                "Parent_Planner_Node", "dispatcher_node",
                                "worker_node", "reducer_node"
                            }

                            if node_name not in _silenced_text_nodes:
                                if msg_chunk.content and not getattr(msg_chunk, "tool_calls", None):
                                    # 允许主路径与各专业 Agent 的正文流，避免“有思考无最终答复”
                                    allowed_names = {None, "", "ChatAgent", "Aggregator", *agent_classes.keys()}
                                    if getattr(msg_chunk, "name", None) in allowed_names:
                                        yield self._format_sse("stream", msg_chunk.content)

                            # 工具调用推送：无论哪个节点，只要拉起工具，必然展示给用户
                            if getattr(msg_chunk, "tool_calls", None):
                                for tc in msg_chunk.tool_calls:
                                    yield self._format_sse("thinking", f"🔧 正在调度: {tc.get('name', '...')}")
                    elif event_type == "updates":
                        inband_interrupt = self._extract_interrupt_from_updates(event)
                        if inband_interrupt and not interrupt_emitted:
                            source_node, inband_payload = inband_interrupt
                            payload = dict(inband_payload) if isinstance(inband_payload, dict) else inband_payload
                            if isinstance(payload, dict) and not payload.get("agent_name") and source_node in agent_classes:
                                payload["agent_name"] = source_node
                            interrupt_event = self._format_interrupt_payload(session_id, payload)
                            self._register_interrupts(session_id, interrupt_event)
                            yield self._format_sse("thinking", "检测到需要人工审核，请点击“批准/拒绝”后继续。")
                            yield self._format_sse("interrupt", json.dumps(interrupt_event, ensure_ascii=False))
                            interrupt_emitted = True
                        yield from self._process_supervisor_event(event)

            # 4. 扫描内部子图是否抛出了中断 (Interrupt Bubble-up)
            if not interrupt_emitted:
                interrupt_event = self._scan_subgraph_interrupts(session_id)
                if interrupt_event:
                    self._register_interrupts(session_id, interrupt_event)
                    yield self._format_sse("thinking", "检测到需要人工审核，请点击“批准/拒绝”后继续。")
                    yield self._format_sse("interrupt", json.dumps(interrupt_event, ensure_ascii=False))

            yield self._format_sse("response_end", "")

        finally:
            CustomLogger.remove_global_sse_callback(log_interceptor)
            if worker_thread.is_alive():
                log.warning("Graph worker 线程仍在运行，已由主流程提前结束返回。")

    @staticmethod
    def _build_initial_messages(history: list, user_input: str) -> List[BaseMessage]:
        """
        将前端传入的历史消息列表转换为 LangChain 的 BaseMessage 列表。

        Args:
            history: MongoDB 中存储的历史消息（字典列表）
            user_input: 当前用户输入

        Returns:
            LangChain 消息对象列表，末尾为当前用户输入
        """
        messages: List[BaseMessage] = [SystemMessage(content=get_agent_date_context())]
        for msg in history:
            if msg.get("user_content"): messages.append(HumanMessage(content=msg["user_content"]))
            if msg.get("model_content"): messages.append(AIMessage(content=msg["model_content"], name=msg.get("name")))
        messages.append(HumanMessage(content=user_input))
        return messages

    def _retrieve_rag_context(self, user_input: str, model_config: dict) -> Tuple[str, List[str]]:
        """检索 RAG 知识库，返回拼接后的上下文文本和来源列表。失败时静默降级为空。"""
        try:
            docs = vector_store_service.search_documents(user_input,
                                                         threshold=float(model_config.get("similarity_threshold", 0.7)))
            return vector_store_service.get_context(docs), [str(d.metadata.get("source")) for d in docs if
                                                            hasattr(d, "metadata")]
        except Exception:
            return "", []

    def _check_pending_approval(self, session_id: str) -> Optional[dict]:
        """检查当前会话是否有前端已提交的审批结果，若有则构造恢复元信息。"""
        approval = interrupt_service.fetch_latest_resolved_approval(session_id)
        if approval:
            action = "approve" if approval.get("status") == "approve" else "reject"
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

    def _handle_resume(self, session_id: str, resume_meta: dict, supervisor_graph) -> Generator[str, None, None]:
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
        model, _ = create_model_from_config(**self._effective_config)
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
                snapshot = candidate.get_state()
                if self._snapshot_has_interrupt(snapshot):
                    target_agent = candidate

        # 寻找挂起状态的子 Agent
        if not target_agent:
            for name, info in agent_classes.items():
                agent = info.cls(AgentRequest(user_input="", model=model, session_id=session_id, subgraph_id=name))
                snapshot = agent.get_state()
                if self._snapshot_has_interrupt(snapshot):
                    target_agent = agent
                    break

        if not target_agent:
            # 兜底：若找不到挂起子图，但审批的是 SQL 执行，直接执行已审批 SQL，避免前端空转报错。
            if decision == "approve" and resume_meta.get("action_name") == "execute_sql":
                sql_args = resume_meta.get("action_args") or {}
                approved_sql = sql_args.get("sql")
                if approved_sql:
                    try:
                        from agent.tools.sql_tools import execute_sql, format_sql_result_for_user
                        fallback_result = execute_sql(approved_sql, domain="LOCAL_DB")
                        formatted_result = format_sql_result_for_user(approved_sql, fallback_result)
                        yield self._format_sse("stream", formatted_result)
                        if message_id:
                            interrupt_service.mark_approval_consumed(session_id, message_id)
                        return
                    except Exception as sql_exc:
                        yield self._format_sse("error", f"未找到中断任务且 SQL 兜底执行失败: {sql_exc}")
                        return
            yield self._format_sse("error", "未找到处于中断状态的子任务")
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
                                yield self._format_sse("stream", msg.content)
        except Exception as exc:
            err_msg = str(exc)
            if "Interrupt(" in err_msg or exc.__class__.__name__ == "GraphInterrupt":
                interrupt_event = self._scan_subgraph_interrupts(session_id)
                if interrupt_event:
                    if message_id:
                        interrupt_service.mark_approval_consumed(session_id, message_id)
                    self._register_interrupts(session_id, interrupt_event)
                    yield self._format_sse("thinking", "恢复执行后仍需人工审核，请继续审批。")
                    yield self._format_sse("interrupt", json.dumps(interrupt_event, ensure_ascii=False))
                else:
                    yield self._format_sse("error", "恢复执行进入审批状态，但未获取到审批详情，请重试。")
            else:
                log.exception(f"Resume non-interrupt failure: {err_msg}")
                # 兜底策略：部分模型服务在 resume 场景会报 messages 参数非法(1214)。
                # 若审批的是 execute_sql，直接执行已审批 SQL，避免链路中断。
                if (
                    "1214" in err_msg
                    and resume_meta.get("action_name") == "execute_sql"
                    and decision == "approve"
                ):
                    sql_args = resume_meta.get("action_args") or {}
                    approved_sql = sql_args.get("sql")
                    if approved_sql:
                        try:
                            from agent.tools.sql_tools import execute_sql, format_sql_result_for_user
                            fallback_result = execute_sql(approved_sql, domain="LOCAL_DB")
                            formatted_result = format_sql_result_for_user(approved_sql, fallback_result)
                            yield self._format_sse(
                                "stream",
                                formatted_result
                            )
                            if message_id:
                                interrupt_service.mark_approval_consumed(session_id, message_id)
                            return
                        except Exception as sql_exc:
                            yield self._format_sse("error", f"任务恢复失败且 SQL 兜底执行失败: {sql_exc}")
                            return
                yield self._format_sse("error", f"任务恢复失败: {err_msg}")
            return

        if not final_msgs:
            if decision == "reject":
                if message_id:
                    interrupt_service.mark_approval_consumed(session_id, message_id)
                yield self._format_sse("stream", "已拒绝本次敏感操作。")
                return
            interrupt_event = self._scan_subgraph_interrupts(session_id)
            if interrupt_event:
                if message_id:
                    interrupt_service.mark_approval_consumed(session_id, message_id)
                self._register_interrupts(session_id, interrupt_event)
                yield self._format_sse("thinking", "恢复执行后仍需人工审核，请继续审批。")
                yield self._format_sse("interrupt", json.dumps(interrupt_event, ensure_ascii=False))
            else:
                yield self._format_sse("error", "恢复执行未生成结果，请重试。")
            return

        if message_id:
            interrupt_service.mark_approval_consumed(session_id, message_id)

        # 状态回填 Supervisor
        if final_msgs:
            sup_config = {"configurable": {"thread_id": session_id}}
            if supervisor_graph.get_state(sup_config).values:
                supervisor_graph.update_state(sup_config, {"messages": final_msgs})

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
                yield self._format_sse(
                    "thinking",
                    f"数据域路由: {node_val.get('data_domain')} "
                    f"(置信度: {float(node_val.get('domain_confidence', 0.0)):.2f}, 来源: {node_val.get('domain_route_source', 'unknown')})"
                )
                continue

            # 1. 意图路由器的决策日志
            if node_name == "Intent_Router_Node" and "intent" in node_val:
                yield self._format_sse("thinking",
                                       f"智能路由指派: {node_val['intent']} (置信度: {node_val.get('intent_confidence', 0):.2f})")
                continue

            # DAG 调度器决策日志
            if node_name == "dispatcher_node" and "active_tasks" in node_val:
                count = len(node_val.get("active_tasks", []))
                wave = node_val.get("current_wave", "?")
                if count > 0:
                    yield self._format_sse("thinking", f"🚀 DAG 第 {wave} 波次：派发 {count} 个并行子任务")
                continue

            # 2. Agent 节点返回的消息 — 提取操作日志
            if not isinstance(node_val, dict) or "messages" not in node_val:
                continue

            for msg in node_val.get("messages", []):
                if not isinstance(msg, AIMessage):
                    continue
                # 从 response_metadata 中提取 _agent_node 收集的操作日志
                metadata = getattr(msg, "response_metadata", {}) or {}
                operation_logs = metadata.get("operation_logs", [])
                for log_entry in operation_logs:
                    yield self._format_sse("thinking", log_entry)

                # 如果这是一个由于出错等原因人工拼凑的兜底消息（未经历正常 LLM stream 过程）
                # 需将其直接作为 stream 推送给主面板，避免用户界面空白。
                # 过滤掉 aggregator_node，因为它自己的内嵌大模型调用已经触发过 stream 流了，这里如果再推一次会导致重复输出
                if metadata.get("synthetic") and msg.content and node_name != "aggregator_node":
                    yield self._format_sse("stream", msg.content)

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

    def _scan_subgraph_interrupts(self, session_id: str) -> Optional[dict]:
        """图执行结束后，遍历所有子 Agent 的 Checkpoint 状态，检查是否有未处理的 Interrupt 冒泡。"""
        model, _ = create_model_from_config(**self._effective_config)
        for name, info in agent_classes.items():
            agent = info.cls(AgentRequest(user_input="", model=model, session_id=session_id, subgraph_id=name))
            snapshot = agent.get_state()
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

    def _register_interrupts(self, session_id: str, payload: dict):
        """将需要人工审核的 Interrupt 事件注册到 InterruptService，等待前端提交审批结果。"""
        payload_agent = payload.get("agent_name")
        checkpoint_meta: Dict[str, Any] = {}
        if payload_agent and payload_agent in agent_classes:
            try:
                model, _ = create_model_from_config(**self._effective_config)
                agent = agent_classes[payload_agent].cls(
                    AgentRequest(user_input="", model=model, session_id=session_id, subgraph_id=payload_agent)
                )
                snapshot = agent.get_state()
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
            v = {"message": str(val), "allowed_decisions": ["approve", "reject"], "action_requests": []}
        action_requests = []
        for idx, req in enumerate(v.get("action_requests", [])):
            normalized = dict(req)
            if not normalized.get("id"):
                normalized["id"] = self._approval_message_id(session_id, normalized, idx)
            action_requests.append(normalized)
        return {
            "message": v.get("message", "需人工审核"),
            "allowed_decisions": v.get("allowed_decisions", ["approve", "reject"]),
            "action_requests": action_requests,
            "agent_name": v.get("agent_name"),
        }

    @staticmethod
    def _format_sse(type_: str, content: str) -> str:
        """将事件类型和内容格式化为标准的 SSE（Server-Sent Events）字符串。"""
        return f"event: {type_}\ndata: {json.dumps({'type': type_, 'content': content}, ensure_ascii=False)}\n\n"
