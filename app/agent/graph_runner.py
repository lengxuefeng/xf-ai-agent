import hashlib
import json
from typing import Any, Dict, Generator, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from agent.graph_state import AgentRequest
from agent.graphs.supervisor import create_graph as create_supervisor_graph
from agent.llm.unified_loader import create_model_from_config
from agent.registry import agent_classes
from services.interrupt_service import interrupt_service
from utils.custom_logger import get_logger

log = get_logger(__name__)

"""
【模块说明】
API 接口的直接消费者。
负责拉起 Supervisor，将执行过程包装为 SSE 事件推给前端。
同时负责拦截子图 Bubble Up 的 Interrupt（挂起）事件并进行下发。
"""


class GraphRunner:
    def __init__(self, model_config: Optional[dict] = None):
        self.model_config = model_config or {}
        self._supervisor_cache: Dict[str, Any] = {}
        self._effective_config: Dict[str, Any] = {}

    @staticmethod
    def _config_key(model_config: dict) -> str:
        stable = json.dumps(model_config or {}, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(stable.encode("utf-8")).hexdigest()

    def _get_supervisor(self, model_config: dict):
        cache_key = self._config_key(model_config)
        if cache_key not in self._supervisor_cache:
            self._supervisor_cache[cache_key] = create_supervisor_graph(model_config)
        return self._supervisor_cache[cache_key]

    def stream_run(self, user_input: str, session_id: str, model_config: Optional[dict] = None,
                   history_messages: Optional[list] = None) -> Generator[str, None, None]:
        history_messages = history_messages or []
        effective_config = {**self.model_config, **(model_config or {})}
        self._effective_config = effective_config
        graph = self._get_supervisor(effective_config)

        # 1. 检测前端是否发起了审批恢复操作
        resume_payload = self._check_pending_approval(session_id)
        if resume_payload:
            yield self._format_sse("response_start", "")
            yield self._format_sse("thinking", "检测到审批结果，正在恢复执行...")
            yield from self._handle_resume(session_id, resume_payload, graph)
            yield self._format_sse("response_end", "")
            return

        if user_input == "[RESUME]" or not user_input:
            yield self._format_sse("error", "参数无效或无待恢复任务")
            return

        messages = self._build_initial_messages(history_messages, user_input)

        # 2. RAG 上下文注入
        if str(effective_config.get("rag_enabled", "")).lower() == "true":
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

        try:
            yield self._format_sse("response_start", "")

            # 主图执行
            for event in graph.stream(inputs, config=config, stream_mode="updates"):
                yield from self._process_supervisor_event(event)

            # 4. 扫描内部子图是否抛出了中断 (Interrupt Bubble-up)
            interrupt_event = self._scan_subgraph_interrupts(session_id)
            if interrupt_event:
                self._register_interrupts(session_id, interrupt_event)
                yield self._format_sse("interrupt", json.dumps(interrupt_event, ensure_ascii=False))

            yield self._format_sse("response_end", "")

        except Exception as exc:
            log.exception(f"执行异常: {exc}")
            yield self._format_sse("error", str(exc))

    @staticmethod
    def _build_initial_messages(history: list, user_input: str) -> List[BaseMessage]:
        messages: List[BaseMessage] = []
        for msg in history:
            if msg.get("user_content"): messages.append(HumanMessage(content=msg["user_content"]))
            if msg.get("model_content"): messages.append(AIMessage(content=msg["model_content"], name=msg.get("name")))
        messages.append(HumanMessage(content=user_input))
        return messages

    def _retrieve_rag_context(self, user_input: str, model_config: dict) -> Tuple[str, List[str]]:
        try:
            from agent.rag.vector_store import vector_store_service
            docs = vector_store_service.search_documents(user_input,
                                                         threshold=float(model_config.get("similarity_threshold", 0.7)))
            return vector_store_service.get_context(docs), [str(d.metadata.get("source")) for d in docs if
                                                            hasattr(d, "metadata")]
        except Exception:
            return "", []

    def _check_pending_approval(self, session_id: str) -> Optional[Command]:
        approvals = interrupt_service.pending_approvals.get(session_id, {})
        for _, data in list(approvals.items()):
            if data.get("status") == "pending": continue
            approvals.clear()
            return Command(resume="approve" if data["status"] == "approve" else {"action": "reject"})
        return None

    def _handle_resume(self, session_id: str, command: Command, supervisor_graph) -> Generator[str, None, None]:
        target_agent = None
        model, _ = create_model_from_config(**self._effective_config)

        # 寻找挂起状态的子 Agent
        for name, info in agent_classes.items():
            agent = info.cls(AgentRequest(user_input="", model=model, session_id=session_id, subgraph_id=name))
            if agent.get_state().tasks and agent.get_state().tasks[0].interrupts:
                target_agent = agent
                break

        if not target_agent:
            yield self._format_sse("error", "未找到处于中断状态的子任务")
            return

        # 携带 Config 恢复执行
        subgraph_config = {"configurable": {"thread_id": session_id, "checkpoint_ns": target_agent.subgraph_id}}
        final_msgs = []
        try:
            for event in target_agent.graph.stream(command, config=subgraph_config, stream_mode="updates"):
                for _, val in event.items():
                    if "messages" in val:
                        for msg in val["messages"]:
                            if isinstance(msg, AIMessage):
                                final_msgs.append(msg)
                                yield self._format_sse("message", msg.content)
        except Exception as exc:
            yield self._format_sse("error", f"任务恢复失败: {str(exc)}")
            return

        # 状态回填 Supervisor
        if final_msgs:
            sup_config = {"configurable": {"thread_id": session_id}}
            if supervisor_graph.get_state(sup_config).values:
                supervisor_graph.update_state(sup_config, {"messages": final_msgs})

    def _process_supervisor_event(self, event: dict) -> Generator[str, None, None]:
        for node_name, node_val in event.items():
            if node_name == "supervisor" and "next" in node_val:
                yield self._format_sse("thinking",
                                       f"路由指派: {node_val['next']} (置信度: {node_val.get('routing_confidence', 0):.2f})")
            if "messages" in node_val:
                for msg in node_val["messages"]:
                    if isinstance(msg, AIMessage): yield self._format_sse("message", msg.content)

    def _scan_subgraph_interrupts(self, session_id: str) -> Optional[dict]:
        model, _ = create_model_from_config(**self._effective_config)
        for name, info in agent_classes.items():
            agent = info.cls(AgentRequest(user_input="", model=model, session_id=session_id, subgraph_id=name))
            snapshot = agent.get_state()
            if snapshot.tasks and snapshot.tasks[0].interrupts:
                return self._format_interrupt_payload(snapshot.tasks[0].interrupts[0].value)
        return None

    def _register_interrupts(self, session_id: str, payload: dict):
        for req in payload.get("action_requests", []):
            interrupt_service.register_pending_approval(
                session_id, req.get("id") or f"{session_id}_{req['name']}",
                req["name"], req["args"], req.get("description", "")
            )

    @staticmethod
    def _format_interrupt_payload(val: Any) -> dict:
        v = val if isinstance(val, dict) else val.__dict__
        return {"message": v.get("message", "需人工审核"), "allowed_decisions": v.get("allowed_decisions", []),
                "action_requests": v.get("action_requests", [])}

    @staticmethod
    def _format_sse(type_: str, content: str) -> str:
        return f"event: {type_}\ndata: {json.dumps({'type': type_, 'content': content}, ensure_ascii=False)}\n\n"