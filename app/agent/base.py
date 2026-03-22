from abc import ABC, abstractmethod
import re
from typing import Generator, Any, Dict, Optional
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import START, StateGraph
from agent.graph_state import AgentRequest
from agent.graphs.checkpointer import get_checkpointer
from constants.approval_constants import DEFAULT_ALLOWED_DECISIONS, DEFAULT_INTERRUPT_MESSAGE
from config.runtime_settings import AGENT_LOOP_CONFIG, AGENT_LIVE_STREAM_ENABLED
from services.agent_stream_bus import agent_stream_bus
from utils.history_compressor import compress_history_messages
from utils.custom_logger import get_logger
from utils.date_utils import get_agent_date_context, get_current_time_context

log = get_logger(__name__)

"""
所有具体业务智能体（如 YunyouAgent）的抽象基类。

本模块提供了核心的 `BaseAgent` 类，负责：
1. 统一处理来自用户的 AgentRequest 请求封装。
2. 绑定和管理由于多轮对话产生的 Checkpointer 状态持久化。
3. 统一规范流式输出与图执行的生命周期隔离机制。
"""


class BaseAgent(ABC):
    """Agent基类：所有具体业务Agent的抽象父类"""

    @staticmethod
    def _message_text(msg: BaseMessage) -> str:
        """提取消息文本内容"""
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part).strip()
        if isinstance(content, dict):
            text = content.get("text") or content.get("content") or ""
            if isinstance(text, str):
                return text.strip()
        return str(content or "").strip()

    @staticmethod
    def _extract_text_tokens(text: str) -> set[str]:
        """提取轻量关键词用于相关性筛选。"""
        if not text:
            return set()
        min_chars = AGENT_LOOP_CONFIG.context_relevance_min_token_chars
        zh_tokens = re.findall(rf"[\u4e00-\u9fa5]{{{min_chars},}}", text)
        en_tokens = re.findall(rf"[A-Za-z0-9_]{{{min_chars},}}", text.lower())
        return set(zh_tokens + en_tokens)

    @classmethod
    def _extract_history_messages(cls, req: AgentRequest) -> list[BaseMessage]:
        """提取并压缩最近多轮对话消息"""
        state = getattr(req, "state", None) or {}
        if not isinstance(state, dict):
            return []
        messages = state.get("messages")
        if not isinstance(messages, list):
            return []
        valid_messages = [m for m in messages if isinstance(m, BaseMessage)]
        recent_messages = valid_messages[-AGENT_LOOP_CONFIG.context_history_messages:]
        if not recent_messages:
            return []

        # 相关性筛选：保留尾部窗口 + 与当前问题词面有交集的历史，减少旧主题污染。
        current_tokens = cls._extract_text_tokens((req.user_input or "").strip())
        tail_window = max(1, min(AGENT_LOOP_CONFIG.context_relevance_tail_messages, AGENT_LOOP_CONFIG.context_history_messages))
        tail_part = recent_messages[-tail_window:]
        head_part = recent_messages[:-tail_window]
        filtered_messages: list[BaseMessage] = []

        if current_tokens:
            for msg in head_part:
                msg_tokens = cls._extract_text_tokens(cls._message_text(msg))
                if msg_tokens & current_tokens:
                    filtered_messages.append(msg)

        # 尾部窗口只强保留最近用户消息；AI 消息必须与当前问题相关才保留，避免主题漂移。
        for msg in tail_part:
            if isinstance(msg, HumanMessage):
                filtered_messages.append(msg)
                continue
            msg_tokens = cls._extract_text_tokens(cls._message_text(msg))
            if (not current_tokens) or (msg_tokens & current_tokens):
                filtered_messages.append(msg)

        # 兜底：至少保留最后一条用户消息，避免丢上下文锚点。
        if not any(isinstance(m, HumanMessage) for m in filtered_messages):
            for msg in reversed(recent_messages):
                if isinstance(msg, HumanMessage):
                    filtered_messages.append(msg)
                    break

        return compress_history_messages(
            filtered_messages,
            model=getattr(req, "model", None),
            max_tokens=AGENT_LOOP_CONFIG.context_compress_max_tokens,
            max_chars=AGENT_LOOP_CONFIG.context_compress_max_chars,
        )

    @staticmethod
    def _extract_context_summary(req: AgentRequest) -> str:
        """从请求 state 中读取会话摘要，供子 Agent 继承全局上下文。"""
        state = getattr(req, "state", None) or {}
        if not isinstance(state, dict):
            return ""
        summary_text = state.get("context_summary")
        if isinstance(summary_text, str):
            return summary_text.strip()
        return ""

    @staticmethod
    def _extract_interrupt_payload(exc: Exception) -> Optional[dict]:
        """
        尽最大可能从 LangGraph GraphInterrupt 异常对象中提取审批载荷。
        兼容不同版本的异常结构（args / interrupts / value）。
        """
        candidates: list[Any] = []

        if hasattr(exc, "interrupts"):
            interrupts = getattr(exc, "interrupts")
            if interrupts:
                candidates.extend(list(interrupts))

        if hasattr(exc, "value"):
            candidates.append(getattr(exc, "value"))

        if getattr(exc, "args", None):
            candidates.extend(list(exc.args))

        def _unwrap(obj: Any) -> Optional[dict]:
            if obj is None:
                return None

            if isinstance(obj, dict):
                if "action_requests" in obj or "allowed_decisions" in obj or "message" in obj:
                    return obj
                return None

            if isinstance(obj, (list, tuple)):
                for item in obj:
                    payload = _unwrap(item)
                    if payload:
                        return payload
                return None

            # LangGraph Interrupt 对象通常有 value 字段
            if hasattr(obj, "value"):
                return _unwrap(getattr(obj, "value"))

            return None

        for c in candidates:
            payload = _unwrap(c)
            if payload:
                return payload
        return None

    def __init__(self, req: AgentRequest):
        """初始化Agent实例"""
        self.req = req
        self.session_id = req.session_id
        # 子图标识，用于状态隔离
        self.subgraph_id = getattr(req, "subgraph_id", None) or self.__class__.__name__
        self.checkpointer = get_checkpointer(self.subgraph_id)

    @abstractmethod
    def _build_graph(self) -> Runnable:
        """子类必须实现此方法，返回编译后的 StateGraph"""
        pass

    # ------------------------------------------------------------------ #
    #  通用 ReAct 拓扑工厂（简化子类实现）                                  #
    # ------------------------------------------------------------------ #

    def _build_react_graph(
        self,
        *,
        state_schema: type,
        model_node_fn,
        tools: list,
        max_tool_loops: int,
        loop_exceeded_message: str,
    ) -> Runnable:
        """
        通用 ReAct 子图工厂：START → agent → (tools_condition) → tools → agent → END。

        消除各 Agent 中重复的 StateGraph 样板代码，统一拓扑结构，
        各 Agent 只需提供差异化的 model_node_fn 和 tools 列表。

        Args:
            state_schema:          子图 State TypedDict 类型。
            model_node_fn:         Agent 节点函数 (state) -> dict，包含业务逻辑。
            tools:                 绑定到 ToolNode 的工具列表。
            max_tool_loops:        最大工具调用循环次数（超过则强制收尾）。
            loop_exceeded_message: 循环次数超限时返回的提示文本。

        Returns:
            编译好的 CompiledGraph，绑定了当前 Agent 的 checkpointer。
        """
        from langgraph.prebuilt import ToolNode, tools_condition

        workflow = StateGraph(state_schema)

        def _guarded_model_node(state):
            """在业务 model_node 外包一层循环上限保护。"""
            loop_count = int(state.get("tool_loop_count", 0) or 0)
            if loop_count >= max_tool_loops:
                from langchain_core.messages import AIMessage
                return {
                    "tool_loop_count": loop_count,
                    "messages": [AIMessage(content=loop_exceeded_message)],
                }
            return model_node_fn(state)

        def _route_after_agent(state):
            """条件边：循环超限直接 END，否则走 tools_condition 判断。"""
            if int(state.get("tool_loop_count", 0) or 0) > max_tool_loops:
                from langgraph.graph import END
                return END
            return tools_condition(state)

        tool_node = ToolNode(tools)

        workflow.add_node("agent", _guarded_model_node)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", _route_after_agent)
        workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=self.checkpointer)

    def run(self, req: AgentRequest, config: Optional[RunnableConfig] = None) -> Generator[Dict[str, Any], None, None]:
        """
        统一的图执行入口。

        接收外部传入的 config 并进行安全拷贝与合并，以保证 LangSmith Trace 的上下文连续性。
        同时为 LangGraph 配置持久化 ID，确保不同的业务子图在相同 Session 下状态有效隔离。

        Args:
            req (AgentRequest): 包含对话上下文的请求对象。
            config (Optional[RunnableConfig]): 由上游传递的 LangGraph 配置项。

        Yields:
            Generator[Dict[str, Any], None, None]: 流程图中产生的状态更新事件流或错误信息。
        """
        # 1. 继承父级 config（包含 LangSmith 的 tags, callbacks 等）
        base_config = config or {}
        # 安全复制，避免修改上游的 config 指针产生副作用
        configurable = base_config.get("configurable", {}).copy()

        # 2. 注入当前子 Agent 的持久化标识
        # 采用联合 thread_id 以在同一对话会话中隔离不同子图的状态，避免直接使用 checkpoint_ns 报错
        configurable["thread_id"] = f"{self.session_id}_{self.subgraph_id}"

        final_config = {**base_config, "configurable": configurable}

        # 3. 构造初始状态
        # 为每次 Agent 调用注入严格日期上下文，并尽量携带最近多轮上下文。
        history_messages = self._extract_history_messages(req)
        # 读取会话摘要（如城市/用户画像），减少反复追问
        context_summary = self._extract_context_summary(req)
        input_messages: list[BaseMessage] = [
            SystemMessage(content=get_agent_date_context()),
            SystemMessage(content=get_current_time_context()),
        ]
        if context_summary:
            input_messages.append(SystemMessage(content=context_summary))
        input_messages.extend(history_messages)
        latest_human_text = ""
        for msg in reversed(history_messages):
            if isinstance(msg, HumanMessage):
                latest_human_text = self._message_text(msg)
                break
        current_text = (req.user_input or "").strip()
        if not latest_human_text or latest_human_text != current_text:
            input_messages.append(HumanMessage(content=req.user_input))

        input_message = {"messages": input_messages}

        try:
            # 同时监听 updates/messages；updates 供状态推进，messages 用于子 Agent 实时出字。
            stream_channel_id = str(configurable.get("run_id") or "").strip()
            stream_mode = ["updates", "messages"] if AGENT_LIVE_STREAM_ENABLED else ["updates"]
            for raw_event in self.graph.stream(
                input_message,
                config=final_config,
                stream_mode=stream_mode,
            ):
                event_type = "updates"
                event_payload = raw_event
                if isinstance(raw_event, tuple) and len(raw_event) == 2:
                    event_type, event_payload = raw_event

                if event_type == "updates":
                    if isinstance(event_payload, dict):
                        yield event_payload
                    continue

                if event_type == "messages":
                    if not AGENT_LIVE_STREAM_ENABLED:
                        continue
                    if not (isinstance(event_payload, tuple) and len(event_payload) == 2):
                        continue
                    msg_chunk, _metadata = event_payload
                    if not isinstance(msg_chunk, AIMessageChunk):
                        continue
                    if not stream_channel_id:
                        continue
                    if getattr(msg_chunk, "tool_calls", None):
                        continue
                    chunk_text = self._message_text(msg_chunk)
                    if chunk_text:
                        agent_stream_bus.publish(
                            run_id=stream_channel_id,
                            agent_name=self.subgraph_id,
                            content=chunk_text,
                        )
        except Exception as e:
            err_msg = str(e)
            if "Interrupt(" in err_msg or e.__class__.__name__ == "GraphInterrupt":
                log.info(f"Agent {self.subgraph_id} 触发原生中断挂起，等待审批恢复。")
                payload = self._extract_interrupt_payload(e)
                if payload:
                    yield {"interrupt": payload}
                else:
                    yield {
                        "interrupt": {
                            "message": DEFAULT_INTERRUPT_MESSAGE,
                            "allowed_decisions": list(DEFAULT_ALLOWED_DECISIONS),
                            "action_requests": []
                        }
                    }
                return
            log.error(f"Agent {self.subgraph_id} 运行出错: {err_msg}")
            yield {"error": err_msg}

    def get_state(self) -> Any:
        """
        查询当前子图运行状态。

        提供给外部系统查询智能体当前的图状态（例如是否处于 Interrupt 中断挂起等待审核等）。

        Returns:
            Any: 当前图状态对象。
        """
        config = {
            "configurable": {
                "thread_id": f"{self.session_id}_{self.subgraph_id}"
            }
        }
        return self.graph.get_state(config)
