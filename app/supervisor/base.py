"""
所有具体业务智能体（如 YunyouAgent）的抽象基类。

本模块提供了核心的 `BaseAgent` 类，负责：
1. 统一处理来自用户的 AgentRequest 请求封装。
2. 绑定和管理由于多轮对话产生的 Checkpointer 状态持久化。
3. 统一规范流式输出与图执行的生命周期隔离机制。
4. 提供通用的 ReAct 模式子图构建方法，消除重复代码。
5. 上下文管理和历史消息压缩。

设计理念：
- 抽象基类：定义所有业务 Agent 的通用接口和行为
- 统一流程：标准化图执行、状态管理、错误处理
- 代码复用：提供通用 ReAct 子图工厂，避免每个 Agent 重复构建拓扑
- 上下文优化：智能压缩历史消息，提升性能
- 状态隔离：不同子图使用独立的 thread_id，避免状态冲突

使用方式：
- 继承 BaseAgent 实现具体业务 Agent
- 实现 _build_graph() 方法构建子图
- 使用 _build_react_graph() 快速构建标准 ReAct 子图
- 重写 run() 方法自定义执行逻辑（可选）

典型子类：
- SearchAgent：搜索 Agent
- SqlAgent：SQL 查询 Agent
- CodeAgent：代码执行 Agent
- MedicalAgent：医疗问答 Agent
- YunyouAgent：云柚业务 Agent
"""

from abc import ABC, abstractmethod
import functools
import re
from typing import Generator, Any, Dict, Optional
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, START, StateGraph
from supervisor.graph_state import AgentRequest
from supervisor.checkpointer import get_checkpointer
from config.constants.approval_constants import DEFAULT_ALLOWED_DECISIONS, DEFAULT_INTERRUPT_MESSAGE
from config.runtime_settings import AGENT_LOOP_CONFIG, AGENT_LIVE_STREAM_ENABLED
from services.agent_stream_bus import agent_stream_bus
from common.utils.history_compressor import compress_history_messages
from common.utils.custom_logger import get_logger
from common.utils.date_utils import get_agent_date_context, get_current_time_context

log = get_logger(__name__)


def _unwrap_interrupt_candidate(obj: Any) -> Optional[dict]:
    """
    从复杂对象中提取中断载荷（interrupt payload）

    设计目的：
    - LangGraph 的中断异常可能嵌套在多层结构中
    - 兼容不同版本的异常结构（args / interrupts / value）
    - 提取包含审批信息的 dict（action_requests、allowed_decisions、message）

    检测逻辑：
    1. 如果 obj 是 dict，检查是否包含关键字段
   2. 如果 obj 是 list 或 tuple，递归查找
   3. 如果 obj 有 value 属性，递归查找

    关键字段：
    - action_requests：审批动作列表
    - allowed_decisions：允许的决策选项
    - message：审批提示消息

    参数说明：
    - obj: 可能包含中断载荷的对象

    返回值：
    - Optional[dict]: 提取的中断载荷，如果未找到返回 None

    示例：
    >>> obj = {"action_requests": [...], "allowed_decisions": [...], "message": "..."}
    >>> _unwrap_interrupt_candidate(obj)
    {"action_requests": [...], "allowed_decisions": [...], "message": "..."}
    """
    if obj is None:
        return None

    # 检查是否是包含关键字的 dict
    if isinstance(obj, dict):
        if "action_requests" in obj or "allowed_decisions" in obj or "message" in obj:
            return obj
        return None

    # 递归查找 list 或 tuple 中的候选对象
    if isinstance(obj, (list, tuple)):
        for item in obj:
            payload = _unwrap_interrupt_candidate(item)
            if payload:
                return payload
        return None

    # 检查是否有 value 属性（兼容某些版本的异常结构）
    if hasattr(obj, "value"):
        return _unwrap_interrupt_candidate(getattr(obj, "value"))

    return None


def _guarded_model_node(
    state,
    *,
    model_node_fn,
    max_tool_loops: int,
    loop_exceeded_message: str,
):
    """
    在业务 model_node 外包一层循环上限保护

    设计目的：
    - 防止工具调用无限循环
    - 超过循环上限时强制终止并返回友好提示
    - 保护资源和用户体验

    工作流程：
    1. 获取当前工具调用次数（tool_loop_count）
    2. 检查是否超过上限
    3. 如果超过，返回终止消息
    4. 如果未超过，调用实际的 model_node_fn

    参数说明：
    - state: 子图状态
    - model_node_fn: 实际的模型节点函数
    - max_tool_loops: 最大工具循环次数
    - loop_exceeded_message: 超限时的提示消息

    返回值：
    - dict: 更新后的状态

    示例：
    >>> def my_model_node(state):
    ...     # 业务逻辑
    ...     return {"tool_loop_count": state["tool_loop_count"] + 1}
    >>> _guarded_model_node(
    ...     {"tool_loop_count": 5},
    ...     model_node_fn=my_model_node,
    ...     max_tool_loops=5,
    ...     loop_exceeded_message="已达到最大循环次数"
    ... )
    {"tool_loop_count": 5, "messages": [AIMessage(content="已达到最大循环次数")]}
    """
    loop_count = int(state.get("tool_loop_count", 0) or 0)
    if loop_count >= max_tool_loops:
        from langchain_core.messages import AIMessage
        return {
            "tool_loop_count": loop_count,
            "messages": [AIMessage(content=loop_exceeded_message)],
        }
    return model_node_fn(state)


def _route_after_agent(state, *, max_tool_loops: int, tools_condition_fn):
    """
    条件边：循环超限直接 END，否则走 tools_condition 判断

    设计目的：
    - 实现 ReAct 循环的条件路由
    - 超过循环上限时强制结束
    - 正常情况下根据是否有工具调用决定路由

    路由规则：
    1. 如果 tool_loop_count > max_tool_loops → END（强制结束）
    2. 否则，调用 tools_condition_fn 判断：
       - 有工具调用 → tools 节点
       - 无工具调用 → END

    参数说明：
    - state: 子图状态
    - max_tool_loops: 最大工具循环次数
    - tools_condition_fn: 工具条件判断函数

    返回值：
    - str: 下一个节点名称或 "END"

    示例：
    >>> state = {"tool_loop_count": 5}
    >>> _route_after_agent(state, max_tool_loops=5, tools_condition_fn=lambda x: "tools")
    "END"
    >>> state = {"tool_loop_count": 3}
    >>> _route_after_agent(state, max_tool_loops=5, tools_condition_fn=lambda x: "tools")
    "tools"
    """
    if int(state.get("tool_loop_count", 0) or 0) > max_tool_loops:
        return END
    return tools_condition_fn(state)


class BaseAgent(ABC):
    """
    Agent 基类：所有具体业务 Agent 的抽象父类

    核心职责：
    - 统一处理 AgentRequest 请求
    - 管理 checkpointer 状态持久化
    - 提供通用的 ReAct 子图构建方法
    - 上下文管理和历史消息压缩
    - 流式输出和错误处理

    主要方法：
    - _message_text: 提取消息文本内容
    - _extract_text_tokens: 提取关键词用于相关性筛选
    - _extract_history_messages: 提取并压缩历史消息
    - _extract_context_summary: 提取会话摘要
    - _extract_interrupt_payload: 提取中断载荷
    - _build_graph: 抽象方法，子类必须实现
    - _build_react_graph: 通用 ReAct 子图工厂
    - run: 统一的图执行入口
    - get_state: 查询当前子图状态

    设计模式：
    - 模板方法模式：定义算法骨架，子类实现具体步骤
    - 工厂方法模式：提供 _build_react_graph 工厂方法
    - 策略模式：不同的子类有不同的执行策略

    使用示例：
    >>> class MyAgent(BaseAgent):
    ...     def __init__(self, req: AgentRequest):
    ...         super().__init__(req)
    ...         self.graph = self._build_graph()
    ...
    ...     def _build_graph(self) -> Runnable:
    ...         return self._build_react_graph(
    ...             state_schema=MyState,
    ...             model_node_fn=self._model_node,
    ...             tools=[my_tool],
    ...             max_tool_loops=5,
    ...             loop_exceeded_message="超过最大循环次数"
    ...         )
    ...
    ...     def _model_node(self, state: MyState):
    ...         # 业务逻辑
    ...         return {"tool_loop_count": state["tool_loop_count"] + 1}
    """

    @staticmethod
    def _message_text(msg: BaseMessage) -> str:
        """
        提取消息文本内容

        设计目的：
        - 统一处理不同类型的消息内容（str、list、dict）
        - 提取纯文本，便于后续处理
        - 避免复杂结构导致的解析问题

        处理逻辑：
        1. 如果 content 是字符串，直接返回
        2. 如果 content 是列表，提取所有文本部分
        3. 如果 content 是字典，提取 text 或 content 字段
        4. 其他情况转为字符串

        参数说明：
        - msg: 消息对象

        返回值：
        - str: 提取的文本内容

        示例：
        >>> msg = AIMessage(content="这是一条消息")
        >>> BaseAgent._message_text(msg)
        "这是一条消息"
        >>> msg = AIMessage(content=[{"type": "text", "text": "这是一条消息"}])
        >>> BaseAgent._message_text(msg)
        "这是一条消息"
        """
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
        """
        提取轻量关键词用于相关性筛选

        设计目的：
        - 从文本中提取关键词，用于判断历史消息的相关性
        - 减少不相关历史消息的干扰
        - 提升上下文质量和性能

        提取规则：
        - 中文词组：连续 min_chars 个及以上中文字符
        - 英文词组：连续 min_chars 个及以上字母、数字、下划线
        - 英文词组转换为小写

        参数说明：
        - text: 待提取文本

        返回值：
        - set[str]: 关键词集合

        示例：
        >>> BaseAgent._extract_text_tokens("查询Holter数据")
        {"查询", "Holter数据"}
        >>> BaseAgent._extract_text_tokens("search python code")
        {"search", "python", "code"}
        """
        if not text:
            return set()

        min_chars = AGENT_LOOP_CONFIG.context_relevance_min_token_chars

        # 提取中文词组
        zh_tokens = re.findall(rf"[\u4e00-\u9fa5]{{{min_chars},}}", text)

        # 提取英文词组（小写）
        en_tokens = re.findall(rf"[A-Za-z0-9_]{{{min_chars},}}", text.lower())

        return set(zh_tokens + en_tokens)

    @classmethod
    def _extract_history_messages(cls, req: AgentRequest) -> list[BaseMessage]:
        """
        提取并压缩最近多轮对话消息

        设计目的：
        - 从请求中提取历史消息
        - 基于相关性筛选历史消息，减少干扰
        - 压缩消息以控制 token 使用量
        - 提升上下文质量和性能

        提取流程：
        1. 从 req.state 中获取消息列表
        2. 截取最近的 N 条消息（context_history_messages）
        3. 相关性筛选：
           - 提取当前问题的关键词
           - 保留尾部窗口（context_relevance_tail_messages）
           - 在头部窗口中筛选与当前问题相关的消息
        4. 兜底机制：至少保留最后一条用户消息
        5. 压缩消息（token/字符限制）

        相关性筛选逻辑：
        - 尾部窗口：强制保留最近用户消息，AI 消息需与当前问题相关
        - 头部窗口：只保留与当前问题有关键词交集的消息
        - 避免旧主题污染当前对话

        参数说明：
        - req: Agent 请求对象

        返回值：
        - list[BaseMessage]: 提取并压缩后的消息列表

        示例：
        >>> req = AgentRequest(
        ...     state={"messages": [HumanMessage("查询天气"), AIMessage("今天晴天"), HumanMessage("明天呢")]},
        ...     user_input="明天呢"
        ... )
        >>> BaseAgent._extract_history_messages(req)
        [HumanMessage("明天呢")]  # 简化示例
        """
        state = getattr(req, "state", None) or {}
        if not isinstance(state, dict):
            return []

        messages = state.get("messages")
        if not isinstance(messages, list):
            return []

        # 筛选出有效的消息对象
        valid_messages = [m for m in messages if isinstance(m, BaseMessage)]

        # 截取最近的消息
        recent_messages = valid_messages[-AGENT_LOOP_CONFIG.context_history_messages:]
        if not recent_messages:
            return []

        # 相关性筛选：保留尾部窗口 + 与当前问题词面有交集的历史
        current_tokens = cls._extract_text_tokens((req.user_input or "").strip())
        tail_window = max(1, min(AGENT_LOOP_CONFIG.context_relevance_tail_messages, AGENT_LOOP_CONFIG.context_history_messages))
        tail_part = recent_messages[-tail_window:]
        head_part = recent_messages[:-tail_window]
        filtered_messages: list[BaseMessage] = []

        # 头部窗口：只保留与当前问题相关的消息
        if current_tokens:
            for msg in head_part:
                msg_tokens = cls._extract_text_tokens(cls._message_text(msg))
                if msg_tokens & current_tokens:
                    filtered_messages.append(msg)

        # 尾部窗口：强保留最近用户消息，AI 消息需与当前问题相关
        for msg in tail_part:
            if isinstance(msg, HumanMessage):
                filtered_messages.append(msg)
                continue
            msg_tokens = cls._extract_text_tokens(cls._message_text(msg))
            if (not current_tokens) or (msg_tokens & current_tokens):
                filtered_messages.append(msg)

        # 兜底：至少保留最后一条用户消息，避免丢上下文锚点
        if not any(isinstance(m, HumanMessage) for m in filtered_messages):
            for msg in reversed(recent_messages):
                if isinstance(msg, HumanMessage):
                    filtered_messages.append(msg)
                    break

        # 压缩消息以控制 token 使用量
        return compress_history_messages(
            filtered_messages,
            model=getattr(req, "model", None),
            max_tokens=AGENT_LOOP_CONFIG.context_compress_max_tokens,
            max_chars=AGENT_LOOP_CONFIG.context_compress_max_chars,
        )

    @staticmethod
    def _extract_context_summary(req: AgentRequest) -> str:
        """
        从请求 state 中读取会话摘要，供子 Agent 继承全局上下文

        设计目的：
        - 读取会话摘要（如城市、用户画像等）
        - 将摘要传递给子 Agent，减少反复追问
        - 保持上下文连续性

        摘要来源：
        - req.state.context_summary 字段
        - 由上层（如 Supervisor）维护的全局会话摘要

        参数说明：
        - req: Agent 请求对象

        返回值：
        - str: 会话摘要文本，如果不存在返回空字符串

        示例：
        >>> req = AgentRequest(state={"context_summary": "用户在北京，关注房价"})
        >>> BaseAgent._extract_context_summary(req)
        "用户在北京，关注房价"
        """
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
        尽最大可能从 LangGraph GraphInterrupt 异常对象中提取审批载荷

        设计目的：
        - 从中断异常中提取审批信息
        - 兼容不同版本的异常结构（args / interrupts / value）
        - 提取 action_requests、allowed_decisions、message 等字段

        异常结构：
        - 不同版本的 LangGraph 可能有不同的异常结构
        - 载荷可能在 exc.args、exc.interrupts、exc.value 等位置
        - 需要递归查找嵌套的 payload 对象

        参数说明：
        - exc: LangGraph GraphInterrupt 异常对象

        返回值：
        - Optional[dict]: 提取的中断载荷，如果未找到返回 None

        示例：
        >>> try:
        ...     interrupt({"action_requests": [...], "message": "..."})
        ... except Exception as e:
        ...     payload = BaseAgent._extract_interrupt_payload(e)
        ...     print(payload)
        {"action_requests": [...], "message": "..."}
        """
        candidates: list[Any] = []

        # 检查 interrupts 属性
        if hasattr(exc, "interrupts"):
            interrupts = getattr(exc, "interrupts")
            if interrupts:
                candidates.extend(list(interrupts))

        # 检查 value 属性
        if hasattr(exc, "value"):
            candidates.append(getattr(exc, "value"))

        # 检查 args 属性
        if getattr(exc, "args", None):
            candidates.extend(list(exc.args))

        # 遍历候选对象，提取 payload
        for c in candidates:
            payload = _unwrap_interrupt_candidate(c)
            if payload:
                return payload

        return None

    def __init__(self, req: AgentRequest):
        """
        初始化 Agent 实例

        设计目的：
        - 初始化 Agent 的基本属性
        - 配置状态持久化（checkpointer）
        - 设置子图标识符

        初始化流程：
        1. 保存 AgentRequest 对象
        2. 提取 session_id
        3. 设置 subgraph_id（用于状态隔离）
        4. 创建 checkpointer（状态持久化）

        参数说明：
        - req: Agent 请求对象，包含用户输入、会话信息、模型配置等

        属性说明：
        - req: AgentRequest 对象
        - session_id: 会话 ID
        - subgraph_id: 子图标识符，用于状态隔离
        - checkpointer: LangGraph checkpointer，用于状态持久化

        示例：
        >>> req = AgentRequest(session_id="xxx", model=llm)
        >>> agent = BaseAgent(req)
        >>> agent.subgraph_id
        "BaseAgent"  # 默认使用类名
        """
        self.req = req
        self.session_id = req.session_id

        # 子图标识，用于状态隔离
        # 默认使用类名，也可在 req 中指定
        self.subgraph_id = getattr(req, "subgraph_id", None) or self.__class__.__name__

        # 创建 checkpointer，用于状态持久化
        self.checkpointer = get_checkpointer(self.subgraph_id)

    @abstractmethod
    def _build_graph(self) -> Runnable:
        """
        子类必须实现此方法，返回编译后的 StateGraph

        设计目的：
        - 定义子图的拓扑结构
        - 子类根据业务需求构建不同的图
        - 使用 LangGraph 的 StateGraph 构建流程

        实现要求：
        - 必须返回编译后的 Runnable
        - 应该使用 self.checkpointer
        - 可以使用 _build_react_graph() 快速构建标准 ReAct 子图

        返回值：
        - Runnable: 编译后的 LangGraph 子图

        示例：
        >>> def _build_graph(self) -> Runnable:
        ...     workflow = StateGraph(MyState)
        ...     workflow.add_node("agent", self._model_node)
        ...     workflow.add_node("tools", self.tool_node)
        ...     workflow.add_edge(START, "agent")
        ...     workflow.add_conditional_edges("agent", self._should_continue)
        ...     workflow.add_edge("tools", "agent")
        ...     return workflow.compile(checkpointer=self.checkpointer)
        """
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
        通用 ReAct 子图工厂：START → agent → (tools_condition) → tools → agent → END

        设计目的：
        - 消除各 Agent 中重复的 StateGraph 样板代码
        - 统一拓扑结构（ReAct 模式）
        - 各 Agent 只需提供差异化的 model_node_fn 和 tools 列表
        - 提升代码复用性和可维护性

        ReAct 模式：
        - Reasoning（推理）：agent 节点分析问题并决定下一步行动
        - Acting（行动）：tools 节点执行工具调用
        - Observation（观察）：将工具结果反馈给 agent
        - 循环直到完成或超限

        子图结构：
        1. START → agent：初始调用 agent 节点
        2. agent → (条件判断)：
           - 有工具调用 → tools
           - 无工具调用 → END
        3. tools → agent：工具结果返回给 agent
        4. 循环直到完成或超限

        参数说明：
            - state_schema: 子图 State TypedDict 类型
            - model_node_fn: Agent 节点函数 (state) -> dict，包含业务逻辑
            - tools: 绑定到 ToolNode 的工具列表
            - max_tool_loops: 最大工具调用循环次数（超过则强制收尾）
            - loop_exceeded_message: 循环次数超限时返回的提示文本

        返回值：
            - CompiledGraph: 编译好的图，绑定了当前 Agent 的 checkpointer

        使用示例：
        >>> def _build_graph(self) -> Runnable:
        ...     return self._build_react_graph(
        ...         state_schema=MyState,
        ...         model_node_fn=self._model_node,
        ...         tools=[search_tool, weather_tool],
        ...         max_tool_loops=5,
        ...         loop_exceeded_message="已达到最大循环次数"
        ...     )
        """
        from langgraph.prebuilt import ToolNode, tools_condition

        workflow = StateGraph(state_schema)

        # 创建工具节点
        tool_node = ToolNode(tools)

        # 添加 agent 节点（带循环保护）
        workflow.add_node(
            "agent",
            functools.partial(
                _guarded_model_node,
                model_node_fn=model_node_fn,
                max_tool_loops=max_tool_loops,
                loop_exceeded_message=loop_exceeded_message,
            ),
        )

        # 添加 tools 节点
        workflow.add_node("tools", tool_node)

        # 添加边
        workflow.add_edge(START, "agent")

        # 添加条件边：agent → (有工具调用？tools : END)
        workflow.add_conditional_edges(
            "agent",
            functools.partial(
                _route_after_agent,
                max_tool_loops=max_tool_loops,
                tools_condition_fn=tools_condition,
            ),
        )

        # 添加边：tools → agent（循环）
        workflow.add_edge("tools", "agent")

        # 编译图并绑定 checkpointer
        return workflow.compile(checkpointer=self.checkpointer)

    def run(self, req: AgentRequest, config: Optional[RunnableConfig] = None) -> Generator[Dict[str, Any], None, None]:
        """
        统一的图执行入口

        设计目的：
        - 接收外部传入的 config 并进行安全拷贝与合并
        - 保证 LangSmith Trace 的上下文连续性
        - 为 LangGraph 配置持久化 ID
        - 确保不同的业务子图在相同 Session 下状态有效隔离

        执行流程：
        1. 继承父级 config（包含 LangSmith 的 tags、callbacks 等）
        2. 安全复制 config，避免修改上游的 config 指针产生副作用
        3. 注入当前子 Agent 的持久化标识（thread_id）
        4. 构造初始状态：
           - 注入日期和时间上下文
           - 读取会话摘要
           - 提取并压缩历史消息
           - 添加当前用户输入
        5. 执行图并流式输出：
           - 监听 updates 事件（状态更新）
           - 监听 messages 事件（实时输出，可选）
        6. 处理中断异常（审批流程）
        7. 处理其他异常

        状态隔离：
        - 使用联合 thread_id：{session_id}_{subgraph_id}
        - 确保不同子图的状态不冲突
        - 支持在同一会话中运行多个子图

        实时流式输出：
        - 当 AGENT_LIVE_STREAM_ENABLED=True 时，监听 messages 事件
        - 将 AIMessageChunk 内容实时推送到 agent_stream_bus
        - 前端可以通过订阅 run_id 获取实时输出

        参数说明：
            - req: 包含对话上下文的请求对象
            - config: 由上游传递的 LangGraph 配置项（可选）

        Yields:
            - Generator[Dict[str, Any], None, None]: 流程图中产生的状态更新事件流或错误信息
            - {"updates": {...}}: 状态更新
            - {"messages": [...]}: 消息更新
            - {"interrupt": {...}}: 中断载荷（审批请求）
            - {"error": "..."}: 错误信息

        异常处理：
        - GraphInterrupt（审批中断）：提取载荷并返回 interrupt 事件
        - 其他异常：返回 error 事件

        示例：
        >>> req = AgentRequest(
        ...     session_id="xxx",
        ...     user_input="查询天气",
        ...     model=llm
        ... )
        >>> for event in agent.run(req):
        ...     print(event)
        {"updates": {"tool_loop_count": 1}}
        {"messages": [AIMessage(content="...")]}
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
        # 为每次 Agent 调用注入严格日期上下文，并尽量携带最近多轮上下文
        history_messages = self._extract_history_messages(req)

        # 读取会话摘要（如城市/用户画像），减少反复追问
        context_summary = self._extract_context_summary(req)

        input_messages: list[BaseMessage] = [
            SystemMessage(content=get_agent_date_context()),
            SystemMessage(content=get_current_time_context()),
        ]

        # 添加会话摘要（如果有）
        if context_summary:
            input_messages.append(SystemMessage(content=context_summary))

        # 添加历史消息
        input_messages.extend(history_messages)

        # 检查最新用户消息，避免重复添加
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

                # 解析事件类型和载荷
                if isinstance(raw_event, tuple) and len(raw_event) == 2:
                    event_type, event_payload = raw_event

                # 处理 updates 事件（状态更新）
                if event_type == "updates":
                    if isinstance(event_payload, dict):
                        yield event_payload
                    continue

                # 处理 messages 事件（实时输出）
                if event_type == "messages":
                    # 如果未启用实时流式输出，跳过
                    if not AGENT_LIVE_STREAM_ENABLED:
                        continue
                    # 检查事件格式
                    if not (isinstance(event_payload, tuple) and len(event_payload) == 2):
                        continue

                    msg_chunk, _metadata = event_payload

                    # 只处理 AIMessageChunk
                    if not isinstance(msg_chunk, AIMessageChunk):
                        continue

                    # 需要有效的 stream_channel_id
                    if not stream_channel_id:
                        continue

                    # 跳过工具调用
                    if getattr(msg_chunk, "tool_calls", None):
                        continue

                    # 提取文本并推送到流式总线
                    chunk_text = self._message_text(msg_chunk)
                    if chunk_text:
                        agent_stream_bus.publish(
                            run_id=stream_channel_id,
                            agent_name=self.subgraph_id,
                            content=chunk_text,
                        )

        except Exception as e:
            err_msg = str(e)

            # 处理中断异常（审批流程）
            if "Interrupt(" in err_msg or e.__class__.__name__ == "GraphInterrupt":
                log.info(f"Agent {self.subgraph_id} 触发原生中断挂起，等待审批恢复。")

                # 提取中断载荷
                payload = self._extract_interrupt_payload(e)
                if payload:
                    yield {"interrupt": payload}
                else:
                    # 如果无法提取载荷，使用默认载荷
                    yield {
                        "interrupt": {
                            "message": DEFAULT_INTERRUPT_MESSAGE,
                            "allowed_decisions": list(DEFAULT_ALLOWED_DECISIONS),
                            "action_requests": []
                        }
                    }
                return

            # 处理其他异常
            log.error(f"Agent {self.subgraph_id} 运行出错: {err_msg}")
            yield {"error": err_msg}

    def get_state(self) -> Any:
        """
        查询当前子图运行状态

        设计目的：
        - 提供给外部系统查询智能体当前的图状态
        - 用于判断是否处于中断挂起等待审核等状态
        - 支持状态监控和调试

        使用场景：
        - 前端查询 Agent 状态（是否正在执行、是否等待审批）
        - 调试和监控 Agent 执行过程
        - 获取当前图状态快照

        返回值：
            - Any: 当前图状态对象（StateSnapshot）

        状态信息：
        - values: 当前状态值
        - next: 下一个节点
        - config: 配置信息
        - metadata: 元数据

        示例：
        >>> state = agent.get_state()
        >>> print(state.values)
        {"messages": [...], "tool_loop_count": 1}
        >>> print(state.next)
        ["tools"]
        """
        config = {
            "configurable": {
                "thread_id": f"{self.session_id}_{self.subgraph_id}"
            }
        }
        return self.graph.get_state(config)
