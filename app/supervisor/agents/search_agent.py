"""
搜索 Agent：处理互联网搜索查询的智能体

核心职责：
- 使用 Tavily 搜索工具进行互联网信息检索
- 处理用户的搜索需求，返回相关网络内容
- 防止搜索主题跑偏（如房产查询误答天气信息）
- 控制工具调用循环次数，避免无限循环
- 检测搜索失败情况，及时终止并返回友好提示

业务场景示例：
1. 用户问"北京今天天气" → 应该由 weather_agent 处理，但如果是"北京今天天气如何，影响房价吗"，则本 agent 会搜索相关信息
2. 用户问"上海最近的房产政策" → 本 agent 会搜索最新房产政策信息
3. 用户问"人工智能最新进展" → 本 agent 会搜索 AI 相关最新资讯
4. 用户问一个复杂问题需要联网 → 本 agent 使用 tavily_search_tool 进行搜索

设计要点：
- 使用 LangGraph 构建基于消息传递的子图
- 集成 ReAct（推理-行动）模式：先推理需要什么信息，再执行搜索工具
- 防护机制：检测主题跑偏（房产问题误答天气）并自动纠偏
- 失败检测：检测搜索超时、失败等异常，避免陷入死循环
- 循环控制：通过 tool_loop_count 限制工具调用次数
- 复用 BaseAgent 的 _build_react_graph 方法，避免重复代码

与其他 Agent 的区别：
- weather_agent：专门处理天气查询，使用天气 API
- search_agent：处理通用互联网搜索，使用 Tavily 搜索引擎
- code_agent：处理代码相关问题
- sql_agent：处理 SQL 查询和数据库操作

输出结果：
- AI 消息：包含搜索结果的总结性回答
- 支持多轮搜索：根据第一次搜索结果判断是否需要进一步搜索
"""

from typing import Annotated, List, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langgraph.graph import add_messages

from supervisor.base import BaseAgent
from supervisor.graph_state import AgentRequest
from config.runtime_settings import AGENT_LOOP_CONFIG
from config.constants.agent_messages import (
    SEARCH_TOOL_FAILURE_MESSAGE,
    SEARCH_TOOL_LOOP_EXCEEDED_MESSAGE,
    SEARCH_TOPIC_DRIFT_BLOCK_MESSAGE,
)
from config.constants.search_keywords import SEARCH_REAL_ESTATE_KEYWORDS, SEARCH_WEATHER_KEYWORDS
from tools.agent_tools.search_tools import tavily_search_tool
from common.utils.custom_logger import get_logger
from prompts.agent_prompts.tools_prompt import ToolsPrompt

log = get_logger(__name__)


class SearchAgentState(TypedDict):
    """搜索子图状态

    状态字段说明：
    - messages: 消息列表，包括用户消息、AI 消息、工具消息
    - tool_loop_count: 工具调用次数计数器，用于防止无限循环

    使用场景：
    - 在子图执行过程中维护状态
    - 跨节点传递消息和控制信息
    - 追踪工具调用次数，实现循环控制
    """
    messages: Annotated[List[BaseMessage], add_messages]
    tool_loop_count: int  # 工具调用次数计数器


class SearchAgent(BaseAgent):
    """搜索 Agent：处理互联网搜索查询

    主要功能：
    - 使用 Tavily 搜索工具进行网络信息检索
    - 根据用户问题智能构建搜索查询
    - 处理搜索结果并生成友好回复
    - 防止搜索主题跑偏，保持回答聚焦
    - 控制搜索循环次数，避免资源浪费

    工作流程：
    1. 接收用户查询
    2. 分析查询内容，判断是否需要搜索
    3. 构建搜索查询，调用 Tavily 搜索工具
    4. 处理搜索结果
    5. 检测主题是否跑偏，如有则纠偏重试
    6. 生成最终回复

    典型使用场景：
    - "北京最近的房价趋势"
    - "人工智能领域的最新突破"
    - "2024年中国经济政策变化"
    - "深圳最近的房产调控政策"

    不适合场景：
    - 实时天气查询（应使用 weather_agent）
    - 代码相关问题（应使用 code_agent）
    - 数据库查询（应使用 sql_agent）
    """

    def __init__(self, req: AgentRequest):
        """
        初始化搜索 Agent

        参数说明：
        - req: Agent 请求对象，包含模型配置、会话信息等

        初始化步骤：
        1. 调用父类 BaseAgent 初始化
        2. 验证模型配置，确保搜索模型可用
        3. 创建 LLM 实例，绑定搜索工具
        4. 构建主提示词和防护提示词
        5. 构建搜索子图

        异常处理：
        - 如果模型未配置，抛出 ValueError 提示检查配置

        示例：
        >>> req = AgentRequest(model=llm, session_id="xxx")
        >>> search_agent = SearchAgent(req)
        >>> result = search_agent.invoke({"messages": [HumanMessage("搜索深圳房价")]})
        """
        super().__init__(req)
        if not req.model:
            raise ValueError("搜索模型初始化失败，请检查配置。")
        self.llm = req.model
        self.subgraph_id = "search_agent"

        # 工具：绑定 Tavily 搜索工具
        # Tavily 是一个专为大语言模型优化的搜索引擎 API
        self.tools = [tavily_search_tool]
        self.model_with_tools = self.llm.bind_tools(self.tools)

        # 提示词：主提示词用于正常搜索流程
        # SEARCH_SYSTEM 包含搜索 agent 的角色定义和指令
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ToolsPrompt.SEARCH_SYSTEM),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # 防护提示词：用于主题跑偏后的纠偏重试
        # SEARCH_TOPIC_GUARD 包含纠偏指令，要求回答聚焦于原问题
        self.guard_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ToolsPrompt.SEARCH_SYSTEM),
                ("system", ToolsPrompt.SEARCH_TOPIC_GUARD),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        self.graph = self._build_graph()

    @staticmethod
    def _latest_human_text(messages: List[BaseMessage]) -> str:
        """
        提取最新的人类消息文本

        设计目的：
        - 在多轮对话中找到用户的最新提问
        - 用于主题漂移检测时获取用户原始问题

        实现逻辑：
        - 从消息列表末尾反向遍历
        - 找到第一个 HumanMessage 类型消息
        - 提取其文本内容并去除首尾空白

        使用场景：
        - 检测回答是否跑偏时，需要获取用户的原始问题
        - 纠偏重试时，需要将用户原始问题传递给模型

        参数说明：
        - messages: 消息列表，可能包含 HumanMessage、AIMessage、ToolMessage

        返回值：
        - str: 最新的人类消息文本，如果没有人类消息返回空字符串

        示例：
        >>> messages = [
        ...     HumanMessage("深圳房价如何？"),
        ...     AIMessage("我来帮你查询深圳房价信息。"),
        ...     ToolMessage("深圳房价均价...")
        ... ]
        >>> SearchAgent._latest_human_text(messages)
        "深圳房价如何？"
        """
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, str):
                    return content.strip()
                return str(content or "").strip()
        return ""

    @staticmethod
    def _is_offtopic_weather_reply(user_text: str, answer_text: str) -> bool:
        """
        检测是否为主题跑偏：房产问题被误答为天气信息

        业务背景：
        - 房产类查询和天气查询容易混淆
        - 用户问"深圳房价"，模型可能误认为"深圳天气"
        - 需要检测这种情况并触发纠偏重试

        检测逻辑：
        1. 用户问题不包含天气关键词（如"天气"、"气温"等）
        2. 用户问题包含房产关键词（如"房价"、"楼盘"等）
        3. 回答中包含 2 个及以上天气关键词
        满足以上三个条件判定为主题跑偏

        设计考虑：
        - 使用关键词列表检测，避免误判
        - 天气关键词需要出现 2 次以上才判定，避免正常提及天气的情况
        - 用户文本必须包含房产关键词，避免其他主题被误判

        参数说明：
        - user_text: 用户原始问题文本
        - answer_text: AI 生成的回答文本

        返回值：
        - bool: True 表示判定为主题跑偏，False 表示正常

        示例：
        >>> user = "深圳最近房价怎么样？"
        >>> answer = "深圳今天气温 25 度，天气晴朗..."
        >>> SearchAgent._is_offtopic_weather_reply(user, answer)
        True
        >>> answer = "深圳最近房价稳中有升..."
        >>> SearchAgent._is_offtopic_weather_reply(user, answer)
        False
        """
        if not user_text or not answer_text:
            return False
        lower_user = user_text.lower()
        lower_answer = answer_text.lower()

        # 检测用户问题中是否包含天气和房产关键词
        user_has_weather = any(k in lower_user for k in SEARCH_WEATHER_KEYWORDS)
        user_has_estate = any(k in lower_user for k in SEARCH_REAL_ESTATE_KEYWORDS)

        # 统计回答中天气关键词出现次数
        answer_weather_hits = sum(1 for k in SEARCH_WEATHER_KEYWORDS if k in lower_answer)

        # 判定条件：用户没提天气 + 用户提了房产 + 回答包含多个天气关键词
        return (not user_has_weather) and user_has_estate and answer_weather_hits >= 2

    @staticmethod
    def _has_recent_tool_failure(messages: List[BaseMessage]) -> bool:
        """
        检测最近的工具调用是否失败

        设计目的：
        - 快速检测搜索工具是否超时或失败
        - 避免在工具失败后继续循环尝试，浪费 token
        - 及早返回友好提示，提升用户体验

        检测逻辑：
        - 检查最近 4 条消息中的 ToolMessage
        - 提取工具返回的文本内容
        - 检测是否包含失败关键词（中文或英文）

        失败关键词：
        - 中文：搜索超时、搜索失败、联网检索异常
        - 英文：timed out、timeout、error

        参数说明：
        - messages: 消息列表，用于查找工具消息

        返回值：
        - bool: True 表示检测到工具失败，False 表示未检测到

        使用场景：
        - 在模型调用前检查，如果已有失败则直接返回错误提示
        - 避免继续调用模型导致无限循环

        示例：
        >>> messages = [
        ...     HumanMessage("搜索深圳房价"),
        ...     AIMessage(content="", tool_calls=[{"name": "tavily_search"}]),
        ...     ToolMessage(content="搜索超时，请重试")
        ... ]
        >>> SearchAgent._has_recent_tool_failure(messages)
        True
        """
        # 只检查最近 4 条消息，避免误判历史失败
        recent = messages[-4:]
        for msg in recent:
            if not isinstance(msg, ToolMessage):
                continue

            # 提取工具返回的文本内容
            text = msg.content if isinstance(msg.content, str) else str(msg.content or "")
            lower = text.lower()

            # 检测失败关键词
            if any(k in lower for k in ("搜索超时", "搜索失败", "联网检索异常", "timed out", "timeout", "error")):
                return True
        return False

    async def _model_node(self, state: SearchAgentState):
        """
        搜索子图的模型节点：执行推理和搜索

        核心职责：
        1. 检查工具调用次数，超过上限则终止
        2. 检测工具失败，如有则返回错误提示
        3. 调用 LLM 生成搜索查询或直接回答
        4. 检测主题跑偏，如有则纠偏重试
        5. 更新工具调用次数计数器

        执行流程：
        1. 获取当前工具调用次数 loop_count
        2. 检查是否超过上限（search_max_tool_loops），如超过则返回终止消息
        3. 检查最近工具是否失败，如有则返回失败提示
        4. 使用 prompt + model_with_tools 链式调用 LLM
        5. 如果 LLM 未调用工具（直接生成回答），检测主题是否跑偏
        6. 如跑偏，使用 guard_prompt 重试一次
        7. 重试后仍跑偏，返回主题跑偏拦截消息
        8. 更新 loop_count + 1，返回 AI 消息

        循环控制机制：
        - 通过 tool_loop_count 追踪工具调用次数
        - 每次调用模型节点，计数器 +1
        - 超过配置的上限（search_max_tool_loops）则终止
        - 默认上限配置在 AGENT_LOOP_CONFIG 中

        参数说明：
        - state: 搜索子图状态，包含 messages 和 tool_loop_count

        返回值：
        - dict: 更新后的状态，包含 tool_loop_count 和 messages

        使用场景：
        - 由 LangGraph 在搜索子图中调用
        - 作为 ReAct 循环中的推理节点

        示例：
        >>> state = {
        ...     "messages": [HumanMessage("搜索深圳房价")],
        ...     "tool_loop_count": 0
        ... }
        >>> new_state = search_agent._model_node(state)
        >>> # new_state 包含 AI 消息和更新后的 loop_count
        """
        # 获取当前工具调用次数，默认为 0
        loop_count = int(state.get("tool_loop_count", 0) or 0)

        # 检查是否超过最大工具调用次数
        if loop_count >= AGENT_LOOP_CONFIG.search_max_tool_loops:
            return {
                "tool_loop_count": loop_count,
                "messages": [AIMessage(content=SEARCH_TOOL_LOOP_EXCEEDED_MESSAGE)],
            }

        # 检查最近工具是否失败
        if self._has_recent_tool_failure(state.get("messages", [])):
            return {
                "tool_loop_count": loop_count + 1,
                "messages": [AIMessage(content=SEARCH_TOOL_FAILURE_MESSAGE)],
            }

        # 调用 LLM，可能返回搜索工具调用或直接回答
        chain = self.prompt | self.model_with_tools
        ai_msg = await chain.ainvoke(state)

        # 如果 LLM 直接生成回答（未调用工具），检测主题是否跑偏
        if isinstance(ai_msg, AIMessage) and not getattr(ai_msg, "tool_calls", None):
            user_text = self._latest_human_text(state.get("messages", []))
            answer_text = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content or "")

            # 检测主题跑偏：房产问题误答天气
            if self._is_offtopic_weather_reply(user_text, answer_text):
                log.warning("SearchAgent 检测到主题跑偏（房产问题误答天气），执行一次纠偏重试。")

                # 构建重试消息：在原消息列表基础上添加纠偏指令
                retry_messages = list(state.get("messages", []))
                retry_messages.append(
                    HumanMessage(
                        content=(
                            f"纠偏要求：仅围绕当前问题回答，禁止天气播报。"
                            f"当前问题是：{user_text}"
                        )
                    )
                )

                # 使用防护提示词重试
                retry_chain = self.guard_prompt | self.model_with_tools
                retry_msg = await retry_chain.ainvoke({"messages": retry_messages})

                # 检查重试结果是否仍然跑偏
                if isinstance(retry_msg, AIMessage):
                    retry_text = retry_msg.content if isinstance(retry_msg.content, str) else str(retry_msg.content or "")
                    # 如果仍未调用工具且仍然跑偏，返回拦截消息
                    if not getattr(retry_msg, "tool_calls", None) and self._is_offtopic_weather_reply(user_text, retry_text):
                        ai_msg = AIMessage(content=SEARCH_TOPIC_DRIFT_BLOCK_MESSAGE)
                    else:
                        # 使用重试结果
                        ai_msg = retry_msg
                else:
                    # 重试失败，返回拦截消息
                    ai_msg = AIMessage(content=SEARCH_TOPIC_DRIFT_BLOCK_MESSAGE)

        # 更新工具调用次数，返回 AI 消息
        return {"tool_loop_count": loop_count + 1, "messages": [ai_msg]}

    def _build_graph(self) -> Runnable:
        """
        构建搜索子图

        设计目的：
        - 使用 BaseAgent 提供的通用 ReAct 工厂方法
        - 避免重复的拓扑代码
        - 统一子图构建模式

        子图结构：
        - 使用 _build_react_graph 方法构建
        - 包含模型节点（_model_node）和工具节点
        - 自动处理条件路由：有工具调用 → 工具节点，否则 → 结束
        - 自动处理循环控制和次数限制

        参数说明：
        - state_schema: 状态模式 SearchAgentState
        - model_node_fn: 模型节点函数 self._model_node
        - tools: 可用工具列表 [tavily_search_tool]
        - max_tool_loops: 最大工具循环次数
        - loop_exceeded_message: 超限时的提示消息

        返回值：
        - Runnable: 可执行的 LangGraph 子图

        使用场景：
        - 在 __init__ 中调用，构建完整的搜索子图
        - 子图可被主图的 Supervisor 调用
        """
        return self._build_react_graph(
            state_schema=SearchAgentState,
            model_node_fn=self._model_node,
            tools=self.tools,
            max_tool_loops=AGENT_LOOP_CONFIG.search_max_tool_loops,
            loop_exceeded_message=SEARCH_TOOL_LOOP_EXCEEDED_MESSAGE,
        )
