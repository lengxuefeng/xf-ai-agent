from typing import Annotated, List, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langgraph.graph import add_messages

from agent.base import BaseAgent
from agent.graph_state import AgentRequest
from config.runtime_settings import AGENT_LOOP_CONFIG
from constants.agent_messages import (
    SEARCH_TOOL_FAILURE_MESSAGE,
    SEARCH_TOOL_LOOP_EXCEEDED_MESSAGE,
    SEARCH_TOPIC_DRIFT_BLOCK_MESSAGE,
)
from constants.search_keywords import SEARCH_REAL_ESTATE_KEYWORDS, SEARCH_WEATHER_KEYWORDS
from agent.tools.search_tools import tavily_search_tool
from utils.custom_logger import get_logger
from agent.prompts.tools_prompt import ToolsPrompt

log = get_logger(__name__)


class SearchAgentState(TypedDict):
    """搜索子图状态"""
    messages: Annotated[List[BaseMessage], add_messages]
    tool_loop_count: int  # 工具调用次数计数器


class SearchAgent(BaseAgent):
    """搜索Agent：处理互联网搜索查询"""

    def __init__(self, req: AgentRequest):
        super().__init__(req)
        if not req.model:
            raise ValueError("搜索模型初始化失败，请检查配置。")
        self.llm = req.model
        self.subgraph_id = "search_agent"

        # 工具
        self.tools = [tavily_search_tool]
        self.model_with_tools = self.llm.bind_tools(self.tools)

        # 提示词
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ToolsPrompt.SEARCH_SYSTEM),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
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
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, str):
                    return content.strip()
                return str(content or "").strip()
        return ""

    @staticmethod
    def _is_offtopic_weather_reply(user_text: str, answer_text: str) -> bool:
        if not user_text or not answer_text:
            return False
        lower_user = user_text.lower()
        lower_answer = answer_text.lower()
        user_has_weather = any(k in lower_user for k in SEARCH_WEATHER_KEYWORDS)
        user_has_estate = any(k in lower_user for k in SEARCH_REAL_ESTATE_KEYWORDS)
        answer_weather_hits = sum(1 for k in SEARCH_WEATHER_KEYWORDS if k in lower_answer)
        return (not user_has_weather) and user_has_estate and answer_weather_hits >= 2

    @staticmethod
    def _has_recent_tool_failure(messages: List[BaseMessage]) -> bool:
        """检测最近工具返回是否为超时/失败，命中后应快速收尾避免循环卡住。"""
        recent = messages[-4:]
        for msg in recent:
            if not isinstance(msg, ToolMessage):
                continue
            text = msg.content if isinstance(msg.content, str) else str(msg.content or "")
            lower = text.lower()
            if any(k in lower for k in ("搜索超时", "搜索失败", "联网检索异常", "timed out", "timeout", "error")):
                return True
        return False

    def _model_node(self, state: SearchAgentState):
        loop_count = int(state.get("tool_loop_count", 0) or 0)
        if loop_count >= AGENT_LOOP_CONFIG.search_max_tool_loops:
            return {
                "tool_loop_count": loop_count,
                "messages": [AIMessage(content=SEARCH_TOOL_LOOP_EXCEEDED_MESSAGE)],
            }
        if self._has_recent_tool_failure(state.get("messages", [])):
            return {
                "tool_loop_count": loop_count + 1,
                "messages": [AIMessage(content=SEARCH_TOOL_FAILURE_MESSAGE)],
            }
        chain = self.prompt | self.model_with_tools
        ai_msg = chain.invoke(state)
        if isinstance(ai_msg, AIMessage) and not getattr(ai_msg, "tool_calls", None):
            user_text = self._latest_human_text(state.get("messages", []))
            answer_text = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content or "")
            if self._is_offtopic_weather_reply(user_text, answer_text):
                log.warning("SearchAgent 检测到主题跑偏（房产问题误答天气），执行一次纠偏重试。")
                retry_messages = list(state.get("messages", []))
                retry_messages.append(
                    HumanMessage(
                        content=(
                            f"纠偏要求：仅围绕当前问题回答，禁止天气播报。"
                            f"当前问题是：{user_text}"
                        )
                    )
                )
                retry_chain = self.guard_prompt | self.model_with_tools
                retry_msg = retry_chain.invoke({"messages": retry_messages})
                if isinstance(retry_msg, AIMessage):
                    retry_text = retry_msg.content if isinstance(retry_msg.content, str) else str(retry_msg.content or "")
                    if not getattr(retry_msg, "tool_calls", None) and self._is_offtopic_weather_reply(user_text, retry_text):
                        ai_msg = AIMessage(content=SEARCH_TOPIC_DRIFT_BLOCK_MESSAGE)
                    else:
                        ai_msg = retry_msg
                else:
                    ai_msg = AIMessage(content=SEARCH_TOPIC_DRIFT_BLOCK_MESSAGE)
        return {"tool_loop_count": loop_count + 1, "messages": [ai_msg]}

    def _build_graph(self) -> Runnable:
        """使用通用 ReAct 工厂构建搜索子图，消除重复拓扑样板。"""

        return self._build_react_graph(
            state_schema=SearchAgentState,
            model_node_fn=self._model_node,
            tools=self.tools,
            max_tool_loops=AGENT_LOOP_CONFIG.search_max_tool_loops,
            loop_exceeded_message=SEARCH_TOOL_LOOP_EXCEEDED_MESSAGE,
        )
