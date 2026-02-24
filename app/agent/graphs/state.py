from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

"""
【模块说明】
定义主图 (Supervisor) 的状态流转机 (StateGraph)。
State 是 LangGraph 的灵魂，所有的节点函数都只是在读取和修改这个字典。
"""


class GraphState(TypedDict):
    """
    定义图的全局状态。

    Attributes:
        messages: 核心对话历史。使用 add_messages 而不是 operator.add，
                  因为它能根据 Message ID 智能合并和去重（比如 ToolMessage 回传时）。
        next: 路由指示器，告诉图下一步该走哪个节点。
        session_id: 当前对话的唯一标识，用于持久化记忆。
        llm_config: 大模型的配置（温度、最大Token等）。
        routing_reason: 记录路由器的决策原因（用于监控和调试）。
        routing_confidence: 路由置信度。
    """
    messages: Annotated[List[BaseMessage], add_messages]
    next: Optional[str]
    session_id: Optional[str]
    llm_config: Optional[dict]
    routing_reason: Optional[str]
    routing_confidence: Optional[float]