import operator
from typing import TypedDict, Annotated, Optional, Any

"""
定义主图 LangGraph 的状态管理
"""


class GraphState(TypedDict):
    """
    定义图的全局状态。

    这个 TypedDict 作为状态图 (StateGraph) 中节点之间传递的数据结构。
    每个节点都可以访问和修改这个状态对象的字段。

    Attributes:
        messages (Annotated[list, operator.add]): 对话消息列表
        next (Optional[str]): 指示下一个要调用的节点的名称。
        interrupt (Optional[str]): 存储需要人工干预时的中断信息。
        session_id (Optional[str]): 会话 ID。
        llm_config (Optional[dict]): 模型配置参数。
    """
    messages: Annotated[list, operator.add]
    next: Optional[str]
    interrupt: Optional[Any]
    session_id: Optional[str]
    llm_config: Optional[dict]
