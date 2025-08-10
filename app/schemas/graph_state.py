from typing import TypedDict, Literal, Annotated, Optional

from langgraph.graph import add_messages


class GraphState(TypedDict):
    """
    定义图状态的类型字典，用于表示图的状态。

    参数：
        TypedDict (class): 用于定义类型字典的基类。
    """
    model_name: str  # 模型名称
    type: Literal["websearch", "file", "chat"]  # 图的可操作类型，包括联网搜索、文件操作和聊天
    messages: Annotated[list, add_messages]  # 消息列表，用于存储对话中的消息，使用add_messages注解处理消息追加
    documents: Optional[list] = []  # 文档列表，用于存储相关文档的信息，默认为空列表
