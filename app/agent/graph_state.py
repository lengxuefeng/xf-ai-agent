import uuid
from typing import TypedDict, List, Optional, Dict, Any

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel


class AgentRequest(BaseModel):
    """
    代码编写智能体请求

    Attributes:
        user_input (str): 用户输入的原始文本。
        state (TypedDict, optional): 智能体的当前状态。
        session_id (str): 会话 ID，用于关联多个交互。
        subgraph_id (str): 子图 ID，用于指定使用的子图。
        model (BaseChatModel): 语言模型，用于执行任务。
        llm_config (Dict[str, Any], optional): 模型配置参数。
    """
    user_input: str
    state: Optional[TypedDict] = None
    session_id: str
    subgraph_id: str
    model: BaseChatModel
    llm_config: Optional[Dict[str, Any]] = None
    
    class Config:
        arbitrary_types_allowed = True


class BatchAgentRequest(BaseModel):
    """
    批量智能体请求

    Attributes:
        inputs (List[AgentRequest]): 包含多个智能体请求的列表。
        max_threads (int, optional): 最大并发线程数，默认值为 2。
        model (BaseChatModel): 语言模型，用于执行批量任务。
    """
    inputs: List[AgentRequest]
    max_threads: int = 2
    model: BaseChatModel
    
    class Config:
        arbitrary_types_allowed = True
