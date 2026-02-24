from typing import TypedDict, List, Optional, Dict, Any

from langchain_core.language_models import BaseChatModel

from schemas.base import ArbitraryTypesBaseSchema

"""
【模块说明】
定义系统内部流转的强类型请求对象。
这保证了从 API 层传递到 Agent 层的参数不会出现类型错误。
"""


class AgentRequest(ArbitraryTypesBaseSchema):
    """
    单智能体请求载荷封装
    """
    user_input: str  # 用户当前轮次的输入文本
    state: Optional[TypedDict] = None  # 外部传入的初始状态（按需使用）
    session_id: str  # 用于绑定 Checkpointer 的线程 ID
    subgraph_id: str  # 子图命名空间标识
    model: BaseChatModel  # 已经初始化好的 LLM 实例（如 GLM-4）
    llm_config: Optional[Dict[str, Any]] = None  # 其他模型参数配置


class BatchAgentRequest(ArbitraryTypesBaseSchema):
    """
    批量处理请求载荷（用于并发场景）
    """
    inputs: List[AgentRequest]
    max_threads: int = 2
    model: BaseChatModel
