"""Supervisor/Agent 全局唯一状态定义。"""

from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from models.schemas.base import ArbitraryTypesBaseSchema


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: list[str]
    current_task: str


class AgentRequest(ArbitraryTypesBaseSchema):
    user_input: str
    state: Optional[dict[str, Any]] = None
    session_id: str
    subgraph_id: str
    model: Any
    llm_config: Optional[dict[str, Any]] = None


class BatchAgentRequest(ArbitraryTypesBaseSchema):
    inputs: list[AgentRequest]
    max_threads: int = 2
    model: Any


__all__ = ["AgentState", "AgentRequest", "BatchAgentRequest"]
