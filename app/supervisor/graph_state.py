import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage

from models.schemas.base import ArbitraryTypesBaseSchema


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    plan: list[str]
    current_task: str


class AgentRequest(ArbitraryTypesBaseSchema):
    user_input: str
    state: Optional[Dict[str, Any]] = None
    session_id: str
    subgraph_id: str
    model: Any
    llm_config: Optional[Dict[str, Any]] = None


class BatchAgentRequest(ArbitraryTypesBaseSchema):
    inputs: List[AgentRequest]
    max_threads: int = 2
    model: Any
