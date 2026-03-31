from typing import Optional, List, TypedDict

from supervisor.graph_state import AgentState


class SubTask(TypedDict):
    id: str
    agent: str
    input: str
    depends_on: List[str]
    status: str
    result: Optional[str]


class WorkerResult(TypedDict):
    task_id: str
    task: Optional[str]
    result: str
    error: Optional[str]
    agent: Optional[str]
    elapsed_ms: Optional[int]


GraphState = AgentState
