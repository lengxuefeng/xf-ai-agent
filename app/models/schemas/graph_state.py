"""兼容旧引用的全局状态别名。"""

from app.supervisor.graph_state import AgentState

GraphState = AgentState

__all__ = ["AgentState", "GraphState"]
