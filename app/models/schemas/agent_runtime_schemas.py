# -*- coding: utf-8 -*-
"""运行时与 Supervisor 之间共享的强类型请求模型。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from models.schemas.base import ArbitraryTypesBaseSchema


class AgentRequest(ArbitraryTypesBaseSchema):
    """单个 Agent 的统一请求载荷。"""

    user_input: str = Field(default="", description="用户本轮输入内容。")
    state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="传递给子 Agent 的初始状态，例如历史消息、上下文摘要等。",
    )
    session_id: str = Field(default="", description="会话 ID，用于 checkpointer 的 thread_id。")
    subgraph_id: str = Field(default="", description="子图命名空间标识，例如 sql_agent。")
    model: Any = Field(default=None, description="已初始化的大模型实例。")
    llm_config: Optional[Dict[str, Any]] = Field(default=None, description="模型扩展配置。")


class SupervisorExecutionState(ArbitraryTypesBaseSchema):
    """Supervisor 串行执行态的轻量声明。"""

    plan: List[str] = Field(default_factory=list, description="待执行的原子步骤列表。")
    current_task: str = Field(default="", description="当前正在执行的任务描述。")


class BatchAgentRequest(ArbitraryTypesBaseSchema):
    """批量 Agent 请求封装。"""

    inputs: List[AgentRequest] = Field(default_factory=list, description="待批量执行的 Agent 请求列表。")
    max_threads: int = Field(default=2, ge=1, description="批量执行时允许的最大并发线程数。")
    model: Any = Field(default=None, description="批量场景共享的大模型实例。")


__all__ = ["AgentRequest", "BatchAgentRequest", "SupervisorExecutionState"]
