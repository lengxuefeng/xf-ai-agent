# -*- coding: utf-8 -*-
from __future__ import annotations

from runtime.query.engine import RuntimeEngine
from harness.types import RunContext
from runtime.query.models import QueryState


class QueryLoop:
    """Query Runtime 主循环骨架。

    当前阶段只负责围绕 legacy LangGraph 执行器维护统一 QueryState，
    后续再逐步承接真正的 tool_use / tool_result 循环。
    """

    def __init__(self, engine: RuntimeEngine) -> None:
        self._engine = engine

    def open_legacy_query(self, run_context: RunContext, *, executor_name: str) -> QueryState:
        state = self._engine.bootstrap_query(run_context, executor_name=executor_name)
        return self._engine.mark_running(state)

