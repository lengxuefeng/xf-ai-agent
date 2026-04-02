# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict

from supervisor.supervisor import create_graph as create_supervisor_graph


class LangGraphSupervisorShell:
    """LangGraph 外层图壳。"""

    shell_name = "langgraph_supervisor_shell"

    def compile_supervisor(self, model_config: Dict[str, Any]) -> Any:
        return create_supervisor_graph(model_config)

    def describe(self, model_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        config = dict(model_config or {})
        return {
            "shell": self.shell_name,
            "checkpoint_backed": True,
            "supports_interrupt_resume": True,
            "router_model": str(config.get("router_model") or ""),
            "simple_chat_model": str(config.get("simple_chat_model") or ""),
        }


langgraph_supervisor_shell = LangGraphSupervisorShell()

