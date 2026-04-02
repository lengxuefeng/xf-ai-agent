# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

from models.schemas.agent_runtime_schemas import AgentRequest
from runtime.workers.base import RuntimeWorker
from runtime.workers.code_worker import code_worker
from runtime.workers.search_worker import search_worker
from runtime.workers.weather_worker import weather_worker


class RuntimeWorkerRegistry:
    """Runtime Worker 注册表。"""

    def __init__(self) -> None:
        self._workers = {
            "search_agent": search_worker,
            "code_agent": code_worker,
            "weather_agent": weather_worker,
        }

    def get_worker(self, agent_name: str, req: AgentRequest) -> Optional[RuntimeWorker]:
        worker = self._workers.get(str(agent_name or "").strip())
        if worker is None:
            return None
        if not worker.supports(req):
            return None
        return worker


runtime_worker_registry = RuntimeWorkerRegistry()
