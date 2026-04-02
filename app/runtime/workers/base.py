# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import re
from typing import Any, Dict

from langchain_core.runnables import RunnableConfig

from services.agent_stream_bus import agent_stream_bus
from models.schemas.agent_runtime_schemas import AgentRequest


class RuntimeWorker(ABC):
    """统一 Runtime Worker 抽象。"""

    worker_name: str = "runtime_worker"
    agent_name: str = ""

    @abstractmethod
    def supports(self, req: AgentRequest) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def run(self, req: AgentRequest, *, config: RunnableConfig) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def _resolve_run_id(config: RunnableConfig) -> str:
        configurable = dict((config or {}).get("configurable") or {})
        return str(
            configurable.get("run_id")
            or configurable.get("thread_id")
            or ""
        ).strip()

    @staticmethod
    def _resolve_task_id(req: AgentRequest) -> str:
        return str(((req.state or {}).get("task_id") or "")).strip()

    @staticmethod
    def _chunk_text(text: str, *, chunk_size: int = 24) -> list[str]:
        raw_text = str(text or "")
        if not raw_text:
            return []

        parts: list[str] = []
        for line in re.split(r"(\n)", raw_text):
            if not line:
                continue
            if line == "\n":
                parts.append(line)
                continue
            for start in range(0, len(line), max(1, chunk_size)):
                parts.append(line[start:start + max(1, chunk_size)])
        return parts

    async def _publish_stream_text(
        self,
        *,
        req: AgentRequest,
        config: RunnableConfig,
        text: str,
        chunk_size: int = 24,
        body_stream: bool = False,
    ) -> None:
        run_id = self._resolve_run_id(config)
        if not run_id:
            return
        task_id = self._resolve_task_id(req)
        for chunk in self._chunk_text(text, chunk_size=chunk_size):
            agent_stream_bus.publish(
                run_id=run_id,
                agent_name=self.agent_name or self.worker_name,
                content=chunk,
                task_id=task_id,
                body_stream=bool(body_stream),
            )
            await asyncio.sleep(0)
