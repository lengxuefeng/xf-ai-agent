# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List

from runtime.memory.agents_memory_loader import load_agents_memory
from runtime.memory.memory_models import MemorySnippet
from runtime.types import RunContext


class RuntimeMemoryService:
    """统一运行时记忆服务。"""

    def build_memory_snippets(
        self,
        *,
        run_context: RunContext,
        session_context: Dict[str, Any],
    ) -> List[MemorySnippet]:
        snippets: List[MemorySnippet] = []
        project_memory = load_agents_memory()
        if project_memory:
            snippets.append(
                MemorySnippet(
                    key="project_agents_md",
                    source="AGENTS.md",
                    content=project_memory,
                    scope="project",
                )
            )

        context_summary = str(session_context.get("context_summary") or "").strip()
        if context_summary:
            snippets.append(
                MemorySnippet(
                    key="session_summary",
                    source="session_state",
                    content=context_summary,
                    scope="session",
                )
            )

        context_slots = session_context.get("context_slots") or {}
        if isinstance(context_slots, dict) and context_slots:
            slots_lines = [f"{key}: {value}" for key, value in context_slots.items() if value not in (None, "", [], {})]
            if slots_lines:
                snippets.append(
                    MemorySnippet(
                        key="session_slots",
                        source="session_state",
                        content="\n".join(slots_lines),
                        scope="session",
                    )
                )

        snippets.append(
            MemorySnippet(
                key="current_request",
                source="run_context",
                content=f"当前用户输入：{run_context.user_input}",
                scope="request",
            )
        )
        return snippets

    def render_memory_block(self, snippets: List[MemorySnippet]) -> str:
        visible = [snippet for snippet in snippets if snippet.content]
        if not visible:
            return ""
        lines = ["【运行时记忆注入】"]
        for snippet in visible:
            lines.append(f"- [{snippet.scope}] {snippet.source}: {snippet.content}")
        return "\n".join(lines)


runtime_memory_service = RuntimeMemoryService()

