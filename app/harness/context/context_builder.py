# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from harness.context.context_budget import ContextBudget
from harness.context.context_injectors import build_context_messages
from harness.memory.memory_models import MemorySnippet
from harness.memory.memory_service import runtime_memory_service
from prompts.runtime_prompts.system_prompt_registry import system_prompt_registry
from harness.types import RunContext
from common.utils.date_utils import get_agent_date_context
from common.utils.history_compressor import compress_history_messages


class RuntimeContextBuilder:
    """统一上下文构建器。"""

    def build_messages(
        self,
        *,
        run_context: RunContext,
        history_messages: List[Dict[str, Any]],
        session_context: Dict[str, Any],
        max_tokens: int,
        max_chars: int,
    ) -> Tuple[List[BaseMessage], List[MemorySnippet], Dict[str, Any]]:
        memory_snippets = runtime_memory_service.build_memory_snippets(
            run_context=run_context,
            session_context=session_context,
        )
        memory_block = runtime_memory_service.render_memory_block(memory_snippets)
        system_prompt = system_prompt_registry.render_global_prompt()
        messages: List[BaseMessage] = build_context_messages(
            get_agent_date_context(),
            system_prompt,
            memory_block,
        )

        raw_history: List[BaseMessage] = []
        for msg in history_messages or []:
            if msg.get("user_content"):
                raw_history.append(HumanMessage(content=msg["user_content"]))
            if msg.get("model_content"):
                raw_history.append(AIMessage(content=msg["model_content"], name=msg.get("name")))

        compressed_history = compress_history_messages(
            raw_history,
            model=None,
            max_tokens=max_tokens,
            max_chars=max_chars,
        )
        messages.extend(compressed_history)
        messages.append(HumanMessage(content=run_context.user_input))

        context_meta = {
            "history_message_count": len(raw_history),
            "compressed_history_count": len(compressed_history),
            "memory_count": len(memory_snippets),
            "estimated_tokens": sum(ContextBudget.estimate_token_budget(getattr(message, "content", "")) for message in messages),
        }
        return messages, memory_snippets, context_meta


runtime_context_builder = RuntimeContextBuilder()

