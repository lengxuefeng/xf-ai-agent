# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from langchain_core.messages import SystemMessage


@dataclass(slots=True)
class RuntimePromptBlock:
    """结构化运行时提示词分段。"""

    segment: str
    content: str


class PromptCacheAdapter:
    """按不同 Provider Family 适配系统提示前缀。"""

    def to_messages(
        self,
        blocks: Iterable[RuntimePromptBlock],
        *,
        provider_family: str,
    ) -> List[SystemMessage]:
        normalized_blocks = [
            RuntimePromptBlock(segment=str(block.segment), content=str(block.content or "").strip())
            for block in (blocks or [])
            if str(getattr(block, "content", "") or "").strip()
        ]
        if not normalized_blocks:
            return []

        family = str(provider_family or "generic").strip().lower()
        messages: list[SystemMessage] = []
        for block in normalized_blocks:
            additional_kwargs = {}
            if family == "anthropic" and block.segment in {"static", "config"}:
                additional_kwargs["cache_control"] = {"type": "ephemeral"}
            messages.append(
                SystemMessage(
                    content=block.content,
                    additional_kwargs=additional_kwargs,
                )
            )
        return messages


prompt_cache_adapter = PromptCacheAdapter()


__all__ = ["PromptCacheAdapter", "RuntimePromptBlock", "prompt_cache_adapter"]
