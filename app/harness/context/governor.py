# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

from langchain_core.messages import BaseMessage

from prompts.runtime_prompts.prompt_cache_adapter import RuntimePromptBlock


@dataclass(slots=True)
class ContextLayer:
    """上下文治理层快照。"""

    segment: str
    boundary: str
    char_count: int
    item_count: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RuntimeContextGovernor:
    """统一描述 prompt / history / memory 的上下文边界。"""

    @staticmethod
    def _message_char_count(messages: Sequence[BaseMessage]) -> int:
        total = 0
        for message in messages or []:
            content = getattr(message, "content", "")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                total += sum(len(str(item or "")) for item in content)
            elif isinstance(content, dict):
                total += len(str(content))
            else:
                total += len(str(content or ""))
        return total

    def describe_layers(
        self,
        *,
        prompt_blocks: Iterable[RuntimePromptBlock] | None,
        filtered_history: Sequence[BaseMessage] | None,
        compressed_history: Sequence[BaseMessage] | None,
        memory_count: int,
        dynamic_blocks: Sequence[str] | None,
    ) -> List[ContextLayer]:
        layers: list[ContextLayer] = []
        for block in prompt_blocks or []:
            layers.append(
                ContextLayer(
                    segment=str(block.segment),
                    boundary=str(getattr(block, "boundary", "volatile") or "volatile"),
                    char_count=len(str(block.content or "")),
                    item_count=1,
                )
            )

        normalized_dynamic_blocks = [str(item or "").strip() for item in (dynamic_blocks or []) if str(item or "").strip()]
        if normalized_dynamic_blocks:
            layers.append(
                ContextLayer(
                    segment="dynamic_blocks",
                    boundary="volatile",
                    char_count=sum(len(item) for item in normalized_dynamic_blocks),
                    item_count=len(normalized_dynamic_blocks),
                )
            )

        layers.append(
            ContextLayer(
                segment="recent_history",
                boundary="volatile",
                char_count=self._message_char_count(filtered_history or []),
                item_count=len(filtered_history or []),
            )
        )
        layers.append(
            ContextLayer(
                segment="compacted_history",
                boundary="compact_boundary",
                char_count=self._message_char_count(compressed_history or []),
                item_count=len(compressed_history or []),
                meta={
                    "compacted": len(compressed_history or []) <= len(filtered_history or []),
                },
            )
        )
        layers.append(
            ContextLayer(
                segment="memory_snippets",
                boundary="volatile",
                char_count=0,
                item_count=int(memory_count or 0),
            )
        )
        return layers


runtime_context_governor = RuntimeContextGovernor()

