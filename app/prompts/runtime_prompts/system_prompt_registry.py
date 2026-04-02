# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Iterable, List, Sequence

from common.llm.unified_loader import resolve_provider_family
from prompts.prompt_loader import render_prompt_template
from prompts.runtime_prompts.prompt_cache_adapter import (
    RuntimePromptBlock,
    prompt_cache_adapter,
)


class SystemPromptRegistry:
    """统一系统提示词注册表。"""

    def __init__(self) -> None:
        self._prompt_paths = {
            "global_runtime": "runtime_prompts/templates/global_runtime.txt",
        }

    @staticmethod
    def _normalize_blocks(values: Iterable[str] | None) -> List[str]:
        return [str(item or "").strip() for item in (values or []) if str(item or "").strip()]

    def _build_tool_summary(self, tool_catalog: Sequence[dict] | None) -> str:
        catalog = sorted(
            [dict(item) for item in (tool_catalog or []) if isinstance(item, dict)],
            key=lambda item: str(item.get("name") or "").lower(),
        )
        if not catalog:
            return "## Runtime Tools\n- 当前未启用额外运行时工具。"

        lines = ["## Runtime Tools"]
        for item in catalog:
            name = str(item.get("name") or "").strip()
            description = str(item.get("description") or "").strip()
            source = str(item.get("source") or "").strip()
            if not name:
                continue
            suffix = f": {description}" if description else ""
            if source:
                suffix += f" (source={source})"
            lines.append(f"- {name}{suffix}")
        return "\n".join(lines)

    def _build_skill_rules(
        self,
        *,
        skill_prompts: Sequence[str] | None,
        skill_names: Sequence[str] | None,
    ) -> str:
        normalized_skill_prompts = self._normalize_blocks(skill_prompts)
        normalized_skill_names = self._normalize_blocks(skill_names)
        if not normalized_skill_prompts:
            return "## Active Skills\n- 当前未启用额外 Agent Skills。"

        title_suffix = f" ({', '.join(normalized_skill_names)})" if normalized_skill_names else ""
        lines = [f"## Active Skills{title_suffix}"]
        for index, prompt in enumerate(normalized_skill_prompts, start=1):
            lines.append(f"### Skill {index}")
            lines.append(prompt)
        return "\n\n".join(lines)

    def render_global_prompt_blocks(
        self,
        *,
        model_config: dict | None = None,
        tool_catalog: Sequence[dict] | None = None,
        skill_prompts: Sequence[str] | None = None,
        skill_names: Sequence[str] | None = None,
        memory_block: str = "",
        dynamic_blocks: Iterable[str] | None = None,
        runtime_notes: Iterable[str] | None = None,
        extra_static_blocks: Iterable[str] | None = None,
    ) -> List[RuntimePromptBlock]:
        static_blocks = [
            render_prompt_template(self._prompt_paths["global_runtime"]),
            *self._normalize_blocks(extra_static_blocks),
        ]
        config_blocks = [
            self._build_tool_summary(tool_catalog),
            self._build_skill_rules(skill_prompts=skill_prompts, skill_names=skill_names),
        ]
        normalized_dynamic_blocks = self._normalize_blocks(dynamic_blocks)
        normalized_runtime_notes = self._normalize_blocks(runtime_notes)

        blocks = [
            RuntimePromptBlock(segment="static", content="\n\n".join(static_blocks).strip(), boundary="cacheable"),
            RuntimePromptBlock(segment="config", content="\n\n".join(config_blocks).strip(), boundary="cacheable"),
        ]
        if str(memory_block or "").strip():
            blocks.append(
                RuntimePromptBlock(
                    segment="memory",
                    content=str(memory_block).strip(),
                    boundary="volatile",
                )
            )
        if normalized_dynamic_blocks:
            blocks.append(
                RuntimePromptBlock(
                    segment="dynamic",
                    content="\n\n".join(normalized_dynamic_blocks).strip(),
                    boundary="volatile",
                )
            )
        if normalized_runtime_notes:
            blocks.append(
                RuntimePromptBlock(
                    segment="runtime_notes",
                    content="\n\n".join(normalized_runtime_notes).strip(),
                    boundary="compact_boundary",
                )
            )
        return [block for block in blocks if block.content]

    def build_runtime_messages(
        self,
        *,
        model_config: dict | None = None,
        tool_catalog: Sequence[dict] | None = None,
        skill_prompts: Sequence[str] | None = None,
        skill_names: Sequence[str] | None = None,
        memory_block: str = "",
        dynamic_blocks: Iterable[str] | None = None,
        runtime_notes: Iterable[str] | None = None,
        extra_static_blocks: Iterable[str] | None = None,
    ):
        config = model_config or {}
        blocks = self.render_global_prompt_blocks(
            model_config=config,
            tool_catalog=tool_catalog,
            skill_prompts=skill_prompts,
            skill_names=skill_names,
            memory_block=memory_block,
            dynamic_blocks=dynamic_blocks,
            runtime_notes=runtime_notes,
            extra_static_blocks=extra_static_blocks,
        )
        provider_family = resolve_provider_family(
            service_type=str(config.get("service_type") or ""),
            model_service=str(config.get("model_service") or ""),
            model_name=str(config.get("model") or ""),
        ).value
        return prompt_cache_adapter.to_messages(blocks, provider_family=provider_family)

    def render_global_prompt(self, *, skill_prompts: list[str] | None = None) -> str:
        blocks = self.render_global_prompt_blocks(skill_prompts=skill_prompts)
        return "\n\n".join(block.content for block in blocks if block.content).strip()


system_prompt_registry = SystemPromptRegistry()
