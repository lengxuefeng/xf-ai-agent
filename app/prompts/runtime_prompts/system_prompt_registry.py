# -*- coding: utf-8 -*-
from __future__ import annotations

from prompts.prompt_loader import render_prompt_template


class SystemPromptRegistry:
    """统一系统提示词注册表。"""

    def __init__(self) -> None:
        self._prompt_paths = {
            "global_runtime": "runtime_prompts/templates/global_runtime.txt",
        }

    def render_global_prompt(self, *, skill_prompts: list[str] | None = None) -> str:
        base_prompt = render_prompt_template(self._prompt_paths["global_runtime"])
        normalized_skill_prompts = [str(item or "").strip() for item in (skill_prompts or []) if str(item or "").strip()]
        if not normalized_skill_prompts:
            return base_prompt
        skill_block = "\n\n".join(normalized_skill_prompts)
        return (
            f"{base_prompt}\n\n"
            "## 当前启用的 Agent Skills\n"
            f"{skill_block}"
        ).strip()


system_prompt_registry = SystemPromptRegistry()
