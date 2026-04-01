# -*- coding: utf-8 -*-
from __future__ import annotations

from prompts.prompt_loader import render_prompt_template


class SystemPromptRegistry:
    """统一系统提示词注册表。"""

    def __init__(self) -> None:
        self._prompt_paths = {
            "global_runtime": "runtime_prompts/templates/global_runtime.txt",
        }

    def render_global_prompt(self) -> str:
        return render_prompt_template(self._prompt_paths["global_runtime"])


system_prompt_registry = SystemPromptRegistry()
