# -*- coding: utf-8 -*-
from __future__ import annotations

from runtime.prompts.templates.defaults import GLOBAL_RUNTIME_PROMPT


class SystemPromptRegistry:
    """统一系统提示词注册表。"""

    def __init__(self) -> None:
        self._prompts = {
            "global_runtime": GLOBAL_RUNTIME_PROMPT,
        }

    def render_global_prompt(self) -> str:
        return self._prompts["global_runtime"]


system_prompt_registry = SystemPromptRegistry()
