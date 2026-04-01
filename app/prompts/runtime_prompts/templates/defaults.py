# -*- coding: utf-8 -*-
"""运行时 Prompt 默认导出。"""
from prompts.prompt_loader import load_prompt_template

GLOBAL_RUNTIME_PROMPT = load_prompt_template("runtime_prompts/templates/global_runtime.txt")
