# -*- coding: utf-8 -*-
"""统一的 Prompt 模板加载器。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


_PROMPTS_ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=128)
def load_prompt_template(relative_path: str) -> str:
    """按相对路径加载 Prompt 模板文件。"""
    template_path = (_PROMPTS_ROOT / relative_path).resolve()
    return template_path.read_text(encoding="utf-8").strip()


def render_prompt_template(relative_path: str, **kwargs: Any) -> str:
    """加载并渲染 Prompt 模板。"""
    template = load_prompt_template(relative_path)
    if not kwargs:
        return template
    return template.format(**kwargs).strip()
