# -*- coding: utf-8 -*-
"""Tracing 运行时守卫。"""

from __future__ import annotations

import os
from typing import MutableMapping


_TRUTHY_VALUES = {"1", "true", "yes", "y", "on"}
_TRACING_FLAG_KEYS = (
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
    "TRACING",
    "TRACING_V2",
)


def _is_tracing_enabled(raw: str) -> bool:
    """统一判定 tracing 是否显式开启。"""
    return str(raw or "").strip().lower() in _TRUTHY_VALUES


def apply_tracing_env_guard(env: MutableMapping[str, str] | None = None) -> bool:
    """
    当 LANGSMITH_TRACING 未开启时，统一关闭所有兼容 tracing 开关。

    Returns:
        bool: True 表示本次执行了 guard 并改写了环境变量。
    """
    target_env = env if env is not None else os.environ
    if _is_tracing_enabled(target_env.get("LANGSMITH_TRACING", "false")):
        return False

    for key in _TRACING_FLAG_KEYS:
        target_env[key] = "false"
    return True

