# -*- coding: utf-8 -*-
from __future__ import annotations


class ContextBudget:
    """轻量上下文预算估算器。"""

    @staticmethod
    def estimate_chars_budget(text: str) -> int:
        return len(str(text or ""))

    @staticmethod
    def estimate_token_budget(text: str) -> int:
        return max(1, len(str(text or "")) // 2)

