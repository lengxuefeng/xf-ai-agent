# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class QueryBudget:
    """统一 Query Runtime 预算声明。"""

    max_model_turns: int = 8
    max_tool_calls: int = 8
    max_output_chars: int = 24000
    max_worker_dispatches: int = 8

    def to_dict(self) -> dict:
        return asdict(self)

