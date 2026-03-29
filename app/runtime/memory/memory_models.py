# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class MemorySnippet:
    key: str
    source: str
    content: str
    scope: str = "session"

    def to_dict(self) -> dict:
        return asdict(self)

