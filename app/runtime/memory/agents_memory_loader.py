# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path


def load_agents_memory() -> str:
    project_root = Path(__file__).resolve().parents[3]
    agents_path = project_root / "AGENTS.md"
    if not agents_path.exists():
        return ""
    return agents_path.read_text(encoding="utf-8").strip()

