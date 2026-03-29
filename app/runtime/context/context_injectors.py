# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List

from langchain_core.messages import SystemMessage


def build_context_messages(*system_blocks: str) -> List[SystemMessage]:
    messages: List[SystemMessage] = []
    for block in system_blocks:
        text = str(block or "").strip()
        if text:
            messages.append(SystemMessage(content=text))
    return messages

