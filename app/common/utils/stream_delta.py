# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple


def resolve_stream_delta(accumulated_text: str, next_chunk: str) -> Tuple[str, str]:
    """将流式 chunk 统一规整为真正的增量文本。

    部分模型适配层返回的是“累计到当前为止的完整文本”，
    也有一部分返回“纯增量 token”。这里统一兼容两种语义：
    - 如果 `next_chunk` 是累计文本，则只返回新增尾部；
    - 如果 `next_chunk` 已经完全包含在当前累计结果末尾，则忽略；
    - 其他情况按普通增量直接追加。
    """
    accumulated = str(accumulated_text or "")
    chunk = str(next_chunk or "")

    if not chunk:
        return "", accumulated
    if not accumulated:
        return chunk, chunk
    if chunk == accumulated:
        return "", accumulated
    if chunk.startswith(accumulated):
        return chunk[len(accumulated):], chunk
    if accumulated.endswith(chunk):
        return "", accumulated

    max_overlap = min(len(accumulated), len(chunk))
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if accumulated.endswith(chunk[:size]):
            overlap = size
            break

    if overlap > 0:
        delta = chunk[overlap:]
        if not delta:
            return "", accumulated
        return delta, accumulated + delta

    return chunk, accumulated + chunk
