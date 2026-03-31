# -*- coding: utf-8 -*-
"""
历史消息压缩工具。

策略：
1. 优先使用 LangChain 原生 `trim_messages` 做窗口裁剪（兼容 token 计数器）。
2. 若运行环境或参数不支持，则回退到本地字符预算压缩。
"""
from __future__ import annotations

from typing import Any, Iterable, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from common.utils.custom_logger import get_logger

log = get_logger(__name__)


class _ModelTokenCounter:
    """对模型 token 计数能力做安全包装，失败时回退本地估算。"""

    def __init__(self, get_tokens) -> None:
        self._get_tokens = get_tokens

    def __call__(self, messages: List[BaseMessage]) -> int:
        try:
            value = self._get_tokens(messages)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
        except Exception:
            pass
        return _estimate_tokens_from_messages(messages)


def _message_text(msg: BaseMessage) -> str:
    """将消息内容规范化为字符串"""
    # 获取消息内容
    content = getattr(msg, "content", "")

    # 如果是字符串直接返回
    if isinstance(content, str):
        return content

    # 如果是列表，提取所有文本内容
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)  # 直接添加字符串
            elif isinstance(item, dict):
                # 从字典中提取 text 或 content 字段
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)  # 用换行符连接

    # 其他情况转为字符串
    return str(content or "")


def _fallback_trim_by_chars(messages: Iterable[BaseMessage], max_chars: int) -> List[BaseMessage]:
    """
    按字符预算回退压缩策略
    从最新消息向前保留，直到达到预算上限
    """
    budget = max(100, int(max_chars))  # 最小保留 100 字符
    kept: list[BaseMessage] = []      # 保留的消息列表
    used = 0                           # 已使用的字符数

    # 从后往前遍历消息（保留最新的）
    for msg in reversed(list(messages)):
        txt = _message_text(msg)     # 获取消息文本
        msg_len = len(txt)            # 计算消息长度
        # 如果已有消息且加入后会超预算，则停止
        if kept and used + msg_len > budget:
            break
        kept.append(msg)              # 保留该消息
        used += msg_len               # 累计使用字符

    # 反转回原始顺序返回
    return list(reversed(kept))


def _estimate_tokens_from_messages(messages: Iterable[BaseMessage]) -> int:
    """模型缺少原生 token 计数能力时的轻量估算器。"""
    total = 0
    for msg in messages:
        text = _message_text(msg)
        if not text:
            continue
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        english_words = len([word for word in text.split() if word.isalpha()])
        other_chars = max(0, len(text) - chinese_chars)
        total += chinese_chars + english_words + int(other_chars * 0.5)
    return max(1, int(total))


def _build_token_counter(model: Optional[Any]) -> Any:
    """
    构建 trim_messages 的 token_counter。

    优先使用 LangChain 新版内置近似计数器（'approximate'），
    只有在模型明确支持 get_num_tokens_from_messages 时才使用模型计数。
    这样可避免 GLM 等模型抛出“未实现 token 计数”导致压缩失败。
    """
    if model is None:
        return "approximate"

    get_tokens = getattr(model, "get_num_tokens_from_messages", None)
    if not callable(get_tokens):
        return "approximate"

    # 先做一次轻量探测，确认该模型真正实现了计数接口。
    try:
        probe_value = get_tokens([HumanMessage(content="ping")])
        if not isinstance(probe_value, (int, float)) or probe_value <= 0:
            return "approximate"
    except Exception:
        return "approximate"
    return _ModelTokenCounter(get_tokens)


def compress_history_messages(
    messages: List[BaseMessage],
    *,
    model: Optional[Any] = None,
    max_tokens: int = 1800,
    max_chars: int = 12000,
) -> List[BaseMessage]:
    """
    压缩历史消息，减少 token 浪费

    Args:
        messages: 原始历史消息（建议为 Human/AI 历史，不含新的 system 指令）
        model: 可选，LangChain 聊天模型实例。提供后可参与 token 计数
        max_tokens: 目标 token 上限，默认 1800
        max_chars: 回退压缩时的字符预算上限，默认 12000

    Returns:
        压缩后的消息列表
    """
    if not messages:
        return []

    # 只保留对话轮次，避免把外层 system 注入重复压进子图
    dialog_messages = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]
    if not dialog_messages:
        return []

    try:
        # LangChain 原生压缩（1.2.x + langchain-core 1.x 可用）
        from langchain_core.messages import trim_messages  # type: ignore

        # 构造压缩参数
        trim_kwargs = {
            "messages": dialog_messages,        # 待压缩消息
            "max_tokens": int(max_tokens),       # token 上限
            "strategy": "last",                 # 保留最后 N 条
            "include_system": False,            # 不包含 system 消息
            "allow_partial": False,              # 不允许截断
            "token_counter": _build_token_counter(model),
        }
        # 执行压缩
        compressed = trim_messages(**trim_kwargs)
        if isinstance(compressed, list) and compressed:
            return [m for m in compressed if isinstance(m, BaseMessage)]
    except Exception as exc:
        log.warning(f"历史压缩 trim_messages 不可用或执行失败，启用字符预算回退: {exc}")

    # 回退到字符预算压缩
    return _fallback_trim_by_chars(dialog_messages, max_chars=max_chars)
