# -*- coding: utf-8 -*-
"""
历史消息压缩工具。

策略：
1. 优先使用 LangChain 原生 `trim_messages` 做窗口裁剪（兼容 token 计数器）。
2. 若运行环境或参数不支持，则回退到本地字符预算压缩。
"""
from __future__ import annotations

from typing import Any, Iterable, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from common.utils.custom_logger import get_logger

log = get_logger(__name__)

HISTORY_SUMMARY_MARKER = "【历史摘要记忆】"
HISTORY_SUMMARY_TRIGGER_ROUNDS = 10
HISTORY_SUMMARY_BATCH_ROUNDS = 8
HISTORY_SUMMARY_KEEP_RECENT_ROUNDS = 2
HISTORY_SUMMARY_MAX_SOURCE_CHARS = 12000


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


def _is_summary_system_message(message: BaseMessage) -> bool:
    return isinstance(message, SystemMessage) and HISTORY_SUMMARY_MARKER in _message_text(message)


def _extract_summary_text(message: BaseMessage) -> str:
    text = _message_text(message).strip()
    return text.replace(HISTORY_SUMMARY_MARKER, "", 1).strip()


def split_dialog_rounds(messages: Iterable[BaseMessage]) -> List[List[BaseMessage]]:
    """按用户轮次切分对话历史。"""
    rounds: List[List[BaseMessage]] = []
    current_round: List[BaseMessage] = []

    dialog_messages = [
        message
        for message in list(messages)
        if isinstance(message, (HumanMessage, AIMessage))
    ]

    for message in dialog_messages:
        if isinstance(message, HumanMessage):
            if current_round:
                rounds.append(current_round)
            current_round = [message]
            continue
        if not current_round:
            current_round = [message]
        else:
            current_round.append(message)

    if current_round:
        rounds.append(current_round)
    return rounds


def count_dialog_rounds(messages: Iterable[BaseMessage]) -> int:
    """统计对话轮数，只计算 Human/AI 对话，不计算 system 注入。"""
    return len(split_dialog_rounds(messages))


def _format_rounds_for_summary(rounds: List[List[BaseMessage]], *, max_chars: int) -> str:
    sections: List[str] = []
    for index, round_messages in enumerate(rounds, start=1):
        lines = [f"第{index}轮"]
        for message in round_messages:
            if isinstance(message, HumanMessage):
                role = "用户"
            elif isinstance(message, AIMessage):
                role = "助手"
            else:
                role = "消息"
            text = _message_text(message).strip()
            if text:
                lines.append(f"{role}：{text}")
        block = "\n".join(lines).strip()
        if block:
            sections.append(block)

    joined = "\n\n---\n\n".join(sections).strip()
    if len(joined) <= max_chars:
        return joined
    return joined[:max_chars].rstrip() + "\n\n[内容过长，后续已截断]"


def _build_fallback_summary(existing_summary: str, rounds: List[List[BaseMessage]]) -> str:
    """LLM 摘要失败时的确定性降级摘要。"""
    history_text = _format_rounds_for_summary(
        rounds,
        max_chars=min(3000, HISTORY_SUMMARY_MAX_SOURCE_CHARS),
    )
    fragments: List[str] = []
    if existing_summary:
        fragments.append(existing_summary.strip())
    if history_text:
        fragments.append(history_text)
    merged = "\n\n".join(item for item in fragments if item).strip()
    if not merged:
        return ""
    if len(merged) > 3500:
        return merged[:3500].rstrip() + "\n\n[摘要降级截断]"
    return merged


async def build_sliding_summary_messages(
    messages: List[BaseMessage],
    *,
    model: Optional[BaseChatModel] = None,
    config: Optional[RunnableConfig] = None,
    trigger_rounds: int = HISTORY_SUMMARY_TRIGGER_ROUNDS,
    summarize_rounds: int = HISTORY_SUMMARY_BATCH_ROUNDS,
    keep_recent_rounds: int = HISTORY_SUMMARY_KEEP_RECENT_ROUNDS,
) -> List[BaseMessage]:
    """
    将较早多轮对话折叠为一条全局摘要 SystemMessage，只保留最近少量原始轮次。

    设计目标：
    1. 当原始对话轮次超过阈值时，自动执行滑动摘要；
    2. 新摘要会融合已有摘要，形成长期记忆；
    3. 始终保留最近 `keep_recent_rounds` 轮原始对话作为短时记忆。
    """
    if not messages:
        return []

    dialog_rounds = split_dialog_rounds(messages)
    if len(dialog_rounds) <= trigger_rounds:
        return list(messages)

    rounds_to_keep = max(1, int(keep_recent_rounds))
    rounds_to_summarize = dialog_rounds[:-rounds_to_keep]
    recent_rounds = dialog_rounds[-rounds_to_keep:]
    if len(rounds_to_summarize) < max(1, int(summarize_rounds)):
        return list(messages)

    existing_summary = "\n\n".join(
        _extract_summary_text(message)
        for message in messages
        if _is_summary_system_message(message)
    ).strip()
    history_text = _format_rounds_for_summary(
        rounds_to_summarize,
        max_chars=HISTORY_SUMMARY_MAX_SOURCE_CHARS,
    )
    if not history_text:
        return list(messages)

    summary_text = ""
    if model is not None:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是会话长期记忆整理器。你的任务是把较早多轮对话压缩为高信息密度的中文摘要，"
                    "供后续轮次作为全局 SystemMessage 使用。"
                    "\n要求："
                    "\n1. 只保留对后续推理真正有用的信息。"
                    "\n2. 明确写出用户事实、长期偏好、已确认约束、关键结论、未完成事项。"
                    "\n3. 不要编造，不要保留寒暄废话，不要输出 Markdown 标题。"
                    "\n4. 输出应可直接作为系统记忆注入。"
                ),
                (
                    "human",
                    "已有长期摘要：\n{existing_summary}\n\n"
                    "请把以下较早历史轮次与已有摘要融合，生成新的长期记忆摘要：\n{history_text}",
                ),
            ]
        )
        try:
            response = await (prompt | model).ainvoke(
                {
                    "existing_summary": existing_summary or "（无）",
                    "history_text": history_text,
                },
                config=config,
            )
            summary_text = _message_text(response).strip()
        except Exception as exc:
            log.warning(f"滑动摘要生成失败，启用确定性降级摘要: {exc}")

    if not summary_text:
        summary_text = _build_fallback_summary(existing_summary, rounds_to_summarize)
    if not summary_text:
        return list(messages)

    summary_message = SystemMessage(content=f"{HISTORY_SUMMARY_MARKER}\n{summary_text}")
    compact_messages: List[BaseMessage] = [summary_message]
    for round_messages in recent_rounds:
        compact_messages.extend(round_messages)
    return compact_messages


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
