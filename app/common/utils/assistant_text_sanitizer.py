import re
from typing import Any

_INLINE_TOOL_INVOCATION_PATTERNS = (
    re.compile(
        r"(?:web_search_proxy|tavily_search_tool|get_weathers?|execute_sql|federated_query_gateway)"
        r"\s*(?:\{[\s\S]*?\}|\([\s\S]*?\))",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\{[^{}\n]*[\"']query[\"']\s*:\s*[\"'][^\"']+[\"'][^{}\n]*"
        r"[\"']source[\"']\s*:\s*[\"']external_(?:search|weather)[\"'][^{}\n]*\}",
        flags=re.IGNORECASE,
    ),
)

_LINE_TOOL_NOISE_PATTERNS = (
    re.compile(r"(^|\n)\s*🛠\s*调用工具[:：][^\n]*", flags=re.IGNORECASE),
    re.compile(r"(^|\n)\s*正在调用工具[^\n]*", flags=re.IGNORECASE),
    re.compile(r"(^|\n)\s*tool_(?:call|result)\s*[:：][^\n]*", flags=re.IGNORECASE),
)


def _preserve_leading_newline(match: re.Match[str]) -> str:
    return "\n" if match.group(0).startswith("\n") else ""


def strip_internal_execution_noise(
    content: Any,
    *,
    trim: bool = True,
    collapse_blank_lines: bool = True,
) -> str:
    """移除不应暴露给最终用户的内部工具调用/执行噪音。"""
    text = str(content or "")
    if not text:
        return ""

    sanitized = text.replace("\u200b", "")
    for pattern in _INLINE_TOOL_INVOCATION_PATTERNS:
        sanitized = pattern.sub("", sanitized)
    for pattern in _LINE_TOOL_NOISE_PATTERNS:
        sanitized = pattern.sub(_preserve_leading_newline, sanitized)

    sanitized = re.sub(r"[ \t]{2,}", " ", sanitized)
    sanitized = re.sub(r"\n[ \t]+\n", "\n\n", sanitized)
    if collapse_blank_lines:
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)

    return sanitized.strip() if trim else sanitized
