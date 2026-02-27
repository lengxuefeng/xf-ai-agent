# -*- coding: utf-8 -*-
import json
import uuid
import time
import re
from typing import Optional, Dict, Any
from datetime import datetime

# 预编译正则，提升高频调用时的验证性能
SESSION_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')


class ChatUtils:
    """
    【学习笔记】聊天相关的通用工具函数
    包含：ID生成、SSE流格式化、基础安全校验、Token 粗略估算等。
    """

    @staticmethod
    def generate_session_id(prefix: str = "chat") -> str:
        """生成唯一的会话ID"""
        timestamp = int(datetime.now().timestamp() * 1000)
        random_id = uuid.uuid4().hex[:8]
        return f"{prefix}_{timestamp}_{random_id}"

    @staticmethod
    def format_sse_data(event_type: str, data: Dict[str, Any]) -> str:
        """
        【核心修复】标准 SSE 流格式化
        必须包含 event 字段，否则前端 EventSource 无法通过 addEventListener 精准监听。
        """
        # 确保 data 内部也带有 type，保持双重兼容
        if "type" not in data:
            data["type"] = event_type

        payload = json.dumps(data, ensure_ascii=False)
        return f"event: {event_type}\ndata: {payload}\n\n"

    @staticmethod
    def parse_sse_data(chunk: str) -> Optional[Dict[str, Any]]:
        """解析 SSE 数据块"""
        try:
            # 兼容带有 event: 行的新格式
            lines = chunk.strip().split('\n')
            data_line = next((line for line in lines if line.startswith('data: ')), None)

            if data_line:
                return json.loads(data_line[6:])
        except (json.JSONDecodeError, IndexError, ValueError):
            pass
        return None

    @staticmethod
    def extract_content_by_type(chunk: str, content_type: str) -> Optional[str]:
        """从SSE chunk中提取指定类型的内容"""
        data = ChatUtils.parse_sse_data(chunk)
        if data and data.get('type') == content_type:
            return data.get('content', '')
        return None

    @staticmethod
    def validate_session_id(session_id: str) -> bool:
        """验证会话ID格式 (长度5-100，仅允许字母、数字、下划线、中划线)"""
        if not session_id or not isinstance(session_id, str):
            return False
        if len(session_id) < 5 or len(session_id) > 100:
            return False
        return bool(SESSION_ID_PATTERN.match(session_id))

    @staticmethod
    def sanitize_user_input(user_input: str, max_length: int = 5000) -> str:
        """清理并截断用户输入，防止超大文本恶意注入"""
        if not user_input:
            return ""
        return user_input.strip()[:max_length]

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        估算文本的token数量 (轻量级本地估算)
        注意：精确计算需使用大模型厂商对应的 tokenizer (如 tiktoken)
        """
        if not text:
            return 0

        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        english_words = len([word for word in text.split() if word.isalpha()])
        other_chars = len(text) - chinese_chars

        # 中文1个字约1个Token，英文1个单词约1个Token，其他标点符号折半
        return chinese_chars + english_words + int(other_chars * 0.5)

    @staticmethod
    def create_error_response(error_type: str, message: str) -> str:
        """生成标准错误 SSE 响应"""
        return ChatUtils.format_sse_data(
            event_type="error",
            data={"error_type": error_type, "content": message}
        )

    @staticmethod
    def create_status_response(status: str, message: str) -> str:
        """生成状态提示 SSE 响应 (如 thinking, processing)"""
        return ChatUtils.format_sse_data(
            event_type=status,
            data={"content": message}
        )

    @staticmethod
    def calculate_response_metrics(start_time: float, content_length: int) -> Dict[str, Any]:
        """计算响应耗时与吞吐量指标"""
        end_time = time.time()
        duration_seconds = end_time - start_time
        latency_ms = int(duration_seconds * 1000)

        chars_per_second = int(content_length / duration_seconds) if duration_seconds > 0 else 0

        return {
            "latency_ms": latency_ms,
            "duration_seconds": round(duration_seconds, 2),
            "content_length": content_length,
            "chars_per_second": chars_per_second
        }