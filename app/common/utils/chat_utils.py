# -*- coding: utf-8 -*-
"""
聊天工具类（Chat Utils）。

提供聊天相关的工具方法，包括ID生成、SSE格式化、校验、token估算等功能。
这些工具方法被多个模块共享使用，是聊天服务的基础工具库。

设计要点：
1. ID生成：生成唯一的会话ID
2. SSE格式化：统一SSE事件格式
3. 数据解析：解析SSE事件数据
4. 校验和清洗：输入数据校验和清理
5. Token估算：粗略估算文本的token数量

使用场景：
- 生成会话ID
- 格式化和解析SSE事件
- 校验用户输入
- 估算token数量
- 生成错误和状态响应
"""
import json
import uuid
import time
import re
from typing import Optional, Dict, Any
from datetime import datetime

from config.constants.sse_constants import SseEventType, SsePayloadField

# 预编译正则，提升高频调用时的验证性能
SESSION_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')


class ChatUtils:
    """
    聊天工具类：提供ID生成、SSE格式化、校验等功能。

    核心职责：
    1. 生成唯一的会话ID
    2. 格式化和解析SSE事件
    3. 校验和清洗输入数据
    4. 估算文本token数量
    5. 生成错误和状态响应

    设计理由：
    1. 集中管理工具方法，避免重复代码
    2. 提供统一的接口，便于使用
    3. 优化性能（预编译正则）
    4. 完善的参数校验

    使用场景：
    - 聊天服务：生成会话ID、格式化SSE
    - API接口：校验输入参数
    - 日志和监控：计算响应指标
    """

    @staticmethod
    def generate_session_id(prefix: str = "chat") -> str:
        """
        生成唯一的会话ID。

        设计要点：
        1. 使用时间戳保证时序性
        2. 使用UUID保证唯一性
        3. 使用前缀便于识别和分类

        Args:
            prefix: ID前缀，默认为"chat"

        Returns:
            str: 格式为"prefix_timestamp_uuid8"的会话ID

        示例：
        >>> ChatUtils.generate_session_id()
        "chat_1234567890_a1b2c3d4"
        >>> ChatUtils.generate_session_id("test")
        "test_1234567890_a1b2c3d4"
        """
        timestamp = int(datetime.now().timestamp() * 1000)
        random_id = uuid.uuid4().hex[:8]
        return f"{prefix}_{timestamp}_{random_id}"

    @staticmethod
    def format_sse_data(event_type: str, data: Dict[str, Any]) -> str:
        """
        格式化SSE数据。

        设计要点：
        1. 遵循SSE标准格式
        2. 确保data内部也有type字段（双重兼容）
        3. 使用UTF-8编码，支持中文

        Args:
            event_type: 事件类型（如"stream"、"thinking"）
            data: 事件数据字典

        Returns:
            str: 格式化后的SSE字符串

        格式示例：
        event: stream
        data: {"type": "stream", "content": "你好"}

        设计理由：
        1. 标准SSE格式，兼容浏览器
        2. 双重type字段，便于解析
        3. JSON序列化，支持复杂结构
        """
        # 确保 data 内部也带有 type，保持双重兼容
        if SsePayloadField.TYPE.value not in data:
            data[SsePayloadField.TYPE.value] = event_type

        payload = json.dumps(data, ensure_ascii=False)
        return f"event: {event_type}\ndata: {payload}\n\n"

    @staticmethod
    def parse_sse_data(chunk: str) -> Optional[Dict[str, Any]]:
        """
        解析SSE数据块。

        设计要点：
        1. 支持标准SSE格式
        2. 提取data行并解析JSON
        3. 容错处理，解析失败返回None

        Args:
            chunk: SSE事件chunk字符串

        Returns:
            Optional[Dict[str, Any]]: 解析后的数据字典，失败则返回None

        解析流程：
        1. 按行分割chunk
        2. 查找data:开头的行
        3. 去掉"data: "前缀
        4. 解析JSON并返回
        """
        try:
            # 兼容带有 event: 行的新格式
            lines = chunk.strip().split('\n')
            data_line = next((line for line in lines if line.startswith('data: ')), None)

            if data_line:
                return json.loads(data_line[6:])
        except (json.JSONDecodeError, IndexError, ValueError):
            pass  # 解析失败，返回None
        return None

    @staticmethod
    def extract_content_by_type(chunk: str, content_type: str) -> Optional[str]:
        """
        从SSE chunk中提取指定类型的内容。

        设计要点：
        1. 解析SSE数据
        2. 检查类型是否匹配
        3. 提取content字段

        Args:
            chunk: SSE事件chunk字符串
            content_type: 要提取的内容类型（如"stream"）

        Returns:
            Optional[str]: 提取的内容，失败则返回None

        使用场景：
        - 提取流式内容
        - 过滤特定类型的事件
        """
        data = ChatUtils.parse_sse_data(chunk)
        if data and data.get(SsePayloadField.TYPE.value) == content_type:
            return data.get(SsePayloadField.CONTENT.value, '')
        return None

    @staticmethod
    def validate_session_id(session_id: str) -> bool:
        """
        验证会话ID格式。

        设计要点：
        1. 长度限制：5-100个字符
        2. 字符限制：只允许字母、数字、下划线、短横线
        3. 预编译正则，提升性能

        Args:
            session_id: 要验证的会话ID

        Returns:
            bool: True表示格式正确，False表示格式错误

        使用场景：
        - API接口参数校验
        - 防止恶意输入
        """
        if not session_id or not isinstance(session_id, str):
            return False
        if len(session_id) < 5 or len(session_id) > 100:
            return False
        return bool(SESSION_ID_PATTERN.match(session_id))

    @staticmethod
    def sanitize_user_input(user_input: str, max_length: int = 5000) -> str:
        """
        清理并截断用户输入。

        设计要点：
        1. 去除前后空格
        2. 限制最大长度
        3. 避免空值

        Args:
            user_input: 原始用户输入
            max_length: 最大长度，默认5000

        Returns:
            str: 清理后的用户输入

        使用场景：
        - API接口参数清洗
        - 防止过长的输入
        - 避免空值导致问题
        """
        if not user_input:
            return ""
        return user_input.strip()[:max_length]

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        估算文本的token数量。

        设计要点：
        1. 中文：1个字约1个token
        2. 英文：1个单词约1个token
        3. 其他：标点符号等折半计算

        注意：
        这是粗略估算，不同模型的实际token数可能有差异。
        精确计算需要使用模型的tokenizer。

        Args:
            text: 要估算的文本

        Returns:
            int: 估算的token数量

        使用场景：
        - 限制输入长度
        - 计算token成本
        - 优化上下文管理
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
        """
        生成错误SSE响应。

        设计要点：
        1. 使用ERROR事件类型
        2. 包含错误类型和消息
        3. 遵循SSE格式

        Args:
            error_type: 错误类型
            message: 错误消息

        Returns:
            str: 格式化后的SSE错误响应

        使用场景：
        - API错误响应
        - 执行失败通知
        """
        return ChatUtils.format_sse_data(
            event_type=SseEventType.ERROR.value,
            data={"error_type": error_type, SsePayloadField.CONTENT.value: message}
        )

    @staticmethod
    def create_status_response(status: str, message: str) -> str:
        """
        生成状态提示SSE响应。

        设计要点：
        1. 使用指定的状态事件类型
        2. 包含状态消息
        3. 遵循SSE格式

        Args:
            status: 状态类型（如"thinking"、"processing"）
            message: 状态消息

        Returns:
            str: 格式化后的SSE状态响应

        使用场景：
        - 处理中提示
        - 思考过程展示
        - 系统状态通知
        """
        return ChatUtils.format_sse_data(
            event_type=status,
            data={SsePayloadField.CONTENT.value: message}
        )

    @staticmethod
    def calculate_response_metrics(start_time: float, content_length: int) -> Dict[str, Any]:
        """
        计算响应指标。

        设计要点：
        1. 计算耗时（毫秒）
        2. 计算字符生成速度
        3. 记录内容长度

        Args:
            start_time: 开始时间（time.time()返回的时间戳）
            content_length: 内容长度

        Returns:
            Dict[str, Any]: 响应指标字典
                - latency_ms: 耗时（毫秒）
                - duration_seconds: 耗时（秒）
                - content_length: 内容长度
                - chars_per_second: 每秒生成字符数

        使用场景：
        - 性能监控
        - 日志记录
        - 质量分析
        """
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
