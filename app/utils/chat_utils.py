# -*- coding: utf-8 -*-
import json
import uuid
from typing import Optional, Dict, Any
from datetime import datetime


class ChatUtils:
    """
    聊天相关的通用工具函数
    """
    
    @staticmethod
    def generate_session_id(prefix: str = "chat") -> str:
        """
        生成唯一的会话ID
        
        Args:
            prefix: ID前缀
            
        Returns:
            str: 唯一的会话ID
        """
        timestamp = int(datetime.now().timestamp() * 1000)
        random_id = uuid.uuid4().hex[:8]
        return f"{prefix}_{timestamp}_{random_id}"
    
    @staticmethod
    def parse_sse_data(chunk: str) -> Optional[Dict[str, Any]]:
        """
        解析SSE数据格式
        
        Args:
            chunk: SSE数据块
            
        Returns:
            dict: 解析后的数据，如果解析失败返回None
        """
        try:
            if chunk.startswith('data: '):
                return json.loads(chunk[6:])
        except (json.JSONDecodeError, IndexError):
            pass
        return None
    
    @staticmethod
    def format_sse_data(data: Dict[str, Any]) -> str:
        """
        格式化数据为SSE格式
        
        Args:
            data: 要格式化的数据
            
        Returns:
            str: SSE格式的字符串
        """
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    
    @staticmethod
    def extract_content_by_type(chunk: str, content_type: str) -> Optional[str]:
        """
        从SSE chunk中提取指定类型的内容
        
        Args:
            chunk: SSE数据块
            content_type: 内容类型（如 'stream', 'thinking', 'error' 等）
            
        Returns:
            str: 提取的内容，如果类型不匹配或解析失败返回None
        """
        data = ChatUtils.parse_sse_data(chunk)
        if data and data.get('type') == content_type:
            return data.get('content', '')
        return None
    
    @staticmethod
    def validate_session_id(session_id: str) -> bool:
        """
        验证会话ID格式
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 是否有效
        """
        if not session_id or not isinstance(session_id, str):
            return False
        
        # 基本长度检查
        if len(session_id) < 5 or len(session_id) > 100:
            return False
        
        # 检查是否包含有效字符
        import re
        pattern = r'^[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, session_id))
    
    @staticmethod
    def sanitize_user_input(user_input: str) -> str:
        """
        清理用户输入
        
        Args:
            user_input: 用户原始输入
            
        Returns:
            str: 清理后的输入
        """
        if not user_input:
            return ""
        
        # 去除首尾空格
        cleaned = user_input.strip()
        
        # 限制长度
        max_length = 5000
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]
        
        return cleaned
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        估算文本的token数量
        
        Args:
            text: 输入文本
            
        Returns:
            int: 估算的token数量
        """
        if not text:
            return 0
        
        # 简单估算规则：
        # - 中文字符：1字符 ≈ 1token
        # - 英文单词：1单词 ≈ 1token
        # - 空格和标点：0.5token
        
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        english_words = len([word for word in text.split() if word.isalpha()])
        other_chars = len(text) - chinese_chars
        
        return chinese_chars + english_words + int(other_chars * 0.5)
    
    @staticmethod
    def create_error_response(error_type: str, message: str) -> str:
        """
        创建标准错误响应
        
        Args:
            error_type: 错误类型
            message: 错误消息
            
        Returns:
            str: SSE格式的错误响应
        """
        return ChatUtils.format_sse_data({
            "type": "error",
            "error_type": error_type,
            "content": message
        })
    
    @staticmethod
    def create_status_response(status: str, message: str) -> str:
        """
        创建状态响应
        
        Args:
            status: 状态类型（thinking, processing, complete等）
            message: 状态消息
            
        Returns:
            str: SSE格式的状态响应
        """
        return ChatUtils.format_sse_data({
            "type": status,
            "content": message
        })
    
    @staticmethod
    def calculate_response_metrics(start_time: float, content_length: int) -> Dict[str, Any]:
        """
        计算响应指标
        
        Args:
            start_time: 开始时间戳
            content_length: 内容长度
            
        Returns:
            dict: 包含延迟、速度等指标的字典
        """
        import time
        
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        
        # 计算处理速度（字符/秒）
        duration_seconds = end_time - start_time
        chars_per_second = int(content_length / duration_seconds) if duration_seconds > 0 else 0
        
        return {
            "latency_ms": latency_ms,
            "duration_seconds": round(duration_seconds, 2),
            "content_length": content_length,
            "chars_per_second": chars_per_second
        }