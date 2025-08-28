# -*- coding: utf-8 -*-
import time
import json
from typing import Generator, Optional, Dict, Any
from datetime import datetime

from fastapi import HTTPException
from starlette.responses import StreamingResponse

from agent.graph_runner import GraphRunner
from services.chat_history_service import chat_history_service
from services.exception_service import exception_handler
from schemas.chat_history_schemas import ChatHistoryCreate
from schemas.chat_schemas import StreamChatRequest


class ChatService:
    """
    聊天服务层，处理聊天逻辑和历史记录保存
    """

    def __init__(self):
        self.graph_runner = GraphRunner()

    def process_stream_chat(self, req: StreamChatRequest, user_id: Optional[int] = None) -> StreamingResponse:
        """
        统一处理流式聊天请求，包含所有业务逻辑
        
        Args:
            req: 聊天请求对象
            user_id: 用户ID，可选
            
        Returns:
            StreamingResponse: 流式响应对象
            
        Raises:
            HTTPException: 当发生业务异常时
        """
        return exception_handler.safe_execute(
            self._process_stream_chat_internal,
            req, user_id,
            context="流式聊天处理"
        )

    def _process_stream_chat_internal(self, req: StreamChatRequest, user_id: Optional[int] = None) -> StreamingResponse:
        """
        内部流式聊天处理逻辑
        
        Args:
            req: 聊天请求对象
            user_id: 用户ID
            
        Returns:
            StreamingResponse: 流式响应对象
        """
        # 构建模型配置
        model_config = self.build_model_config(
            model=req.model,
            model_service=req.model_service,
            deep_thinking_mode=req.deep_thinking_mode,
            rag_enabled=req.rag_enabled,
            similarity_threshold=req.similarity_threshold,
            embedding_model=req.embedding_model
        )

        # 根据是否有用户ID选择处理方式
        if user_id:
            stream_generator = self.stream_chat_with_history(
                user_input=req.user_input,
                session_id=req.session_id,
                model_config=model_config,
                user_id=user_id
            )
        else:
            stream_generator = self.stream_chat_anonymous(
                user_input=req.user_input,
                session_id=req.session_id,
                model_config=model_config
            )

        # 返回流式响应
        return StreamingResponse(
            stream_generator,
            media_type="text/event-stream",
            headers=self._get_stream_headers()
        )

    def _get_stream_headers(self) -> Dict[str, str]:
        """
        获取流式响应头
        
        Returns:
            Dict[str, str]: HTTP响应头字典
        """
        return {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用nginx缓冲
        }

    def stream_chat_with_history(
            self,
            user_input: str,
            session_id: str,
            model_config: Dict[str, Any],
            user_id: Optional[int] = None
    ) -> Generator[str, None, None]:
        """
        带历史记录保存的流式聊天
        
        Args:
            user_input: 用户输入
            session_id: 会话ID
            model_config: 模型配置
            user_id: 用户ID，如果为None则不保存历史记录
            
        Yields:
            str: SSE格式的流式响应
        """
        ai_response = ""
        start_time = time.time()
        error_occurred = False
        error_message = ""

        try:
            # 生成流式响应并收集AI回复内容
            for chunk in self.graph_runner.stream_run(user_input, session_id, model_config):
                yield chunk

                # 提取AI响应内容
                ai_content = self._extract_ai_content_from_chunk(chunk)
                if ai_content:
                    ai_response += ai_content

        except Exception as e:
            error_occurred = True
            error_message = str(e)
            # 重新抛出异常，让上层处理
            raise e
        finally:
            # 如果用户已认证，保存聊天历史
            if user_id:
                self._save_chat_history(
                    user_id=user_id,
                    user_input=user_input,
                    session_id=session_id,
                    ai_response=ai_response if not error_occurred else f"错误: {error_message}",
                    model_config=model_config,
                    start_time=start_time,
                    error_occurred=error_occurred
                )

    def stream_chat_anonymous(
            self,
            user_input: str,
            session_id: str,
            model_config: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """
        匿名流式聊天，不保存历史记录
        
        Args:
            user_input: 用户输入
            session_id: 会话ID
            model_config: 模型配置
            
        Yields:
            str: SSE格式的流式响应
        """
        # 直接返回原始流式响应，不做任何处理
        yield from self.graph_runner.stream_run(user_input, session_id, model_config)

    def _extract_ai_content_from_chunk(self, chunk: str) -> Optional[str]:
        """
        从SSE chunk中提取AI响应内容
        
        Args:
            chunk: SSE格式的数据块
            
        Returns:
            str: 提取的内容，如果不是stream类型则返回None
        """
        try:
            if chunk.startswith('data: '):
                data = json.loads(chunk[6:])
                if data.get('type') == 'stream':
                    return data.get('content', '')
        except (json.JSONDecodeError, KeyError):
            pass
        return None

    def _save_chat_history(
            self,
            user_id: int,
            user_input: str,
            session_id: str,
            ai_response: str,
            model_config: Dict[str, Any],
            start_time: float,
            error_occurred: bool = False
    ) -> None:
        """
        保存聊天历史记录
        
        Args:
            user_id: 用户ID
            user_input: 用户输入
            session_id: 会话ID
            ai_response: AI响应
            model_config: 模型配置
            start_time: 开始时间
            error_occurred: 是否发生错误
        """
        try:
            # 计算响应延迟
            latency_ms = int((time.time() - start_time) * 1000)

            # 生成会话标题（取用户输入的前50个字符）
            title = self._generate_chat_title(user_input)

            # 创建聊天历史记录
            chat_data = ChatHistoryCreate(
                user_id=user_id,
                title=title,
                session_id=session_id,
                user_content=user_input,
                model_content=ai_response,
                model=model_config.get('model'),
                latency_ms=latency_ms,
                tokens=self._estimate_tokens(user_input, ai_response)
            )

            # 保存到数据库
            chat_history_service.create_chat(user_id, chat_data)

        except Exception as e:
            # 记录错误但不影响用户体验
            print(f"保存聊天历史失败: {e}")

    def _generate_chat_title(self, user_input: str) -> str:
        """
        生成聊天标题
        
        Args:
            user_input: 用户输入
            
        Returns:
            str: 生成的标题
        """
        if len(user_input) > 50:
            return user_input[:50] + "..."
        return user_input

    def _estimate_tokens(self, user_input: str, ai_response: str) -> int:
        """
        估算token数量（简单估算）
        
        Args:
            user_input: 用户输入
            ai_response: AI响应
            
        Returns:
            int: 估算的token数量
        """
        # 简单估算：中文字符约等于1个token，英文单词约等于1个token
        total_chars = len(user_input) + len(ai_response)
        # 粗略估算
        return int(total_chars * 0.7)

    def build_model_config(
            self,
            model: str = 'google/gemini-1.5-pro',
            model_service: str = 'netlify-gemini',
            deep_thinking_mode: str = 'auto',
            rag_enabled: bool = False,
            similarity_threshold: float = 0.7,
            embedding_model: str = 'bge-m3:latest'
    ) -> Dict[str, Any]:
        """
        构建模型配置字典
        
        Args:
            model: 模型名称
            model_service: 模型服务
            deep_thinking_mode: 深度思考模式
            rag_enabled: 是否启用RAG
            similarity_threshold: 相似度阈值
            embedding_model: 嵌入模型
            
        Returns:
            Dict: 模型配置字典
        """
        return {
            'model': model,
            'model_service': model_service,
            'deep_thinking_mode': deep_thinking_mode,
            'rag_enabled': rag_enabled,
            'similarity_threshold': similarity_threshold,
            'embedding_model': embedding_model
        }


# 创建全局实例
chat_service = ChatService()
