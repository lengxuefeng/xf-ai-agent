# -*- coding: utf-8 -*-
import json
import time
from typing import Generator, Optional, Dict, Any

from starlette.responses import StreamingResponse

from agent.graph_runner import GraphRunner
from schemas.chat_history_schemas import ChatMessageCreate
from schemas.chat_schemas import StreamChatRequest
from services.chat_history_service import chat_history_service
from services.exception_service import exception_handler
from utils.chat_utils import ChatUtils


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
            service_type=req.service_type,
            deep_thinking_mode=req.deep_thinking_mode,
            rag_enabled=req.rag_enabled,
            similarity_threshold=req.similarity_threshold,
            embedding_model=req.embedding_model,
            embedding_model_key=req.embedding_model_key,
            model_key=req.model_key,
            model_url=req.model_url,
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
        # 1. 获取或创建会话
        chat_history_service.get_or_create_session(user_id, session_id, user_input)

        ai_response = ""
        start_time = time.time()
        error_occurred = False
        error_message = ""

        try:
            # 2. 生成流式响应并收集AI回复内容
            for chunk in self.graph_runner.stream_run(user_input, session_id, model_config):
                yield chunk

                # 提取AI响应内容
                ai_content = self._extract_ai_content_from_chunk(chunk)
                if ai_content:
                    ai_response += ai_content

        except RuntimeError as e:
            # 模型加载失败的特殊处理
            error_occurred = True
            error_message = str(e)
            print(f"❌ 模型加载失败: {error_message}")
            # 发送错误消息给前端
            yield f"data: {json.dumps({'type': 'error', 'content': f'模型加载失败: {error_message}'}, ensure_ascii=False)}\n\n"
            return
        except Exception as e:
            error_occurred = True
            error_message = str(e)
            print(f"❌ 聊天处理异常: {error_message}")
            # 重新抛出异常，让上层处理
            raise e
        finally:
            # 3. 如果用户已认证，保存聊天历史
            if user_id:
                chat_data = ChatMessageCreate(
                    user_id=user_id,
                    session_id=session_id,
                    user_content=user_input,
                    model_content=ai_response if not error_occurred else f"错误: {error_message}",
                    model=model_config.get('model'),
                    latency_ms=int((time.time() - start_time) * 1000),
                    tokens=ChatUtils.estimate_tokens(user_input + ai_response)
                )
                chat_history_service.create_chat_message(user_id, chat_data)

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

    def build_model_config(
            self,
            model: str = 'google/gemini-1.5-pro',
            model_service: str = 'netlify-gemini',
            service_type: str = 'ollama',
            deep_thinking_mode: str = 'auto',
            rag_enabled: bool = False,
            similarity_threshold: float = 0.7,
            embedding_model: str = 'bge-m3:latest',
            embedding_model_key: str = '',
            model_key: str = '',
            model_url: str = ''
    ) -> Dict[str, Any]:
        """
        构建模型配置字典
        Args:
            model: 模型名称
            model_service: 模型服务
            service_type: 模型服务类型
            deep_thinking_mode: 深度思考模式
            rag_enabled: 是否启用RAG
            similarity_threshold: 相似度阈值
            embedding_model: 嵌入模型
            embedding_model_key: 嵌入模型密钥
            model_key: 模型密钥
            model_url: 模型URL

        Returns:
            Dict: 模型配置字典
        """
        return {
            'model': model,
            'model_service': model_service,
            'service_type': service_type,
            'deep_thinking_mode': deep_thinking_mode,
            'rag_enabled': rag_enabled,
            'similarity_threshold': similarity_threshold,
            'embedding_model': embedding_model,
            'embedding_model_key': embedding_model_key,
            'model_key': model_key,
            'model_url': model_url
        }


# 创建全局实例
chat_service = ChatService()
