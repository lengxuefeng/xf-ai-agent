# -*- coding: utf-8 -*-
import json
import time
from typing import Generator, Optional, Dict, Any

from starlette.responses import StreamingResponse

from agent.graph_runner import GraphRunner
from db import get_db
from schemas.chat_history_schemas import ChatMessageCreate
from schemas.chat_schemas import StreamChatRequest
from services.chat_history_service import chat_history_service
from services.exception_service import exception_handler
from services.user_model_service import user_model_service
from utils.chat_utils import ChatUtils
from db.mysql.model_setting_db import model_setting_db
from utils.custom_logger import get_logger, LogTarget

log = get_logger(__name__)


class ChatService:
    """聊天服务层，处理聊天逻辑和历史记录保存"""

    def __init__(self):
        self.graph_runner = GraphRunner()

    def process_stream_chat(self, req: StreamChatRequest, user_id: Optional[int] = None) -> StreamingResponse:
        """
        统一处理流式聊天请求，包含所有业务逻辑"""
        log.info(f"收到流式聊天请求，会话ID: {req.session_id}, 用户ID: {user_id}", target=LogTarget.LOG)
        return exception_handler.safe_execute(
            self._process_stream_chat_internal,
            req, user_id,
            context="流式聊天处理"
        )

    def _process_stream_chat_internal(self, req: StreamChatRequest, user_id: Optional[int] = None) -> StreamingResponse:
        """内部流式聊天处理逻辑"""
        log.info(f"请求参数: model={req.model}, user_model_id={req.user_model_id}, service_type={req.service_type}",
                 target=LogTarget.LOG)
        model_config = None
        if req.user_model_id:
            log.info(f"使用用户模型配置: {req.user_model_id}", target=LogTarget.LOG)
            model_config = self._build_model_config_from_user_model(req.user_model_id, user_id)

        if not model_config:
            log.info("使用请求参数构建配置", target=LogTarget.LOG)
            model_config = self._build_model_config_from_request(req)

        # 根据是否有用户ID选择处理方式
        history_messages = []
        if user_id:
            history_data = chat_history_service.get_session_messages(user_id, req.session_id)
            history_messages = history_data.get("messages", [])
            stream_generator = self.stream_chat_with_history(
                user_input=req.user_input,
                session_id=req.session_id,
                model_config=model_config,
                user_id=user_id,
                history_messages=history_messages
            )
        else:
            stream_generator = self.stream_chat_anonymous(
                user_input=req.user_input,
                session_id=req.session_id,
                model_config=model_config,
                history_messages=history_messages
            )

        # 返回流式响应
        return StreamingResponse(
            stream_generator,
            media_type="text/event-stream",
            headers=self._get_stream_headers()
        )

    def _build_model_config_from_user_model(self, user_model_id: int, user_id: Optional[int] = None) -> Optional[
        Dict[str, Any]]:
        """根据用户模型ID构建模型配置"""
        db = next(get_db())
        try:
            user_model = user_model_service.get_user_model_by_id(db, user_model_id, user_id)
            if not user_model:
                return None

            model_setting = user_model.model_setting
            if not model_setting:
                return None

            # 从 custom_config 中读取用户自定义配置
            custom_config = user_model.custom_config or {}

            return {
                'model': user_model.selected_model,
                'model_service': model_setting.service_name,
                'service_type': model_setting.service_type,
                'deep_thinking_mode': custom_config.get('deep_thinking_mode', 'auto'),
                'rag_enabled': custom_config.get('rag_enabled', False),
                'similarity_threshold': custom_config.get('similarity_threshold', 0.7),
                'embedding_model': custom_config.get('embedding_model', 'bge-m3:latest'),
                'embedding_model_key': custom_config.get('embedding_model_key', ''),
                'model_key': user_model.api_key,
                'model_url': user_model.api_url or model_setting.service_url
            }
        finally:
            db.close()

    def _build_model_config_from_request(self, req: StreamChatRequest) -> Dict[str, Any]:
        """从请求中构建模型配置"""
        return self.build_model_config(
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

    def _process_stream_chat_internal(self, req: StreamChatRequest, user_id: Optional[int] = None) -> StreamingResponse:
        """内部流式聊天处理逻辑"""
        # 1. 优先尝试根据 user_model_id 查询数据库配置
        model_config = None
        if req.user_model_id:
            log.info(f"使用用户模型配置: {req.user_model_id}", target=LogTarget.LOG)
            model_config = self._build_model_config_from_user_model(req.user_model_id, user_id)

        # 2. 如果没查到（或是匿名用户），则使用请求参数构建默认配置
        if not model_config:
            if req.user_model_id:
                log.warning(f"用户模型配置 {req.user_model_id} 查询失败，回退到默认配置", target=LogTarget.LOG)
            else:
                log.info("未指定用户模型ID，使用请求参数或默认配置", target=LogTarget.LOG)

            # 智能推断 service_type (仅对默认/匿名情况生效)
            service_type = req.service_type
            if not service_type or service_type == 'ollama':
                if 'gemini' in req.model.lower():
                    service_type = 'gemini'
                elif 'gpt' in req.model.lower():
                    service_type = 'openai'
                elif 'deepseek' in req.model.lower():
                    service_type = 'silicon-flow'

            model_config = self.build_model_config(
                model=req.model,
                model_service=req.model_service,
                service_type=service_type,
                deep_thinking_mode=req.deep_thinking_mode,
                rag_enabled=req.rag_enabled,
                similarity_threshold=req.similarity_threshold,
                embedding_model=req.embedding_model,
                embedding_model_key=req.embedding_model_key,
                model_key=req.model_key,
                model_url=req.model_url,
            )

        # 根据是否有用户ID选择处理方式
        history_messages = []
        if user_id:
            log.info(f"用户已登录，加载历史消息，会话ID: {req.session_id}", target=LogTarget.LOG)
            history_data = chat_history_service.get_session_messages(user_id, req.session_id)
            history_messages = history_data.get("messages", [])
            stream_generator = self.stream_chat_with_history(
                user_input=req.user_input,
                session_id=req.session_id,
                model_config=model_config,
                user_id=user_id,
                history_messages=history_messages
            )
        else:
            log.info("匿名用户，不加载历史消息", target=LogTarget.LOG)
            stream_generator = self.stream_chat_anonymous(
                user_input=req.user_input,
                session_id=req.session_id,
                model_config=model_config,
                history_messages=history_messages
            )

        # 返回流式响应
        return StreamingResponse(
            stream_generator,
            media_type="text/event-stream",
            headers=self._get_stream_headers()
        )

    def _get_stream_headers(self) -> Dict[str, str]:
        """获取流式响应头"""
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
            user_id: Optional[int] = None,
            history_messages: list = []
    ) -> Generator[str, None, None]:
        """带历史记录保存的流式聊天"""
        log.info(f"开始带历史的流式聊天，会话ID: {session_id}", target=LogTarget.LOG)

        # 获取或创建会话
        chat_history_service.get_or_create_session(user_id, session_id, user_input)

        ai_response = ""
        start_time = time.time()
        error_occurred = False
        error_message = ""

        try:
            # 生成流式响应并收集AI回复内容
            for chunk in self.graph_runner.stream_run(user_input, session_id, model_config, history_messages):
                yield chunk
                # 提取AI响应内容
                ai_content = self._extract_ai_content_from_chunk(chunk)
                if ai_content:
                    ai_response += ai_content

        except RuntimeError as e:
            # 模型加载失败的特殊处理
            error_occurred = True
            error_message = str(e)
            log.error(f"模型加载失败: {error_message}", target=LogTarget.ALL)
            # 发送错误消息给前端
            yield f"data: {json.dumps({'type': 'error', 'content': f'模型加载失败: {error_message}'}, ensure_ascii=False)}\n\n"
            return
        except Exception as e:
            error_occurred = True
            error_message = str(e)
            log.exception(f"聊天处理异常: {error_message}", target=LogTarget.ALL)
            # 重新抛出异常，让上层处理
            raise e
        finally:
            # 如果用户已认证，保存聊天历史
            if user_id:
                log.info(f"保存聊天历史，会话ID: {session_id}", target=LogTarget.LOG)
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
            model_config: Dict[str, Any],
            history_messages: list = []
    ) -> Generator[str, None, None]:
        """匿名流式聊天，不保存历史记录"""
        log.info(f"匿名流式聊天，会话ID: {session_id}", target=LogTarget.LOG)
        # 直接返回原始流式响应，不做任何处理
        yield from self.graph_runner.stream_run(user_input, session_id, model_config, history_messages)

    def _extract_ai_content_from_chunk(self, chunk: str) -> Optional[str]:
        """从SSE chunk中提取AI响应内容"""
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
        """构建模型配置字典"""
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
