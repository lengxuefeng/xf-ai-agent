# -*- coding: utf-8 -*-
import json
import time
from typing import Generator, Optional, Dict, Any

from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from agent.graph_runner import GraphRunner
from constants.chat_service_constants import (
    CHAT_AI_CONTENT_TYPES,
    CHAT_DEFAULT_MODEL_CONFIG,
    CHAT_SERVICE_ERROR_CONNECTION,
    CHAT_SERVICE_ERROR_FALLBACK_TEMPLATE,
    CHAT_SERVICE_ERROR_RUNTIME_TEMPLATE,
    CHAT_SERVICE_ERROR_TIMEOUT,
    CHAT_SERVICE_INTERRUPTED_APPEND_TEMPLATE,
    CHAT_SERVICE_INTERRUPTED_TEMPLATE,
    STREAM_HEADERS,
)
from constants.sse_constants import SseEventType, SsePayloadField
from config.runtime_settings import MODEL_TIERING_CONFIG
from db import get_db_context
from schemas.chat_history_schemas import ChatMessageCreate
from schemas.chat_schemas import StreamChatRequest
from services.chat_history_service import chat_history_service
from services.exception_service import exception_handler
from services.route_metrics_service import route_metrics_service
from services.session_state_service import session_state_service
from services.user_model_service import user_model_service
from utils.chat_utils import ChatUtils
from utils.custom_logger import get_logger, LogTarget

log = get_logger(__name__)


class ChatService:
    """聊天服务层：处理流式聊天请求、历史记录和配置管理"""

    def __init__(self):
        """初始化聊天服务"""
        self.graph_runner = GraphRunner()

    def process_stream_chat(
            self,
            req: StreamChatRequest,
            user_id: Optional[int],
            db: Optional[Session] = None,
            dynamic_model_config: Optional[Dict[str, Any]] = None
    ) -> StreamingResponse:
        """
        【入口】统一处理流式聊天请求
        """
        log.info(f"收到流式聊天请求，会话ID: {req.session_id}, 用户ID: {user_id}", target=LogTarget.LOG)
        if req.user_input and req.user_input != "[RESUME]":
            route_metrics_service.detect_and_record_correction(req.session_id, req.user_input)

        # 注意：safe_execute 只能捕获组装阶段的异常。流式输出期间的异常需要在 stream_chat_with_history 中捕获
        return exception_handler.safe_execute(
            self._process_stream_chat_internal,
            req, user_id, db, dynamic_model_config,
            context="流式聊天准备阶段"
        )

    def _process_stream_chat_internal(
            self,
            req: StreamChatRequest,
            user_id: Optional[int],
            db: Optional[Session] = None,
            dynamic_model_config: Optional[Dict[str, Any]] = None
    ) -> StreamingResponse:
        """内部流式聊天处理逻辑：准备配置、挂载生成器"""
        log.info(f"请求参数: model={req.model}, user_model_id={req.user_model_id}, service_type={req.service_type}",
                 target=LogTarget.LOG)

        # 1. 如果中间件已经解析好模型配置，则直接使用
        if dynamic_model_config:
            log.info("使用中间件预加载的动态模型配置", target=LogTarget.LOG)
            model_config = dynamic_model_config
        else:
            # 2. 兜底方案：使用前端传来的零散参数
            log.info("使用请求自带的默认参数构建配置", target=LogTarget.LOG)
            model_config = self._build_model_config_from_request(req)

        history_messages = []
        # 会话结构化上下文：包含城市/用户画像等槽位
        runtime_context: Dict[str, Any] = {}
        is_resume = req.user_input == "[RESUME]"

        # 3. 核心分流：注册用户带记忆，匿名用户无记忆
        if user_id and db:
            # 拉取历史并标准化成 GraphRunner 需要的 dict 结构
            history_data = chat_history_service.get_session_messages(db, user_id, req.session_id, page=1, size=100)
            history_messages = self._normalize_history_messages(history_data)
            # 在进入图执行前，先构建并持久化一份会话槽位上下文
            runtime_context = session_state_service.build_runtime_context(
                db=db,
                user_id=user_id,
                session_id=req.session_id,
                user_input=req.user_input,
                history_messages=history_messages,
            )
            stream_generator = self.stream_chat_with_history(
                user_input=req.user_input,
                session_id=req.session_id,
                model_config=model_config,
                user_id=user_id,
                history_messages=history_messages,
                runtime_context=runtime_context,
                is_resume=is_resume,
            )
        else:
            stream_generator = self.stream_chat_anonymous(
                user_input=req.user_input,
                session_id=req.session_id,
                model_config=model_config,
                history_messages=history_messages,
                runtime_context=runtime_context,
            )

        # 4. 返回流式响应，控制权正式交给 FastAPI 后台协程
        return StreamingResponse(
            stream_generator,
            media_type="text/event-stream",
            headers=self._get_stream_headers()
        )

    def _build_model_config_from_user_model(self, user_model_id: int, user_id: Optional[int] = None) -> Optional[
        Dict[str, Any]]:
        """
        根据用户模型ID构建模型配置。
        【优化细节】使用 contextmanager 安全管理数据库连接，防止泄露。
        """
        # 只要在 with 代码块内，数据库连接就一直保持活跃；离开立刻释放
        with get_db_context() as db:
            user_model = user_model_service.get_user_model_by_id(db, user_model_id, user_id)
            if not user_model or not user_model.model_setting:
                return None
            model_setting = user_model.model_setting
            custom_config = user_model.custom_config or {}
            router_model = (
                custom_config["router_model"]
                if "router_model" in custom_config
                else MODEL_TIERING_CONFIG.router_model
            )
            simple_chat_model = (
                custom_config["simple_chat_model"]
                if "simple_chat_model" in custom_config
                else MODEL_TIERING_CONFIG.simple_chat_model
            )
            return {
                'model': user_model.selected_model,
                'model_service': model_setting.service_name,
                'service_type': model_setting.service_type,
                'router_model': router_model,
                'simple_chat_model': simple_chat_model,
                'deep_thinking_mode': custom_config.get('deep_thinking_mode', CHAT_DEFAULT_MODEL_CONFIG["deep_thinking_mode"]),
                'rag_enabled': custom_config.get('rag_enabled', CHAT_DEFAULT_MODEL_CONFIG["rag_enabled"]),
                'similarity_threshold': custom_config.get(
                    'similarity_threshold',
                    CHAT_DEFAULT_MODEL_CONFIG["similarity_threshold"],
                ),
                'embedding_model': custom_config.get('embedding_model', CHAT_DEFAULT_MODEL_CONFIG["embedding_model"]),
                'embedding_model_key': custom_config.get(
                    'embedding_model_key',
                    CHAT_DEFAULT_MODEL_CONFIG["embedding_model_key"],
                ),
                'temperature': custom_config.get('temperature', CHAT_DEFAULT_MODEL_CONFIG["temperature"]),
                'top_p': custom_config.get('top_p', CHAT_DEFAULT_MODEL_CONFIG["top_p"]),
                'max_tokens': custom_config.get('max_tokens', CHAT_DEFAULT_MODEL_CONFIG["max_tokens"]),
                'model_key': user_model.api_key,
                'model_url': user_model.api_url or model_setting.service_url
            }

    def stream_chat_with_history(
            self,
            user_input: str,
            session_id: str,
            model_config: Dict[str, Any],
            user_id: Optional[int] = None,
            history_messages: Optional[list] = None,
            runtime_context: Optional[Dict[str, Any]] = None,
            is_resume: bool = False,
    ) -> Generator[str, None, None]:
        """
        【核心】带历史记录保存的流式聊天。
        这里的异常处理必须极其严密，因为抛出异常时 HTTP Headers 已经发给前端了。
        """
        log.info(f"开始带历史的流式聊天，会话ID: {session_id}", target=LogTarget.ALL)
        history_messages = history_messages or []
        runtime_context = runtime_context or {}

        if not is_resume:
            with get_db_context() as db:
                chat_history_service.get_or_create_session(db, user_id, session_id, user_input)

        ai_response = ""
        start_time = time.time()
        error_occurred = False
        error_message = ""
        runner_stream = None

        try:
            runner_stream = self.graph_runner.stream_run(
                user_input=user_input,
                session_id=session_id,
                model_config=model_config,
                history_messages=history_messages,
                session_context=runtime_context,
            )
            # 1. 遍历 GraphRunner 推送过来的每一个流式事件
            for chunk in runner_stream:
                yield chunk
                # 2. 只有当事件包含大模型的正文时，才拼接到 ai_response 里准备落库
                ai_content = self._extract_ai_content_from_chunk(chunk)
                if ai_content:
                    ai_response += ai_content

        except GeneratorExit:
            # 前端主动断开连接 (比如关掉网页/点击停止生成)
            if runner_stream is not None:
                try:
                    runner_stream.close()
                except Exception:
                    pass
            log.warning(f"客户端主动断开连接，会话ID: {session_id}。保留已有记录。", target=LogTarget.LOG)
            error_occurred = True
            error_message = "客户端主动断开连接"
            raise  # 必须 re-raise GeneratorExit，这是 Python 协程规范

        except TimeoutError:
            error_occurred = True
            error_message = "模型响应超时，请重试"
            log.warning(f"流式聊天超时: {session_id}", target=LogTarget.LOG)
            yield self._format_error_event(CHAT_SERVICE_ERROR_TIMEOUT)

        except ConnectionError:
            error_occurred = True
            error_message = "远端服务连接断开"
            log.warning(f"流式聊天连接异常: {session_id}", target=LogTarget.LOG)
            yield self._format_error_event(CHAT_SERVICE_ERROR_CONNECTION)

        except Exception as e:
            # 流式传输中断裂，必须以 SSE 格式告诉前端
            error_occurred = True
            error_message = str(e)
            log.exception(f"聊天流处理异常: {error_message}", target=LogTarget.ALL)
            yield self._format_error_event(CHAT_SERVICE_ERROR_RUNTIME_TEMPLATE.format(error=error_message))

        finally:
            # 3. 收尾动作：落库。无论是正常结束、报错，还是前端断开，只要有话出来，统统存下。
            if user_id and not is_resume:
                # 哪怕 error_occurred，只要 ai_response 里面有半截话，也值得保存
                log.info(f"保存聊天历史，会话ID: {session_id}，总长: {len(ai_response)}字符", target=LogTarget.LOG)
                with get_db_context() as db:
                    # 组装要入库的最终回答文本
                    final_content = ai_response
                    if error_occurred:
                        if not ai_response:
                            final_content = CHAT_SERVICE_INTERRUPTED_TEMPLATE.format(error=error_message)
                        else:
                            final_content += CHAT_SERVICE_INTERRUPTED_APPEND_TEMPLATE.format(error=error_message)

                    # 仅当本轮有可存内容时才写聊天消息，避免空白消息污染历史
                    if final_content:
                        chat_data = ChatMessageCreate(
                            user_id=user_id,
                            session_id=session_id,
                            user_content=user_input,
                            model_content=final_content,
                            model_name=model_config.get('model'),
                            latency_ms=int((time.time() - start_time) * 1000),
                            tokens=ChatUtils.estimate_tokens(user_input + ai_response)
                        )
                        chat_history_service.create_chat_message(db, user_id, chat_data)

                    # 无论是否产出正文，都回写会话状态（轮次、路由快照、槽位补全）
                    session_state_service.update_after_turn(
                        db=db,
                        user_id=user_id,
                        session_id=session_id,
                        user_input=user_input,
                        ai_response=final_content,
                    )

    def stream_chat_anonymous(
            self,
            user_input: str,
            session_id: str,
            model_config: Dict[str, Any],
            history_messages: Optional[list] = None,
            runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """匿名流式聊天，不触碰数据库，纯粹的管道转发"""
        log.info(f"匿名流式聊天，会话ID: {session_id}", target=LogTarget.LOG)
        runner_stream = None
        try:
            runner_stream = self.graph_runner.stream_run(
                user_input=user_input,
                session_id=session_id,
                model_config=model_config,
                history_messages=history_messages or [],
                session_context=runtime_context or {},
            )
            yield from runner_stream
        except GeneratorExit:
            if runner_stream is not None:
                try:
                    runner_stream.close()
                except Exception:
                    pass
            raise
        except TimeoutError:
            yield self._format_error_event(CHAT_SERVICE_ERROR_TIMEOUT)
        except ConnectionError:
            yield self._format_error_event(CHAT_SERVICE_ERROR_CONNECTION)
        except Exception as e:
            log.error(f"匿名聊天异常: {e}")
            yield self._format_error_event(CHAT_SERVICE_ERROR_FALLBACK_TEMPLATE.format(error=str(e)))

    def _extract_ai_content_from_chunk(self, chunk: str) -> Optional[str]:
        """
        安全提取 SSE 格式内容
        SSE 格式标准为:
        event: message
        data: {"type": "message", "content": "你好"}
        """
        try:
            if chunk.startswith('event: '):
                lines = chunk.split('\n')
                data_line = next((line for line in lines if line.startswith('data: ')), None)
                if data_line:
                    data = json.loads(data_line[6:])
                    # 只记录主输出正文，忽略 thinking/interrupt 等中间状态
                    if data.get(SsePayloadField.TYPE.value) in CHAT_AI_CONTENT_TYPES:
                        return data.get(SsePayloadField.CONTENT.value, '')
            elif chunk.startswith('data: '):
                # 兼容旧版本纯 data 输出
                data = json.loads(chunk[6:])
                if data.get(SsePayloadField.TYPE.value) in CHAT_AI_CONTENT_TYPES:
                    return data.get(SsePayloadField.CONTENT.value, '')
        except (json.JSONDecodeError, KeyError):
            pass
        return None

    @staticmethod
    def _normalize_history_messages(history_data: Any) -> list[dict]:
        if not history_data:
            return []

        # 兼容 chat_history_service 返回 {"messages": [...]} 的结构
        if isinstance(history_data, dict):
            source_items = history_data.get("messages", []) or []
        elif isinstance(history_data, list):
            source_items = history_data
        else:
            source_items = []

        normalized = []
        for item in source_items:
            if isinstance(item, dict):
                normalized.append(item)
                continue

            normalized.append({
                "user_content": getattr(item, "user_content", ""),
                "model_content": getattr(item, "model_content", ""),
                "name": getattr(item, "model_name", None),
            })
        return normalized

    def _build_model_config_from_request(self, req: StreamChatRequest) -> Dict[str, Any]:
        """从请求中构建模型配置"""
        return self.build_model_config(
            model=req.model,
            model_service=req.model_service,
            service_type=req.service_type,
            router_model=req.router_model or None,
            simple_chat_model=req.simple_chat_model or None,
            deep_thinking_mode=req.deep_thinking_mode,
            rag_enabled=req.rag_enabled,
            similarity_threshold=req.similarity_threshold,
            embedding_model=req.embedding_model,
            embedding_model_key=req.embedding_model_key,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            model_key=req.model_key,
            model_url=req.model_url,
        )

    def build_model_config(
            self,
            model: Optional[str] = None,
            model_service: Optional[str] = None,
            service_type: Optional[str] = None,
            router_model: Optional[str] = None,
            simple_chat_model: Optional[str] = None,
            deep_thinking_mode: Optional[str] = None,
            rag_enabled: Optional[bool] = None,
            similarity_threshold: Optional[float] = None,
            embedding_model: Optional[str] = None,
            embedding_model_key: Optional[str] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            max_tokens: Optional[int] = None,
            model_key: Optional[str] = None,
            model_url: Optional[str] = None
    ) -> Dict[str, Any]:
        defaults = CHAT_DEFAULT_MODEL_CONFIG
        return {
            'model': model or defaults["model"],
            'model_service': model_service or defaults["model_service"],
            'service_type': service_type or defaults["service_type"],
            'router_model': MODEL_TIERING_CONFIG.router_model if router_model is None else router_model,
            'simple_chat_model': MODEL_TIERING_CONFIG.simple_chat_model if simple_chat_model is None else simple_chat_model,
            'deep_thinking_mode': deep_thinking_mode or defaults["deep_thinking_mode"],
            'rag_enabled': defaults["rag_enabled"] if rag_enabled is None else rag_enabled,
            'similarity_threshold': defaults["similarity_threshold"] if similarity_threshold is None else similarity_threshold,
            'embedding_model': embedding_model or defaults["embedding_model"],
            'embedding_model_key': embedding_model_key if embedding_model_key is not None else defaults["embedding_model_key"],
            'temperature': defaults["temperature"] if temperature is None else temperature,
            'top_p': defaults["top_p"] if top_p is None else top_p,
            'max_tokens': defaults["max_tokens"] if max_tokens is None else max_tokens,
            'model_key': model_key if model_key is not None else defaults["model_key"],
            'model_url': model_url if model_url is not None else defaults["model_url"],
        }

    def _get_stream_headers(self) -> Dict[str, str]:
        return dict(STREAM_HEADERS)

    @staticmethod
    def _format_error_event(message: str) -> str:
        """构造统一错误事件，避免服务层重复拼接 SSE 字符串。"""
        return ChatUtils.format_sse_data(
            event_type=SseEventType.ERROR.value,
            data={SsePayloadField.CONTENT.value: message},
        )


# 创建全局实例
chat_service = ChatService()
