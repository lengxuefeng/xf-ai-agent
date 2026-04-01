# -*- coding: utf-8 -*-
import asyncio
import time
from typing import AsyncGenerator, Generator, Optional, Dict, Any

from sqlalchemy.orm import Session
from fastapi import Request
from starlette.responses import StreamingResponse

from harness.graph_runner import GraphRunner
from services.request_cancellation_service import request_cancellation_service
from config.constants.chat_service_constants import (
    CHAT_DEFAULT_MODEL_CONFIG,
    CHAT_SERVICE_ERROR_CONNECTION,
    CHAT_SERVICE_ERROR_FALLBACK_TEMPLATE,
    CHAT_SERVICE_ERROR_RUNTIME_TEMPLATE,
    CHAT_SERVICE_ERROR_TIMEOUT,
    STREAM_HEADERS,
)
from config.constants.sse_constants import SseEventType, SsePayloadField
from config.runtime_settings import CHAT_STREAM_HISTORY_LIMIT, MODEL_TIERING_CONFIG
from db import get_db_context
from models.schemas.chat_history_schemas import ChatMessageCreate
from models.schemas.chat_schemas import StreamChatRequest
from services.chat_history_service import chat_history_service
from services.chat_stream_support import (
    build_chat_extra_data,
    build_final_response,
    close_stream_safely,
    collect_trace_from_chunk,
    extract_ai_content_from_chunk,
    format_error_event,
    iterate_stream_sync,
    normalize_history_messages,
)
from services.exception_service import exception_handler
from services.route_metrics_service import route_metrics_service
from services.session_state_service import session_state_service
from services.user_model_service import user_model_service
from common.utils.chat_utils import ChatUtils
from common.utils.custom_logger import get_logger, LogTarget

log = get_logger(__name__)


async def _single_error_stream(detail: str):
    yield format_error_event(detail)


async def _disconnect_aware_stream(original_body, request: Request, session_id: str):
    try:
        async with request_cancellation_service.cancel_on_disconnect(session_id, request):
            async for chunk in original_body:
                if await request.is_disconnected():
                    log.info(f"客户端断连，停止 SSE 推送。session_id={session_id[:16]}")
                    request_cancellation_service.cancel_request(session_id)
                    break
                yield chunk
    finally:
        request_cancellation_service.cleanup_request(session_id)


def _attach_disconnect_cancellation(
    response: StreamingResponse,
    request: Request,
    session_id: str,
) -> StreamingResponse:
    original_body = response.body_iterator
    request_cancellation_service.register_request(session_id)
    response.body_iterator = _disconnect_aware_stream(original_body, request, session_id)
    return response


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
        【入口】统一处理流式聊天请求（同步兼容版，内部仍用同步生成器）
        """
        log.info(f"收到流式聊天请求，会话ID: {req.session_id}, 用户ID: {user_id}", target=LogTarget.LOG)
        if req.user_input and req.user_input != "[RESUME]":
            route_metrics_service.detect_and_record_correction(req.session_id, req.user_input)

        return exception_handler.safe_execute(
            self._process_stream_chat_internal,
            req, user_id, db, dynamic_model_config,
            context="流式聊天准备阶段"
        )

    async def process_stream_chat_async(
            self,
            req: StreamChatRequest,
            request: Request,
            user_id: Optional[int],
            db: Optional[Session] = None,
    ) -> StreamingResponse:
        """
        【异步入口】统一处理流式聊天请求，返回 StreamingResponse 包裹异步生成器。
        
        封装了 Middleware 动态模型配置和 SSE 流控制断连逻辑。
        """
        log.info(f"收到异步流式聊天请求，会话ID: {req.session_id}, 用户ID: {user_id}", target=LogTarget.LOG)
        if req.user_input and req.user_input != "[RESUME]":
            route_metrics_service.detect_and_record_correction(req.session_id, req.user_input)

        dynamic_model_config = getattr(request.state, "model_config", None)
        
        try:
            request_id = str(
                getattr(request.state, "request_id", "")
                or getattr(request.state, "req_id", "")
                or ""
            ).strip()
            stream_gen = self.build_stream_generator(
                req=req,
                user_id=user_id,
                dynamic_model_config=dynamic_model_config,
                request_id=request_id,
            )
        except Exception as exc:
            log.exception(f"流式聊天准备阶段异常: {exc}", target=LogTarget.ALL)
            return StreamingResponse(
                _single_error_stream(str(exc)),
                media_type="text/event-stream",
                headers=self._get_stream_headers(),
            )

        response = StreamingResponse(
            stream_gen,
            media_type="text/event-stream",
            headers=self._get_stream_headers(),
        )
        return _attach_disconnect_cancellation(response, request, req.session_id)

    def _resolve_model_config(
            self,
            req: StreamChatRequest,
            dynamic_model_config: Optional[Dict[str, Any]],
            user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """统一解析模型配置：优先使用中间件预加载配置，其次从请求参数构建。"""
        if dynamic_model_config:
            log.info("使用中间件预加载的动态模型配置", target=LogTarget.LOG)
            return dict(dynamic_model_config)
        if req.user_model_id and user_id is not None:
            resolved = self._build_model_config_from_user_model(req.user_model_id, user_id)
            if resolved:
                log.info("使用 user_model_id 动态装载模型配置", target=LogTarget.LOG)
                return dict(resolved)
        log.info("使用请求自带的默认参数构建配置", target=LogTarget.LOG)
        return self._build_model_config_from_request(req)

    def build_stream_generator(
        self,
        *,
        req: StreamChatRequest,
        user_id: Optional[int],
        dynamic_model_config: Optional[Dict[str, Any]] = None,
        request_id: str = "",
    ) -> AsyncGenerator[str, None]:
        model_config = self._resolve_model_config(req, dynamic_model_config, user_id=user_id)
        model_config = self._attach_runtime_overrides(model_config, req, user_id)
        is_resume = req.user_input == "[RESUME]"

        if user_id:
            return self.stream_chat_with_history_async(
                user_input=req.user_input,
                session_id=req.session_id,
                model_config=model_config,
                user_id=user_id,
                is_resume=is_resume,
                history_limit=CHAT_STREAM_HISTORY_LIMIT,
                request_id=request_id,
            )

        return self.stream_chat_anonymous_async(
            user_input=req.user_input,
            session_id=req.session_id,
            model_config=model_config,
            history_messages=[],
            runtime_context={},
            request_id=request_id,
        )

    @staticmethod
    def _attach_runtime_overrides(
        model_config: Dict[str, Any],
        req: StreamChatRequest,
        user_id: Optional[int],
    ) -> Dict[str, Any]:
        effective_config = dict(model_config or {})
        resolved_skill_ids = req.skill_ids or ([req.skill_id] if req.skill_id else [])
        if resolved_skill_ids:
            effective_config["skill_ids"] = resolved_skill_ids
        effective_config["runtime_user_id"] = user_id
        return effective_config

    def _process_stream_chat_internal(
            self,
            req: StreamChatRequest,
            user_id: Optional[int],
            db: Optional[Session] = None,
            dynamic_model_config: Optional[Dict[str, Any]] = None
    ) -> StreamingResponse:
        """内部流式聊天处理逻辑：准备配置、挂载生成器（同步兼容版）"""
        log.info(f"请求参数: model={req.model}, user_model_id={req.user_model_id}, service_type={req.service_type}",
                 target=LogTarget.LOG)

        model_config = self._resolve_model_config(req, dynamic_model_config, user_id=user_id)
        model_config = self._attach_runtime_overrides(model_config, req, user_id)
        is_resume = req.user_input == "[RESUME]"

        # 核心分流：注册用户带记忆，匿名用户无记忆
        if user_id:
            stream_generator = self.stream_chat_with_history(
                user_input=req.user_input,
                session_id=req.session_id,
                model_config=model_config,
                user_id=user_id,
                is_resume=is_resume,
                history_limit=CHAT_STREAM_HISTORY_LIMIT,
            )
        else:
            stream_generator = self.stream_chat_anonymous(
                user_input=req.user_input,
                session_id=req.session_id,
                model_config=model_config,
                history_messages=[],
                runtime_context={},
            )

        return StreamingResponse(
            stream_generator,
            media_type="text/event-stream",
            headers=self._get_stream_headers()
        )

    @staticmethod
    def _build_extra_data(
        thinking_entries: list[str],
        workflow_trace: list[dict],
        *,
        session_id: str = "",
        final_response: str = "",
    ) -> Optional[Dict[str, Any]]:
        """兼容旧调用方，内部改由独立辅助模块组装扩展数据。"""
        return build_chat_extra_data(
            thinking_entries,
            workflow_trace,
            session_id=session_id,
            final_response=final_response,
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

    async def stream_chat_with_history_async(
            self,
            user_input: str,
            session_id: str,
            model_config: Dict[str, Any],
            user_id: Optional[int] = None,
            is_resume: bool = False,
            history_limit: int = CHAT_STREAM_HISTORY_LIMIT,
            emit_response_start: bool = True,
            request_id: str = "",
    ) -> AsyncGenerator[str, None]:
        """
        【异步核心】带历史记录保存的流式聊天。

        阻塞式 DB 操作全部通过 asyncio.to_thread 在线程池执行，
        不阻塞 asyncio 事件循环，支持真正并发的 SSE 推送。
        """
        log.info(f"开始异步带历史的流式聊天，会话ID: {session_id}", target=LogTarget.ALL)
        history_messages: list[dict] = []
        runtime_context: Dict[str, Any] = {}

        if emit_response_start:
            yield ChatUtils.format_sse_data(
                event_type=SseEventType.RESPONSE_START.value,
                data={SsePayloadField.CONTENT.value: ""},
            )

        ai_response = ""
        thinking_entries: list[str] = []
        workflow_trace: list[dict] = []
        start_time = time.time()
        error_occurred = False
        error_message = ""

        try:
            # 非 resume 场景：在线程池中执行阻塞 DB 操作
            if not is_resume:
                await asyncio.to_thread(
                    self._init_session_and_load_history,
                    user_id, session_id, user_input, history_limit,
                    history_messages, runtime_context,
                )

            # 驱动异步图执行器
            async for chunk in self.graph_runner.stream_run(
                user_input=user_input,
                session_id=session_id,
                model_config=model_config,
                history_messages=history_messages,
                session_context=runtime_context,
                emit_response_start=False,
                request_id=request_id,
            ):
                yield chunk
                collect_trace_from_chunk(chunk, thinking_entries, workflow_trace)
                ai_content = extract_ai_content_from_chunk(chunk)
                if ai_content:
                    ai_response += ai_content

        except GeneratorExit:
            log.warning(f"客户端主动断开连接，会话ID: {session_id}。保留已有记录。", target=LogTarget.LOG)
            error_occurred = True
            error_message = "客户端主动断开连接"
            raise

        except TimeoutError:
            error_occurred = True
            error_message = "模型响应超时，请重试"
            log.warning(f"异步流式聊天超时: {session_id}", target=LogTarget.LOG)
            yield format_error_event(CHAT_SERVICE_ERROR_TIMEOUT)

        except ConnectionError:
            error_occurred = True
            error_message = "远端服务连接断开"
            log.warning(f"异步流式聊天连接异常: {session_id}", target=LogTarget.LOG)
            yield format_error_event(CHAT_SERVICE_ERROR_CONNECTION)

        except Exception as e:
            error_occurred = True
            error_message = str(e)
            log.exception(f"异步聊天流处理异常: {error_message}", target=LogTarget.ALL)
            yield format_error_event(CHAT_SERVICE_ERROR_RUNTIME_TEMPLATE.format(error=error_message))

        finally:
            self._finalize_chat_history_async(
                user_id, session_id, is_resume, user_input, 
                ai_response, model_config, start_time, 
                error_occurred, error_message, 
                thinking_entries, workflow_trace, runtime_context
            )

    def _finalize_chat_history_async(self, user_id, session_id, is_resume, user_input, ai_response, model_config, start_time, error_occurred, error_message, thinking_entries, workflow_trace, runtime_context):
        if user_id and not is_resume:
            log.info(f"保存聊天历史，会话ID: {session_id}，总长: {len(ai_response)}字符", target=LogTarget.LOG)
            asyncio.create_task(asyncio.to_thread(
                self._save_chat_history,
                user_id, session_id, user_input, ai_response,
                model_config, start_time, error_occurred, error_message,
                self._build_extra_data(
                    thinking_entries,
                    workflow_trace,
                    session_id=session_id,
                    final_response=ai_response,
                ),
                runtime_context,
                route_metrics_service.get_last_route(session_id),
            ))

    def _init_session_and_load_history(
            self,
            user_id: Optional[int],
            session_id: str,
            user_input: str,
            history_limit: int,
            history_messages: list,
            runtime_context: dict,
    ) -> None:
        """
        阻塞式初始化会话和加载历史记录。
        设计为可在 asyncio.to_thread 中调用的纯同步方法。
        """
        with get_db_context() as db:
            chat_history_service.get_or_create_session(db, user_id, session_id, user_input)

        if user_id:
            try:
                with get_db_context() as db:
                    recent_history = chat_history_service.get_recent_session_messages(
                        db=db,
                        user_id=user_id,
                        session_id=session_id,
                        limit=history_limit,
                    )
                    loaded_history = normalize_history_messages(recent_history)
                    loaded_context = session_state_service.build_runtime_context(
                        db=db,
                        user_id=user_id,
                        session_id=session_id,
                        user_input=user_input,
                        history_messages=loaded_history,
                    )
                # 就地修改，让调用方可以读到结果
                history_messages.extend(loaded_history)
                runtime_context.update(loaded_context)
            except Exception as preload_exc:
                log.warning(f"会话上下文预热失败，已降级为空上下文继续: {preload_exc}", target=LogTarget.LOG)

    def _save_chat_history(
            self,
            user_id: int,
            session_id: str,
            user_input: str,
            ai_response: str,
            model_config: Dict[str, Any],
            start_time: float,
            error_occurred: bool,
            error_message: str,
            extra_data: Optional[Dict[str, Any]] = None,
            runtime_context: Optional[Dict[str, Any]] = None,
            route_snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        阻塞式落库。设计为可在 asyncio.to_thread 中调用的纯同步方法。
        """
        with get_db_context() as db:
            final_content = build_final_response(ai_response, error_occurred, error_message)

            if final_content:
                chat_data = ChatMessageCreate(
                    user_id=user_id,
                    session_id=session_id,
                    user_content=user_input,
                    model_content=final_content,
                    model_name=model_config.get('model'),
                    latency_ms=int((time.time() - start_time) * 1000),
                    tokens=ChatUtils.estimate_tokens(user_input + ai_response),
                    extra_data=extra_data,
                )
                chat_history_service.create_chat_message(db, user_id, chat_data, ensure_session=False)

            session_state_service.update_after_turn(
                db=db,
                user_id=user_id,
                session_id=session_id,
                user_input=user_input,
                ai_response=final_content,
                runtime_context=runtime_context,
                route_snapshot=route_snapshot,
            )

    async def stream_chat_anonymous_async(
            self,
            user_input: str,
            session_id: str,
            model_config: Dict[str, Any],
            history_messages: Optional[list] = None,
            runtime_context: Optional[Dict[str, Any]] = None,
            request_id: str = "",
    ) -> AsyncGenerator[str, None]:
        """匿名异步流式聊天，不触碰数据库，纯粹的管道转发"""
        log.info(f"匿名异步流式聊天，会话ID: {session_id}", target=LogTarget.LOG)
        try:
            async for chunk in self.graph_runner.stream_run(
                user_input=user_input,
                session_id=session_id,
                model_config=model_config,
                history_messages=history_messages or [],
                session_context=runtime_context or {},
                request_id=request_id,
            ):
                yield chunk
        except GeneratorExit:
            raise
        except TimeoutError:
            yield format_error_event(CHAT_SERVICE_ERROR_TIMEOUT)
        except ConnectionError:
            yield format_error_event(CHAT_SERVICE_ERROR_CONNECTION)
        except Exception as e:
            log.error(f"匿名异步聊天异常: {e}")
            yield format_error_event(CHAT_SERVICE_ERROR_FALLBACK_TEMPLATE.format(error=str(e)))

    def stream_chat_with_history(
            self,
            user_input: str,
            session_id: str,
            model_config: Dict[str, Any],
            user_id: Optional[int] = None,
            is_resume: bool = False,
            history_limit: int = CHAT_STREAM_HISTORY_LIMIT,
            emit_response_start: bool = True,
    ) -> Generator[str, None, None]:
        """
        【核心】带历史记录保存的流式聊天。
        这里的异常处理必须极其严密，因为抛出异常时 HTTP Headers 已经发给前端了。
        """
        log.info(f"开始带历史的流式聊天，会话ID: {session_id}", target=LogTarget.ALL)
        history_messages: list[dict] = []
        runtime_context: Dict[str, Any] = {}

        if emit_response_start:
            yield ChatUtils.format_sse_data(
                event_type=SseEventType.RESPONSE_START.value,
                data={SsePayloadField.CONTENT.value: ""},
            )

        ai_response = ""
        thinking_entries: list[str] = []
        workflow_trace: list[dict] = []
        start_time = time.time()
        error_occurred = False
        error_message = ""
        runner_stream = None

        try:
            # 首包已发出后，再进行会话准备和轻量上下文预热，避免首包阻塞。
            if not is_resume:
                with get_db_context() as db:
                    chat_history_service.get_or_create_session(db, user_id, session_id, user_input)
            if user_id and not is_resume:
                try:
                    with get_db_context() as db:
                        recent_history = chat_history_service.get_recent_session_messages(
                            db=db,
                            user_id=user_id,
                            session_id=session_id,
                            limit=history_limit,
                        )
                        history_messages = normalize_history_messages(recent_history)
                        runtime_context = session_state_service.build_runtime_context(
                            db=db,
                            user_id=user_id,
                            session_id=session_id,
                            user_input=user_input,
                            history_messages=history_messages,
                        )
                except Exception as preload_exc:
                    log.warning(f"会话上下文预热失败，已降级为空上下文继续: {preload_exc}", target=LogTarget.LOG)
                    history_messages = []
                    runtime_context = {}

            runner_stream = self.graph_runner.stream_run(
                user_input=user_input,
                session_id=session_id,
                model_config=model_config,
                history_messages=history_messages,
                session_context=runtime_context,
                emit_response_start=False,
            )
            # 1. 遍历 GraphRunner 推送过来的每一个流式事件
            for chunk in iterate_stream_sync(runner_stream):
                yield chunk
                collect_trace_from_chunk(chunk, thinking_entries, workflow_trace)
                # 2. 只有当事件包含大模型的正文时，才拼接到 ai_response 里准备落库
                ai_content = extract_ai_content_from_chunk(chunk)
                if ai_content:
                    ai_response += ai_content

        except GeneratorExit:
            # 前端主动断开连接 (比如关掉网页/点击停止生成)
            if runner_stream is not None:
                close_stream_safely(runner_stream)
            log.warning(f"客户端主动断开连接，会话ID: {session_id}。保留已有记录。", target=LogTarget.LOG)
            error_occurred = True
            error_message = "客户端主动断开连接"
            raise  # 必须 re-raise GeneratorExit，这是 Python 协程规范

        except TimeoutError:
            error_occurred = True
            error_message = "模型响应超时，请重试"
            log.warning(f"流式聊天超时: {session_id}", target=LogTarget.LOG)
            yield format_error_event(CHAT_SERVICE_ERROR_TIMEOUT)

        except ConnectionError:
            error_occurred = True
            error_message = "远端服务连接断开"
            log.warning(f"流式聊天连接异常: {session_id}", target=LogTarget.LOG)
            yield format_error_event(CHAT_SERVICE_ERROR_CONNECTION)

        except Exception as e:
            # 流式传输中断裂，必须以 SSE 格式告诉前端
            error_occurred = True
            error_message = str(e)
            log.exception(f"聊天流处理异常: {error_message}", target=LogTarget.ALL)
            yield format_error_event(CHAT_SERVICE_ERROR_RUNTIME_TEMPLATE.format(error=error_message))

        finally:
            self._finalize_chat_history_sync(
                user_id, session_id, is_resume, user_input, 
                ai_response, model_config, start_time, 
                error_occurred, error_message, 
                thinking_entries, workflow_trace, runtime_context
            )

    def _finalize_chat_history_sync(self, user_id, session_id, is_resume, user_input, ai_response, model_config, start_time, error_occurred, error_message, thinking_entries, workflow_trace, runtime_context):
        if user_id and not is_resume:
            log.info(f"保存聊天历史，会话ID: {session_id}，总长: {len(ai_response)}字符", target=LogTarget.LOG)
            with get_db_context() as db:
                final_content = build_final_response(ai_response, error_occurred, error_message)
                if final_content:
                    chat_data = ChatMessageCreate(
                        user_id=user_id,
                        session_id=session_id,
                        user_content=user_input,
                        model_content=final_content,
                        model_name=model_config.get('model'),
                        latency_ms=int((time.time() - start_time) * 1000),
                        tokens=ChatUtils.estimate_tokens(user_input + ai_response),
                        extra_data=self._build_extra_data(
                            thinking_entries,
                            workflow_trace,
                            session_id=session_id,
                            final_response=final_content,
                        ),
                    )
                    chat_history_service.create_chat_message(db, user_id, chat_data, ensure_session=False)

                route_snapshot = route_metrics_service.get_last_route(session_id)
                session_state_service.update_after_turn(
                    db=db,
                    user_id=user_id,
                    session_id=session_id,
                    user_input=user_input,
                    ai_response=final_content,
                    runtime_context=runtime_context,
                    route_snapshot=route_snapshot,
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
            for chunk in iterate_stream_sync(runner_stream):
                yield chunk
        except GeneratorExit:
            if runner_stream is not None:
                close_stream_safely(runner_stream)
            raise
        except TimeoutError:
            yield format_error_event(CHAT_SERVICE_ERROR_TIMEOUT)
        except ConnectionError:
            yield format_error_event(CHAT_SERVICE_ERROR_CONNECTION)
        except Exception as e:
            log.error(f"匿名聊天异常: {e}")
            yield format_error_event(CHAT_SERVICE_ERROR_FALLBACK_TEMPLATE.format(error=str(e)))

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
            workspace_root=req.workspace_root,
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
            model_url: Optional[str] = None,
            workspace_root: Optional[str] = None,
            resume_message_id: Optional[str] = None,
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
            'workspace_root': workspace_root or '',
            'resume_message_id': resume_message_id or '',
        }

    def _get_stream_headers(self) -> Dict[str, str]:
        return dict(STREAM_HEADERS)


chat_service = ChatService()
