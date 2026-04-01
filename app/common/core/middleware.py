# -*- coding: utf-8 -*-
import time
import uuid
import logging
from typing import Callable, Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import JSONResponse, Response

from db import get_db_context
from config.runtime_settings import MODEL_TIERING_CONFIG
from services.user_model_service import user_model_service
from common.utils.custom_logger import get_logger
from common.utils.pwd_utils import encryption_utils, TokenError

log = get_logger(__name__)


class ProcessTimeMiddleware(BaseHTTPMiddleware):
    """
    性能监控中间件
    计算路由耗时，并注入 X-Process-Time 到响应头中。
    同时为每个请求生成全局唯一的 request_id，便于追踪。
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        req_id = (
            request.headers.get("X-Request-ID")
            or request.headers.get("X-Request-Id")
            or str(uuid.uuid4())
        )
        request.state.request_id = req_id
        request.state.req_id = req_id
        
        start_time = time.time()
        
        # 将 req_id 存入日志上下文等价的操作（此处简单打印）
        log.info(f"[{req_id}] HTTP Request START: {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
        except Exception as e:
            log.error(f"[{req_id}] 框架内部未捕获异常抛出: {e}", exc_info=True)
            raise e
            
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = req_id
        
        log.info(f"[{req_id}] HTTP Request END: {request.method} {request.url.path} (耗时: {process_time:.4f}s) - {response.status_code}")
        
        return response


class SSESafeGZipMiddleware(GZipMiddleware):
    """
    对 SSE 路径禁用 gzip，避免流式分块被压缩缓冲导致首包/增量延迟。
    """

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            path = str(scope.get("path") or "")
            headers = dict(scope.get("headers") or [])
            accept = headers.get(b"accept", b"").decode("latin1").lower()
            if path.startswith("/api/v1/chat/stream") or "text/event-stream" in accept:
                return await self.app(scope, receive, send)
        return await super().__call__(scope, receive, send)


class DynamicModelMiddleware(BaseHTTPMiddleware):
    """
    动态模型上下文提取中间件
    统一管理大模型的底层调度参数。如果Header或者Body中指明了 user_model_id，则在请求真正击中控制器前，
    前往数据库拉取卡片配置注入到 `request.state.model_config` 中。
    让后端的 chat_service 能够无缝取用，降低 Service 层的 DB 读写查寻耦合。
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 只对特定的路由产生动作，比如 chat
        path = request.url.path
        if not path.startswith("/api/v1/chat"):
            return await call_next(request)
        is_anonymous_route = path.endswith("/chat/stream/anonymous")

        # 尝试从 body 预存取信息
        body_bytes = await request.body()
        user_model_id = None
        user_id = None
        
        # 兼容 JSON 解析
        if body_bytes:
            import json
            try:
                body_json = json.loads(body_bytes)
                if isinstance(body_json, dict):
                    user_model_id = body_json.get("user_model_id")
            except Exception:
                pass

        # 匿名接口禁止通过 user_model_id 读取用户私有模型配置，避免越权使用他人 key。
        if is_anonymous_route and user_model_id:
            log.warning(
                f"[{getattr(request.state, 'req_id', '?')}] 匿名接口忽略 user_model_id={user_model_id}，"
                "仅允许使用请求内显式模型参数。"
            )
            return await call_next(request)

        # 尝试从 Authorization 里解析当前用户；失败时不抛错，让下游认证依赖接管。
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header.split(" ", 1)[1].strip()
            if token:
                try:
                    user_id = encryption_utils.get_user_id_from_token(token)
                except TokenError:
                    user_id = None
                except Exception:
                    user_id = None
        
        # 如果提供了预设模型卡片 ID，就拦截打捞配置
        if user_model_id:
            if user_id is None:
                log.warning(
                    f"[{getattr(request.state, 'req_id', '?')}] user_model_id={user_model_id} 未通过用户鉴权，"
                    "已拒绝预加载动态模型配置。"
                )
                return await call_next(request)
            try:
                with get_db_context() as db:
                    user_model = user_model_service.get_user_model_by_id(db, user_model_id, user_id)
                    if user_model and user_model.model_setting:
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
                        
                        resolved_config = {
                            'model': user_model.selected_model,
                            'model_service': model_setting.service_name,
                            'service_type': model_setting.service_type,
                            'router_model': router_model,
                            'simple_chat_model': simple_chat_model,
                            'deep_thinking_mode': custom_config.get('deep_thinking_mode', 'auto'),
                            'rag_enabled': custom_config.get('rag_enabled', False),
                            'similarity_threshold': custom_config.get('similarity_threshold', 0.7),
                            'embedding_model': custom_config.get('embedding_model', 'bge-m3:latest'),
                            'embedding_model_key': custom_config.get('embedding_model_key', ''),
                            'temperature': custom_config.get('temperature', 0.2),
                            'top_p': custom_config.get('top_p', 1.0),
                            'max_tokens': custom_config.get('max_tokens', 2000),
                            'model_key': user_model.api_key,
                            'model_url': user_model.api_url or model_setting.service_url
                        }
                        
                        log.info(f"[{getattr(request.state, 'req_id', '?')}] 中间件已成功为模型 {user_model_id} 挂载动态配置。")
                        request.state.model_config = resolved_config
                    else:
                        log.warning(
                            f"[{getattr(request.state, 'req_id', '?')}] 动态模型加载失败: "
                            f"模型 {user_model_id} 不存在或不属于用户 {user_id}。"
                        )
            except Exception as e:
                log.error(f"中间件提取动态模型配置失败: {e}", exc_info=True)

        response = await call_next(request)
        return response
