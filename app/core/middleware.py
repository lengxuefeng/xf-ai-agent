# -*- coding: utf-8 -*-
import time
import uuid
import logging
from typing import Callable, Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from db import get_db_context
from services.user_model_service import user_model_service
from utils.custom_logger import get_logger

log = get_logger(__name__)


class ProcessTimeMiddleware(BaseHTTPMiddleware):
    """
    性能监控中间件
    计算路由耗时，并注入 X-Process-Time 到响应头中。
    同时为每个请求生成全局唯一的 request_id，便于追踪。
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        req_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
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
        response.headers["X-Request-Id"] = req_id
        
        log.info(f"[{req_id}] HTTP Request END: {request.method} {request.url.path} (耗时: {process_time:.4f}s) - {response.status_code}")
        
        return response


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

        # 尝试从 body 预存取信息
        body_bytes = await request.body()
        user_model_id = None
        user_id = None # 由于是中间件，JWT 解码可能需要在之前，如果是简化版直接查也是可以的
        
        # 兼容 JSON 解析
        if body_bytes:
            import json
            try:
                body_json = json.loads(body_bytes)
                if isinstance(body_json, dict):
                    user_model_id = body_json.get("user_model_id")
            except Exception:
                pass
        
        # 如果提供了预设模型卡片 ID，就拦截打捞配置
        if user_model_id:
            try:
                with get_db_context() as db:
                    user_model = user_model_service.get_user_model_by_id(db, user_model_id, None)
                    if user_model and user_model.model_setting:
                        model_setting = user_model.model_setting
                        custom_config = user_model.custom_config or {}
                        
                        resolved_config = {
                            'model': user_model.selected_model,
                            'model_service': model_setting.service_name,
                            'service_type': model_setting.service_type,
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
            except Exception as e:
                log.error(f"中间件提取动态模型配置失败: {e}", exc_info=True)

        response = await call_next(request)
        return response
