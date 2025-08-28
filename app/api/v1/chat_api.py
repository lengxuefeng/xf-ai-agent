# -*- coding: utf-8 -*-
from typing import Optional

from fastapi import APIRouter, Depends
from starlette.responses import StreamingResponse

from core.security import get_optional_user_id
from schemas.chat_schemas import StreamChatRequest
from services.chat_service import chat_service

"""
定义流式聊天 API 接口
"""

# 创建一个独立的 API 路由，并命名为 chat_router 以便 main.py 导入
chat_router = APIRouter()


@chat_router.post("/chat/stream", summary="流式聊天接口")
def stream_chat(req: StreamChatRequest, user_id: Optional[int] = Depends(get_optional_user_id)) -> StreamingResponse:
    """
    处理流式聊天请求。

    本接口会实时返回 Agent 的思考过程和最终结果，采用 Server-Sent Events (SSE) 格式。
    支持用户认证，认证用户的聊天记录会自动保存到聊天历史中。

    **参数说明:**
    - **user_input**: 用户发送的消息内容
    - **session_id**: 客户端生成的唯一会话ID，用于多轮对话
    - **model**: 当前选择的模型名称
    - **model_service**: 模型服务类型
    - **deep_thinking_mode**: 深度思考模式
    - **rag_enabled**: 是否启用RAG功能
    - **similarity_threshold**: 相似度阈值
    - **embedding_model**: 嵌入模型名称

    **返回事件流:**
    - `{"type": "thinking", "content": "..."}`: Agent 的思考过程
    - `{"type": "interrupt", "content": "..."}`: 需要用户输入的暂停事件
    - `{"type": "response_start", "content": ""}`: 响应开始
    - `{"type": "stream", "content": "..."}`: 逐字符流式内容
    - `{"type": "response_end", "content": ""}`: 响应结束
    - `{"type": "error", "content": "..."}`: 执行过程中发生的错误
    """
    return chat_service.process_stream_chat(req, user_id)


@chat_router.post("/chat/stream/anonymous", summary="匿名流式聊天接口")
def stream_chat_anonymous(req: StreamChatRequest) -> StreamingResponse:
    """
    匿名用户的流式聊天接口，不需要认证，不保存聊天历史。
    
    接口参数和返回格式与认证接口相同，但不会保存任何聊天记录。
    适用于游客用户或不需要保存历史的场景。
    """
    # 直接调用服务层处理，不做任何业务逻辑处理
    return chat_service.process_stream_chat(req, user_id=None)
