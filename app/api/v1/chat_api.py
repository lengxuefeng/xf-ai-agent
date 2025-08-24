# -*- coding: utf-8 -*-
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from agent.graph_runner import GraphRunner

"""
定义流式聊天 API 接口
"""

# 创建一个独立的 API 路由，并命名为 chat_router 以便 main.py 导入
chat_router = APIRouter()


# 定义请求体的数据模型
class StreamChatRequest(BaseModel):
    user_input: str = Field(..., description="用户的输入文本")
    session_id: str = Field(..., description="用于追踪对话历史的会话ID")
    
    # 模型配置参数
    model: Optional[str] = Field(default='google/gemini-1.5-pro', description="当前选择的模型")
    model_service: Optional[str] = Field(default='netlify-gemini', description="模型服务")
    deep_thinking_mode: Optional[str] = Field(default='auto', description="深度思考模式")
    rag_enabled: Optional[bool] = Field(default=False, description="RAG是否启用")
    similarity_threshold: Optional[float] = Field(default=0.7, description="相似度阈值")
    embedding_model: Optional[str] = Field(default='bge-m3:latest', description="嵌入模型")


@chat_router.post("/chat/stream", summary="流式聊天接口")
def stream_chat(req: StreamChatRequest):
    """
    处理流式聊天请求。

    本接口会实时返回 Agent 的思考过程和最终结果，采用 Server-Sent Events (SSE) 格式。

    - **user_input**: 用户发送的消息内容。
    - **session_id**: 客户端生成的唯一会话ID，用于多轮对话。
    - **model**: 当前选择的模型名称。
    - **model_service**: 模型服务类型。
    - **deep_thinking_mode**: 深度思考模式。
    - **rag_enabled**: 是否启用RAG功能。
    - **similarity_threshold**: 相似度阈值。
    - **embedding_model**: 嵌入模型名称。

    **返回事件流:**
    - `{"type": "thinking", "content": "..."}`: Agent 的思考过程。
    - `{"type": "interrupt", "content": "..."}`: 需要用户输入的暂停事件。
    - `{"type": "response_start", "content": ""}`: 响应开始。
    - `{"type": "stream", "content": "..."}`: 逐字符流式内容。
    - `{"type": "response_end", "content": ""}`: 响应结束。
    - `{"type": "error", "content": "..."}`: 执行过程中发生的错误。
    """
    # 创建模型配置
    model_config = {
        'model': req.model,
        'model_service': req.model_service,
        'deep_thinking_mode': req.deep_thinking_mode,
        'rag_enabled': req.rag_enabled,
        'similarity_threshold': req.similarity_threshold,
        'embedding_model': req.embedding_model
    }
    
    runner = GraphRunner()
    return StreamingResponse(
        runner.stream_run(req.user_input, req.session_id, model_config),
        media_type="text/event-stream"
    )
