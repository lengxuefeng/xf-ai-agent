# -*- coding: utf-8 -*-
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


@chat_router.post("/chat/stream", summary="流式聊天接口")
def stream_chat(req: StreamChatRequest):
    """
    处理流式聊天请求。

    本接口会实时返回 Agent 的思考过程和最终结果，采用 Server-Sent Events (SSE) 格式。

    - **user_input**: 用户发送的消息内容。
    - **session_id**: 客户端生成的唯一会话ID，用于多轮对话。

    **返回事件流:**
    - `{"type": "thinking", "content": "..."}`: Agent 的思考过程。
    - `{"type": "interrupt", "content": "..."}`: 需要用户输入的暂停事件。
    - `{"type": "response", "content": "..."}`: Agent 的最终回复。
    - `{"type": "error", "content": "..."}`: 执行过程中发生的错误。
    """
    runner = GraphRunner()
    return StreamingResponse(
        runner.stream_run(req.user_input, req.session_id),
        media_type="text/event-stream"
    )
