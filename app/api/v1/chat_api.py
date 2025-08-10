from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from schemas.schemas import ChatRequest
from app.services.xf_graph import xf_graph

chat_router = APIRouter(prefix="/chat", tags=["chat"])

"""
聊天接口
"""


@chat_router.post("/chat", response_class=StreamingResponse)
def chat_with_agent_stream(request: ChatRequest) -> StreamingResponse:
    """
    与 LangGraph 智能代理进行聊天（流式返回）。
    """
    response_content = xf_graph.stream_agent(request.message)
    return StreamingResponse(content=response_content, media_type="text/event-stream; charset=utf-8")

