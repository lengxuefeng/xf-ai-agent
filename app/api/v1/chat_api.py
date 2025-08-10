from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from core.security import verify_token
from schemas.schemas import ChatRequest
from app.services.xf_graph import xf_graph

chat_router = APIRouter(prefix="/chat", tags=["chat"])

"""
聊天接口
"""


@chat_router.post("/chat", response_class=StreamingResponse)
def chat_with_agent_stream(request: ChatRequest, user_id: int = Depends(verify_token)) -> StreamingResponse:
    """
    与 LangGraph 智能代理进行聊天（流式返回）。
    """
    # TODO: 此处可以使用 user_id 为用户提供个性化的聊天体验。
    response_content = xf_graph.stream_agent(request.message)
    return StreamingResponse(content=response_content, media_type="text/event-stream; charset=utf-8")