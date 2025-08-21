from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from agent.xf_graph import xf_graph
from schemas.chat_schemas import ChatRequest

chat_router = APIRouter(prefix="/chat", tags=["chat"])

"""
聊天接口
"""


@chat_router.post("/stream", response_class=StreamingResponse)
# def chat_with_agent_stream(request: ChatRequest, user_id: int = Depends(verify_token)) -> StreamingResponse:
def chat_with_agent_stream(request: ChatRequest) -> StreamingResponse:
    """
    与 LangGraph 智能代理进行聊天（流式返回）。
    """
    response_content = xf_graph.stream_agent_handle(request)
    return StreamingResponse(content=response_content, media_type="text/event-stream; charset=utf-8")
