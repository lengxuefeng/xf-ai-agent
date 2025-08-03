from fastapi import APIRouter
from app.models.schemas import ChatRequest, ChatResponse
from app.services.agent import langgraph_agent

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    与 LangGraph 智能代理进行聊天。
    """
    response_content = langgraph_agent.invoke_agent(request.message)
    return ChatResponse(response=response_content)
