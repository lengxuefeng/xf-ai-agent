from pydantic import BaseModel

class ChatRequest(BaseModel):
    """
    聊天请求模型。
    """
    message: str

class ChatResponse(BaseModel):
    """
    聊天响应模型。
    """
    response: str
