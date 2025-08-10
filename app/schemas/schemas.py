from typing import Optional, Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    聊天请求模型。
    Args:
        message: 用户发送的消息内容。
    """
    message: str


class ResponseModel(BaseModel):
    """
    响应模型。
    Args:
        code: 响应状态码，默认值为 200。
        message: 响应状态消息，默认值为 "success"。
        data: 响应数据，默认值为 None。
    """
    code: Optional[int] = 200
    message: Optional[str] = Field(default="success", description="响应状态消息")
    data: Optional[Any] = None
