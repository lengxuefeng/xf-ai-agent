# app/schemas/chat_schemas.py

from typing import Optional

from schemas.base import BaseSchema


class ChatRequest(BaseSchema):
    """
    聊天请求模型。
    Args:
        message: 用户发送的消息内容。
    """
    message: str
    settings: Optional["ModelSettings"] = None
    history: Optional[list[dict]] = None

    class ModelSettings(BaseSchema):
        """
        模型配置。
        """
        deepThinkingMode: Optional[str] = None  # 深度思考模式
        isRagEnabled: bool = False  # RAG模式
        similarity: float = 0.92  # 相似度
        modelName: Optional[str] = "gpt-oss:20b"  # 模型名称
        modelType: Optional[str] = "ollama"  # 模型类型
        model: str = "gpt-oss:20b"  # 模型
        embeddingModel: str = "bge-m3:latest"  # 嵌入模型
