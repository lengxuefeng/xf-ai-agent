# app/schemas/chat_schemas.py
from typing import Optional

from pydantic import Field, field_validator

from schemas.base import BaseSchema
from utils.chat_utils import ChatUtils


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


class StreamChatRequest(BaseSchema):
    user_input: str = Field(..., description="用户的输入文本")
    session_id: str = Field(..., description="用于追踪对话历史的会话ID")

    # 模型配置参数
    model: Optional[str] = Field(default='google/gemini-1.5-pro', description="当前选择的模型")
    model_service: Optional[str] = Field(default='netlify-gemini', description="模型服务")
    deep_thinking_mode: Optional[str] = Field(default='auto', description="深度思考模式")
    rag_enabled: Optional[bool] = Field(default=False, description="RAG是否启用")
    similarity_threshold: Optional[float] = Field(default=0.7, description="相似度阈值")
    embedding_model: Optional[str] = Field(default='bge-m3:latest', description="嵌入模型")

    @field_validator('user_input')
    @classmethod
    def validate_user_input(cls, v: str) -> str:
        """验证用户输入"""
        if not v or not v.strip():
            raise ValueError('用户输入不能为空')

        cleaned = ChatUtils.sanitize_user_input(v)
        if len(cleaned) > 5000:
            raise ValueError('用户输入过长，请控制在5000字符以内')

        return cleaned

    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """验证会话ID"""
        if not ChatUtils.validate_session_id(v):
            raise ValueError('无效的会话ID格式')
        return v

    @field_validator('similarity_threshold')
    @classmethod
    def validate_similarity_threshold(cls, v: float) -> float:
        """验证相似度阈值"""
        if not 0.0 <= v <= 1.0:
            raise ValueError('相似度阈值必须在0.0到1.0之间')
        return v
