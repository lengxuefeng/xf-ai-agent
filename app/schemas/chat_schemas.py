# -*- coding: utf-8 -*-
# app/schemas/chat_schemas.py
from typing import Optional

from pydantic import Field, field_validator

from schemas.base import BaseSchema
from utils.chat_utils import ChatUtils


class ChatRequest(BaseSchema):
    """
    聊天请求模型（简易版/旧版兼容）。
    """
    message: str
    settings: Optional["ModelSettings"] = None
    history: Optional[list[dict]] = None

    class ModelSettings(BaseSchema):
        deepThinkingMode: Optional[str] = None
        isRagEnabled: bool = False
        similarity: float = 0.92
        modelName: Optional[str] = "gpt-oss:20b"
        modelType: Optional[str] = "ollama"
        model: str = "gpt-oss:20b"
        embeddingModel: str = "bge-m3:latest"


class StreamChatRequest(BaseSchema):
    """
    流式聊天请求核心模型
    """
    user_input: str = Field(..., description="用户的输入文本")
    session_id: str = Field(..., description="用于追踪对话历史的会话ID")
    user_model_id: Optional[int] = Field(default=None, description="用户模型配置ID（优先使用此字段）")

    # --- 兼容性保留字段 (不再使用 Optional，直接指定具体类型与默认值) ---
    model: str = Field(default='google/gemini-1.5-pro', description="当前选择的模型")
    model_key: str = Field(default='', description="API Key")
    model_url: str = Field(default='', description="Base URL")
    model_service: str = Field(default='netlify-gemini', description="模型服务")
    service_type: str = Field(default='ollama', description="模型服务类型")
    deep_thinking_mode: str = Field(default='auto', description="深度思考模式")
    rag_enabled: bool = Field(default=False, description="RAG是否启用")
    embedding_model: str = Field(default='bge-m3:latest', description="嵌入模型")
    embedding_model_key: str = Field(default='', description="嵌入模型的key")

    # 【核心优化】利用 Pydantic V2 内置校验，代替手动编写的 validator
    # ge: greater than or equal (>=)
    # le: less than or equal (<=)
    # gt: greater than (>)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="相似度阈值")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="采样温度（越低越稳定）")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0, description="核采样 top_p")
    max_tokens: int = Field(default=2000, gt=0, description="最大输出 token")

    # --- 复杂业务校验保留 ---
    @field_validator('user_input')
    def validate_user_input(cls, v: str) -> str:
        """验证并清洗用户输入"""
        if not v or not v.strip():
            raise ValueError('用户输入不能为空')

        cleaned = ChatUtils.sanitize_user_input(v)
        if len(cleaned) > 5000:
            raise ValueError('用户输入过长，请控制在5000字符以内')

        return cleaned

    @field_validator('session_id')
    def validate_session_id(cls, v: str) -> str:
        """验证会话ID格式"""
        if not ChatUtils.validate_session_id(v):
            raise ValueError('无效的会话ID格式')
        return v
