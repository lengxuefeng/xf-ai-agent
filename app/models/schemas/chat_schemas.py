# -*- coding: utf-8 -*-
"""
聊天请求和响应的数据模型（Chat Schemas）。

定义了聊天服务中使用的所有Pydantic模型，包括请求验证、数据转换等功能。
这些模型用于API接口的参数验证和数据序列化。

设计要点：
1. 使用Pydantic V2进行参数验证和序列化
2. 支持流式聊天和普通聊天两种模式
3. 完整的模型配置参数
4. 严格的输入验证，提高安全性
"""
# app/schemas/chat_schemas.py
from typing import Optional

from pydantic import Field, field_validator

from models.schemas.base import BaseSchema
from common.utils.chat_utils import ChatUtils


class ChatRequest(BaseSchema):
    """
    聊天请求模型（简易版/旧版兼容）。

    设计要点：
    1. 保持向后兼容，支持旧的API调用方式
    2. 简化的请求结构，降低使用复杂度
    3. 支持历史记录传入
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
    流式聊天请求核心模型。

    设计要点：
    1. 支持完整的模型配置参数
    2. 使用Pydantic V2内置校验，替代手写validator
    3. 支持用户模型配置ID，优先级高于直接参数
    4. 支持审批恢复功能
    5. 支持受控执行工作目录
    """
    user_input: str = Field(..., description="用户的输入文本")
    session_id: str = Field(..., description="用于追踪对话历史的会话ID")
    user_model_id: Optional[int] = Field(default=None, description="用户模型配置ID（优先使用此字段）")

    # --- 兼容性保留字段 ---
    model: str = Field(default='glm-4', description="当前选择的模型")
    model_key: str = Field(default='', description="API Key")
    model_url: str = Field(default='', description="Base URL")
    model_service: str = Field(default='zhipu', description="模型服务")
    service_type: str = Field(default='zhipu', description="模型服务类型")
    router_model: str = Field(default='', description="路由专用小模型（可选）")
    simple_chat_model: str = Field(default='', description="简单对话专用小模型（可选）")
    deep_thinking_mode: str = Field(default='auto', description="深度思考模式")
    rag_enabled: bool = Field(default=False, description="RAG是否启用")
    embedding_model: str = Field(default='bge-m3:latest', description="嵌入模型")
    embedding_model_key: str = Field(default='', description="嵌入模型的key")
    workspace_root: Optional[str] = Field(default=None, description="受控执行工作目录")
    resume_message_id: Optional[str] = Field(default=None, description="指定恢复的审批消息 ID")

    # 【核心优化】利用 Pydantic V2 内置校验，代替手动编写的 validator
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="相似度阈值")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="采样温度（越低越稳定）")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0, description="核采样 top_p")
    max_tokens: int = Field(default=2000, gt=0, description="最大输出 token")

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


class ChatResponse(BaseSchema):
    """
    标准聊天响应模型。

    设计要点：
    1. 统一非流式返回结构
    2. 包含完整的监控度量指标(耗时、Token消耗)
    3. 支持扩展字段(如思考过程、引用链接等)
    """
    message_id: int = Field(..., description="落库的消息唯一标识ID")
    session_id: str = Field(..., description="追踪对话的会话ID")
    content: str = Field(..., description="模型或代理生成的回答内容")
    latency_ms: int = Field(default=0, description="执行总耗时(毫秒)")
    tokens: int = Field(default=0, description="预估消耗的总Token数")
    extra_data: Optional[dict] = Field(default=None, description="额外扩展数据，如 agent trace、思考内容等")
