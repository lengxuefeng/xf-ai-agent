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

使用场景：
- FastAPI接口的请求体验证
- 参数序列化和反序列化
- 数据验证和错误提示
"""
# app/schemas/chat_schemas.py
from typing import Optional

from pydantic import Field, field_validator

from schemas.base import BaseSchema
from utils.chat_utils import ChatUtils


class ChatRequest(BaseSchema):
    """
    聊天请求模型（简易版/旧版兼容）。

    设计要点：
    1. 保持向后兼容，支持旧的API调用方式
    2. 简化的请求结构，降低使用复杂度
    3. 支持历史记录传入

    字段说明：
    - message: 用户输入的文本
    - settings: 模型设置（可选）
    - history: 对话历史（可选）

    使用场景：
    - 简易的聊天接口
    - 不需要复杂配置的场景
    """
    message: str
    settings: Optional["ModelSettings"] = None
    history: Optional[list[dict]] = None

    class ModelSettings(BaseSchema):
        """
        模型设置（简易版）。

        字段说明：
        - deepThinkingMode: 深度思考模式
        - isRagEnabled: 是否启用RAG
        - similarity: 相似度阈值
        - modelName: 模型名称
        - modelType: 模型类型
        - model: 模型标识
        - embeddingModel: 嵌入模型
        """
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

    字段说明：
    必填字段：
    - user_input: 用户输入文本
    - session_id: 会话ID，用于追踪对话历史

    可选字段：
    - user_model_id: 用户模型配置ID（优先使用）
    - model: 模型名称
    - model_key: API密钥
    - model_url: API地址
    - model_service: 模型服务提供商
    - service_type: 服务类型
    - router_model: 路由专用小模型
    - simple_chat_model: 简单对话专用小模型
    - deep_thinking_mode: 深度思考模式
    - rag_enabled: RAG开关
    - embedding_model: 嵌入模型
    - embedding_model_key: 嵌入模型密钥
    - workspace_root: 受控执行工作目录
    - resume_message_id: 恢复的审批消息ID

    模型参数：
    - similarity_threshold: RAG相似度阈值
    - temperature: 采样温度，越低越稳定
    - top_p: 核采样参数
    - max_tokens: 最大输出token数

    设计理由：
    1. 灵活的模型配置：支持多个LLM提供商
    2. 用户模型配置：用户可以保存常用的配置
    3. 审批恢复：支持审批后的恢复执行
    4. 受控执行：工作目录受限，保证安全性
    5. 严格校验：Pydantic V2内置校验，减少手写validator

    使用场景：
    - 流式聊天API的请求体验证
    - 支持多种模型的切换
    - 用户可以保存和复用配置
    - 审批恢复流程
    - 代码执行的受控环境
    """
    user_input: str = Field(..., description="用户的输入文本")
    session_id: str = Field(..., description="用于追踪对话历史的会话ID")
    user_model_id: Optional[int] = Field(default=None, description="用户模型配置ID（优先使用此字段）")

    # --- 兼容性保留字段 (不再使用 Optional，直接指定具体类型与默认值) ---
    # 这些字段提供了向后兼容性，新代码推荐使用user_model_id
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
    # ge: greater than or equal (>=)
    # le: less than or equal (<=)
    # gt: greater than (>)
    # 这些内置约束可以自动生成友好的错误提示
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="相似度阈值")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="采样温度（越低越稳定）")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0, description="核采样 top_p")
    max_tokens: int = Field(default=2000, gt=0, description="最大输出 token")

    # --- 复杂业务校验保留 ---
    @field_validator('user_input')
    def validate_user_input(cls, v: str) -> str:
        """验证并清洗用户输入

        设计要点：
        1. 不能为空
        2. 去除首尾空格
        3. 限制最大长度（5000字符）

        Args:
            v: 用户输入

        Returns:
            str: 清洗后的用户输入

        Raises:
            ValueError: 输入为空或过长

        使用场景：
        - API参数验证
        - 防止恶意输入
        """
        if not v or not v.strip():
            raise ValueError('用户输入不能为空')

        cleaned = ChatUtils.sanitize_user_input(v)
        if len(cleaned) > 5000:
            raise ValueError('用户输入过长，请控制在5000字符以内')

        return cleaned

    @field_validator('session_id')
    def validate_session_id(cls, v: str) -> str:
        """验证会话ID格式

        设计要点：
        1. 不能为空
        2. 只允许字母、数字、下划线、短横线
        3. 长度限制5-100字符

        Args:
            v: 会话ID

        Returns:
            str: 验证后的会话ID

        Raises:
            ValueError: 格式无效

        使用场景：
        - API参数验证
        - 防止恶意ID
        - 确保ID可识别性
        """
        if not ChatUtils.validate_session_id(v):
            raise ValueError('无效的会话ID格式')
        return v
