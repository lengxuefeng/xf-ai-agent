# -*- coding: utf-8 -*-
from typing import Any, Dict
from pydantic import BaseModel, ConfigDict, Field

"""
模型配置类 (Data Object) - Pydantic 升级版
职责：作为一个自带强校验的数据传输对象 (DTO)，在层与层之间安全地传递大模型参数。
"""


class ModelConfig(BaseModel):
    # 【Pydantic V2 核心配置】允许传入类定义之外的额外字段 (即 kwargs)
    # 这些额外传入的字段会被自动放进 self.model_extra 字典中
    model_config = ConfigDict(extra="allow")

    # --- 必填字段 (无默认值) ---
    model: str = Field(..., description="模型名称 (如: GLM-4.5-Flash, qwen-turbo)")
    model_service: str = Field(..., description="模型服务组别 (供业务逻辑归类使用)")
    service_type: str = Field(..., description="服务类型 (决定走哪个底层加载器，如: zhipu, openrouter)")

    # --- 可选字段 (有默认值) ---
    deep_thinking_mode: str = Field(default="auto", description="深度思考模式开关")
    model_size: str = Field(default="large", description="模型规格: fast 或 large")
    rag_enabled: bool = Field(default=False, description="是否启用 RAG 向量检索")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="向量检索的相似度阈值")
    embedding_model: str = Field(default="bge-m3:latest", description="所使用的 Embedding 模型名称")

    model_key: str = Field(default="", description="大模型调用的 API Key")
    model_url: str = Field(default="", description="大模型调用的 Base URL")
    embedding_model_key: str = Field(default="", description="Embedding 模型的 API Key")

    @property
    def extra_params(self) -> Dict[str, Any]:
        """
        【兼容性黑魔法】
        为了不破坏已有代码对 config.extra_params 的调用，
        我们将 Pydantic 原生的 model_extra 暴露为 extra_params。
        外部传入的 temperature, top_p 等 **kwargs 都会在这里被安全提取。
        """
        return self.model_extra or {}
