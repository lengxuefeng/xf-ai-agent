# -*- coding: utf-8 -*-
"""
统一模型加载器

根据模型服务类型和配置参数动态加载相应的模型实例。
"""

from typing import Optional, Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

from utils.custom_logger import get_logger
from .loader_llm_multi import (
    load_openai_compatible_model,
    load_openai_embeddings,
    load_tongyi_model,
    load_gemini_model
)
from .model_config import ModelConfig
from .ollama_model import load_ollama_model, load_ollama_embeddings

log = get_logger(__name__)
load_dotenv()


class UnifiedModelLoader:
    """
    模型加载的“中央调度室”
    利用工厂模式，将具体的模型实例化逻辑隐藏。外部使用者只需传入 Config 即可。
    """

    @classmethod
    def load_chat_model(cls, config: ModelConfig) -> BaseChatModel:
        """根据配置加载对应的聊天模型"""
        service = config.service_type.lower()
        model_name = config.model

        # 【核心优化】路由表。所有 OpenAI 兼容的服务直接映射到同一个加载器
        service_router = {
            'ollama': lambda c: load_ollama_model(c.model),  # ollama 的特有加载逻辑
            'openrouter': load_openai_compatible_model,
            'netlify-gemini': load_openai_compatible_model,
            'silicon-flow': load_openai_compatible_model,
            'zhipu': load_openai_compatible_model,
            'modelscope': load_openai_compatible_model,
            'openai': load_openai_compatible_model,
            'tongyi': load_tongyi_model,
            'gemini': load_gemini_model,
        }

        try:
            # 去路由表里找对应的加载函数，找不到则回退到 openai_compatible
            loader_func = service_router.get(service, load_openai_compatible_model)
            return loader_func(config)
        except Exception as e:
            log.error(f"加载模型失败 - 服务: {service}, 模型: {model_name}, 错误详情: {str(e)}")
            raise RuntimeError(f"模型加载失败({service}): {model_name}") from e

    @classmethod
    def load_embedding_model(cls, config: ModelConfig):
        """加载向量嵌入模型"""
        service = config.service_type.lower()
        embedding_model = config.embedding_model
        
        service_router = {
            'ollama': lambda c: load_ollama_embeddings(c.embedding_model),
        }
        
        try:
            # 默认使用 OpenAI 兼容的加载方式
            loader_func = service_router.get(service, load_openai_embeddings)
            return loader_func(config)
        except Exception as e:
            log.error(f"加载嵌入模型失败 - 服务: {service}, 模型: {embedding_model}, 错误: {str(e)}")
            raise RuntimeError(f"嵌入模型加载失败({service}): {embedding_model}") from e


def create_model_from_config(
        model: str,
        model_service: str,
        service_type: str,
        deep_thinking_mode: str = 'auto',
        rag_enabled: bool = False,
        similarity_threshold: float = 0.7,
        embedding_model: str = 'bge-m3:latest',
        model_key: str = '',
        model_url: str = '',
        embedding_model_key: str = '',
        **kwargs
) -> tuple[BaseChatModel, Optional[Any]]:
    """
    【对外接口】封装层，提供给 Supervisor 和 Agent 实例化模型使用。
    """
    config = ModelConfig(
        model=model,
        model_service=model_service,
        service_type=service_type,
        deep_thinking_mode=deep_thinking_mode,
        rag_enabled=rag_enabled,
        similarity_threshold=similarity_threshold,
        embedding_model=embedding_model,
        model_key=model_key,
        model_url=model_url,
        embedding_model_key=embedding_model_key,
        **kwargs
    )

    chat_model = UnifiedModelLoader.load_chat_model(config)
    embedding_model_instance = UnifiedModelLoader.load_embedding_model(config) if rag_enabled else None

    return chat_model, embedding_model_instance
