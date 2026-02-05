# -*- coding: utf-8 -*-
"""
统一模型加载器

根据模型服务类型和配置参数动态加载相应的模型实例。
支持多种模型服务：ollama、OpenRouter、silicon-flow、zhipu、tongyi等。
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

from .model_config import ModelConfig
from .ollama_model import load_ollama_model, load_ollama_embeddings
from .loader_llm_multi import (
    load_open_router,
    load_silicon_flow,
    load_zhipu_model,
    load_tongyi_model,
    load_chat_model,
    load_gemini_model, load_openai_model, load_modelscope_llm
)

load_dotenv()



class UnifiedModelLoader:
    """统一模型加载器"""

    # 支持的模型服务映射
    SERVICE_LOADERS = {
        'ollama': 'load_ollama_model',
        'openrouter': 'load_open_router',
        'netlify-gemini': 'load_open_router',  # Gemini通过OpenRouter调用
        'silicon-flow': 'load_silicon_flow',
        'zhipu': 'load_zhipu_model',
        'tongyi': 'load_tongyi_model',
    }

    @classmethod
    def load_chat_model(cls, config: ModelConfig) -> BaseChatModel:
        """
        根据配置加载聊天模型
        
        Args:
            config: 模型配置对象
            
        Returns:
            BaseChatModel: 加载的聊天模型实例
            
        Raises:
            ValueError: 不支持的模型服务类型
            RuntimeError: 模型加载失败
        """
        service = config.service_type.lower()
        model_name = config.model

        try:
            if service == 'ollama':
                return load_ollama_model(config.model)
            elif service == 'openrouter':
                return load_open_router(config)
            elif service == 'silicon-flow':
                return load_silicon_flow(config)
            elif service == 'zhipu':
                return load_zhipu_model(config)
            elif service == 'tongyi':
                return load_tongyi_model(config)
            elif service == 'gemini':
                return load_gemini_model(config)
            elif service == 'openai':
                return load_openai_model(config)
            elif service == 'modelscope':
                return load_modelscope_llm(config)
            else:
                # 尝试使用通用加载器
                return cls._load_generic_model(config)

        except Exception as e:
            raise RuntimeError(f"加载模型失败 - 服务: {service}, 模型: {model_name}, 错误: {str(e)}")

    @classmethod
    def load_embedding_model(cls, config: ModelConfig):
        """
        加载嵌入模型
        
        Args:
            config: 模型配置对象
            
        Returns:
            嵌入模型实例
        """
        embedding_model = config.embedding_model

        try:
            # 根据嵌入模型名称判断服务类型
            if ':' in embedding_model:  # ollama格式如 bge-m3:latest
                return load_ollama_embeddings(embedding_model)
            else:
                # 其他嵌入模型可以在这里扩展
                return load_ollama_embeddings(embedding_model)

        except Exception as e:
            raise RuntimeError(f"加载嵌入模型失败 - 模型: {embedding_model}, 错误: {str(e)}")

    @classmethod
    def _load_generic_model(cls, config: ModelConfig) -> BaseChatModel:
        """
        使用通用方法加载模型
        
        Args:
            config: 模型配置对象
            
        Returns:
            BaseChatModel: 加载的模型实例
        """
        # 构建通用参数
        params = {
            'model': config.model,
            'temperature': config.extra_params.get('temperature', 0.7),
        }

        # 根据服务类型添加特定参数
        service = config.service_type.lower()
        if service == 'openai':
            params['api_key'] = os.getenv('OPENAI_API_KEY')
        elif service == 'anthropic':
            params['api_key'] = os.getenv('ANTHROPIC_API_KEY')

        return load_chat_model(**params)

    @classmethod
    def get_supported_services(cls) -> list:
        """获取支持的模型服务列表"""
        return list(cls.SERVICE_LOADERS.keys())


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
    便捷函数：根据参数创建模型配置并加载模型
    
    Args:
        model: 模型名称
        model_service: 模型服务类型
        service_type: 服务类型
        deep_thinking_mode: 深度思考模式
        rag_enabled: 是否启用RAG
        similarity_threshold: 相似度阈值
        embedding_model: 嵌入模型名称
        model_key: 模型密钥
        embedding_model_key: 嵌入模型密钥
        model_url: 模型URL
        **kwargs: 其他参数
        
    Returns:
        tuple: (聊天模型, 嵌入模型)
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

    # 加载聊天模型
    chat_model = UnifiedModelLoader.load_chat_model(config)

    # 如果启用RAG，加载嵌入模型
    embedding_model_instance = None
    if rag_enabled:
        embedding_model_instance = UnifiedModelLoader.load_embedding_model(config)

    return chat_model, embedding_model_instance


if __name__ == '__main__':
    # 测试不同的模型服务
    test_configs = [
        {
            'model': 'qwen3:8b',
            'model_service': 'ollama'
        },
        {
            'model': 'gemini-1.5-pro',
            'model_service': 'gemini'
        },
        {
            'model': 'Qwen/QwQ-32B',
            'model_service': 'silicon-flow'
        }
    ]

    for test_config in test_configs:
        try:
            chat_model, embedding_model = create_model_from_config(**test_config)
            print(f"✅ 成功加载模型: {test_config['model_service']} - {test_config['model']}")
        except Exception as e:
            print(f"❌ 加载失败: {test_config['model_service']} - {test_config['model']}: {e}")
