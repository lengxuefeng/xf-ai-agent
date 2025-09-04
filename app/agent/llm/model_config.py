# -*- coding: utf-8 -*-
"""
模型配置类

定义模型加载所需的配置参数。
"""


class ModelConfig:
    """模型配置类"""

    def __init__(
            self,
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
    ):
        self.model = model
        self.model_service = model_service
        self.service_type = service_type
        self.deep_thinking_mode = deep_thinking_mode
        self.rag_enabled = rag_enabled
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.extra_params = kwargs
        self.model_key = model_key
        self.model_url = model_url
        self.embedding_model_key = embedding_model_key
