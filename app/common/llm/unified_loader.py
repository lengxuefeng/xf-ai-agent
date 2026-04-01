# -*- coding: utf-8 -*-
"""
统一模型加载器

根据模型服务类型和配置参数动态加载相应的模型实例。
"""

import os
from typing import Optional, Any
from enum import Enum

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

from common.utils.custom_logger import get_logger
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

FAST_MODEL_CANDIDATES = {
    "zhipu": ("glm-4-flash", "glm-4-air"),
    "openai": ("gpt-4o-mini",),
    "openrouter": ("openai/gpt-4o-mini", "gpt-4o-mini"),
    "netlify-gemini": ("gpt-4o-mini",),
    "silicon-flow": ("Qwen/Qwen2.5-7B-Instruct",),
    "modelscope": ("Qwen/Qwen2.5-7B-Instruct",),
    "tongyi": ("qwen-turbo",),
    "gemini": ("gemini-2.0-flash", "gemini-1.5-flash"),
    "ollama": (),
}

LARGE_MODEL_CANDIDATES = {
    "zhipu": ("glm-4.7", "glm-4-plus"),
    "openai": ("gpt-4.1", "gpt-4o"),
    "openrouter": ("openai/gpt-4.1", "openai/gpt-4o"),
    "netlify-gemini": ("gpt-4o",),
    "silicon-flow": ("deepseek-ai/DeepSeek-V3",),
    "modelscope": ("Qwen/Qwen2.5-72B-Instruct",),
    "tongyi": ("qwen-max",),
    "gemini": ("gemini-2.5-pro", "gemini-1.5-pro"),
    "ollama": (),
}


class ProviderFamily(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GENERIC = "generic"


def resolve_provider_family(
    *,
    service_type: str = "",
    model_service: str = "",
    model_name: str = "",
) -> ProviderFamily:
    tokens = " ".join(
        [
            str(service_type or "").strip().lower(),
            str(model_service or "").strip().lower(),
            str(model_name or "").strip().lower(),
        ]
    )
    if any(keyword in tokens for keyword in ("anthropic", "claude")):
        return ProviderFamily.ANTHROPIC
    if (
        "openai" in tokens
        or "/gpt" in tokens
        or " gpt" in tokens
        or tokens.startswith("gpt")
        or "/o1" in tokens
        or "/o3" in tokens
        or "/o4" in tokens
        or tokens.startswith("o1")
        or tokens.startswith("o3")
        or tokens.startswith("o4")
    ):
        return ProviderFamily.OPENAI
    return ProviderFamily.GENERIC


def _normalize_model_size(model_size: str | None) -> str:
    normalized = str(model_size or "large").strip().lower()
    return "fast" if normalized == "fast" else "large"


def _resolve_model_name(service_type: str, requested_model: str, model_size: str) -> str:
    service = str(service_type or "").strip().lower()
    requested = str(requested_model or "").strip()
    if model_size == "fast":
        candidates = FAST_MODEL_CANDIDATES.get(service, ())
        return candidates[0] if candidates else requested
    if requested:
        return requested
    candidates = LARGE_MODEL_CANDIDATES.get(service, ())
    return candidates[0] if candidates else requested


def _clone_config(config: ModelConfig, **updates) -> ModelConfig:
    return config.model_copy(update=updates)


def _build_fallback_configs(config: ModelConfig, *, primary_model_name: str) -> list[ModelConfig]:
    extra = config.extra_params or {}
    service = config.service_type.lower()
    model_size = _normalize_model_size(getattr(config, "model_size", extra.get("model_size", "large")))
    fallback_configs: list[ModelConfig] = []

    explicit_fallback_model = str(extra.get("fallback_model") or os.getenv("LLM_FALLBACK_MODEL", "")).strip()
    if explicit_fallback_model and explicit_fallback_model != primary_model_name:
        fallback_configs.append(
            _clone_config(
                config,
                model=explicit_fallback_model,
                service_type=str(extra.get("fallback_service_type") or os.getenv("LLM_FALLBACK_SERVICE_TYPE", config.service_type)).strip() or config.service_type,
                model_service=str(extra.get("fallback_model_service") or os.getenv("LLM_FALLBACK_MODEL_SERVICE", config.model_service)).strip() or config.model_service,
                model_key=str(extra.get("fallback_model_key") or os.getenv("LLM_FALLBACK_MODEL_KEY", config.model_key)).strip() or config.model_key,
                model_url=str(extra.get("fallback_model_url") or os.getenv("LLM_FALLBACK_MODEL_URL", config.model_url)).strip() or config.model_url,
            )
        )

    service_fast_candidates = FAST_MODEL_CANDIDATES.get(service, ())
    for candidate in service_fast_candidates:
        if candidate and candidate != primary_model_name:
            fallback_configs.append(_clone_config(config, model=candidate, model_size="fast"))

    original_model = str(extra.get("requested_model") or config.model or "").strip()
    if model_size == "fast" and original_model and original_model != primary_model_name:
        fallback_configs.append(_clone_config(config, model=original_model, model_size="large"))

    deduped: list[ModelConfig] = []
    seen: set[tuple[str, str, str]] = set()
    for item in fallback_configs:
        key = (str(item.service_type).lower(), str(item.model).strip(), str(item.model_url or "").strip())
        if not key[1] or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


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
        model_size: str = "large",
        deep_thinking_mode: str = 'auto',
        rag_enabled: bool = False,
        similarity_threshold: float = 0.7,
        embedding_model: str = 'bge-m3:latest',
        model_key: str = '',
        model_url: str = '',
        embedding_model_key: str = '',
        strict_embedding_init: bool = False,
        **kwargs
) -> tuple[BaseChatModel, Optional[Any]]:
    """
    【对外接口】封装层，提供给 Supervisor 和 Agent 实例化模型使用。
    """
    normalized_model_size = _normalize_model_size(model_size)
    resolved_model = _resolve_model_name(service_type, model, normalized_model_size)
    provider_family = resolve_provider_family(
        service_type=service_type,
        model_service=model_service,
        model_name=resolved_model or model,
    )
    config = ModelConfig(
        model=resolved_model,
        model_service=model_service,
        service_type=service_type,
        model_size=normalized_model_size,
        deep_thinking_mode=deep_thinking_mode,
        rag_enabled=rag_enabled,
        similarity_threshold=similarity_threshold,
        embedding_model=embedding_model,
        model_key=model_key,
        model_url=model_url,
        embedding_model_key=embedding_model_key,
        requested_model=model,
        provider_family=provider_family.value,
        **kwargs
    )

    chat_model = UnifiedModelLoader.load_chat_model(config)
    fallback_models = []
    for fallback_config in _build_fallback_configs(config, primary_model_name=resolved_model):
        try:
            fallback_models.append(UnifiedModelLoader.load_chat_model(fallback_config))
        except Exception as exc:
            log.warning(
                "跳过不可用的 fallback 模型: service=%s model=%s detail=%s",
                fallback_config.service_type,
                fallback_config.model,
                exc,
            )
    if fallback_models:
        chat_model = chat_model.with_fallbacks(fallback_models)
    embedding_model_instance = None
    if rag_enabled:
        try:
            embedding_model_instance = UnifiedModelLoader.load_embedding_model(config)
        except Exception as exc:
            if strict_embedding_init:
                raise
            # 当前主链路只依赖 chat_model，Embedding 初始化失败时直接降级禁用，
            # 避免用户请求被无关的 RAG 预热异常整体打断。
            log.warning(
                f"Embedding 初始化失败，已降级跳过: "
                f"service={service_type}, model={embedding_model}, detail={exc}"
            )

    return chat_model, embedding_model_instance


def create_embedding_model_from_config(
        model: str,
        model_service: str,
        service_type: str,
        model_size: str = "large",
        deep_thinking_mode: str = 'auto',
        rag_enabled: bool = False,
        similarity_threshold: float = 0.7,
        embedding_model: str = 'bge-m3:latest',
        model_key: str = '',
        model_url: str = '',
        embedding_model_key: str = '',
        **kwargs
):
    normalized_model_size = _normalize_model_size(model_size)
    resolved_model = _resolve_model_name(service_type, model, normalized_model_size)
    provider_family = resolve_provider_family(
        service_type=service_type,
        model_service=model_service,
        model_name=resolved_model or model,
    )
    config = ModelConfig(
        model=resolved_model,
        model_service=model_service,
        service_type=service_type,
        model_size=normalized_model_size,
        deep_thinking_mode=deep_thinking_mode,
        rag_enabled=rag_enabled,
        similarity_threshold=similarity_threshold,
        embedding_model=embedding_model,
        model_key=model_key,
        model_url=model_url,
        embedding_model_key=embedding_model_key,
        requested_model=model,
        provider_family=provider_family.value,
        **kwargs
    )
    return UnifiedModelLoader.load_embedding_model(config)
