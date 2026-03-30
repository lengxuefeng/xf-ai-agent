import os
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from agent.llm.model_config import ModelConfig
from config.runtime_settings import (
    CHAT_NODE_TOTAL_TIMEOUT_SEC,
    LLM_MAX_RETRIES,
    LLM_REQUEST_TIMEOUT_SEC,
    ROUTER_POLICY_CONFIG,
    WORKFLOW_REFLECTION_CONFIG,
)
from utils.custom_logger import get_logger

log = get_logger(__name__)
load_dotenv(verbose=True)


def _default_transport_timeout_sec() -> float:
    """
    传输层超时需要略大于上层业务超时，避免由底层 HTTP 提前把调用打断。
    """
    return max(
        float(LLM_REQUEST_TIMEOUT_SEC),
        float(CHAT_NODE_TOTAL_TIMEOUT_SEC) + 10.0,
        float(ROUTER_POLICY_CONFIG.router_llm_timeout_sec) + 5.0,
        float(WORKFLOW_REFLECTION_CONFIG.llm_timeout_sec) + 5.0,
    )


def _common_generation_kwargs(config: ModelConfig) -> Dict[str, Any]:
    """
    参数对齐器
    提取并格式化大模型的通用生成参数（温度、Top-P、最大 Token）。
    确保传入 LangChain 的参数类型绝对正确。
    """
    extra = config.extra_params or {}
    return {
        "temperature": float(extra.get("temperature", 0.2)),
        "top_p": float(extra.get("top_p", 1.0)),
        "max_tokens": int(extra.get("max_tokens", 2000)),
    }


def _common_transport_kwargs(config: ModelConfig) -> Dict[str, Any]:
    """
    统一模型传输层参数（超时/重试）。

    设计说明：
    1. 默认缩短超时，避免网络抖动导致链路长时间无响应；
    2. 支持通过 extra_params 与环境变量覆盖，便于线上灰度调优。
    """
    extra = config.extra_params or {}
    timeout_sec = float(
        extra.get("timeout_sec", os.getenv("LLM_REQUEST_TIMEOUT_SEC", _default_transport_timeout_sec()))
    )
    max_retries = int(
        extra.get("max_retries", os.getenv("LLM_MAX_RETRIES", LLM_MAX_RETRIES))
    )
    timeout_sec = max(3.0, min(timeout_sec, 180.0))
    max_retries = max(0, min(max_retries, 5))
    return {"timeout": timeout_sec, "max_retries": max_retries}


def load_openai_compatible_model(config: ModelConfig) -> ChatOpenAI:
    """
    【核心优化】加载所有兼容 OpenAI 接口协议的模型！
    (包含：OpenAI, OpenRouter, 硅基流动, 智谱GLM, 魔塔社区等)
    这些厂商底层的 SDK 调用方式完全一致，只要换 Base_URL 和 Key 即可，无需重复写函数。
    """
    if not config.model_key:
        log.warning(f"⚠️ 模型 {config.model} 未提供 API Key")

    api_key = SecretStr(config.model_key) if config.model_key else None

    # 针对 OpenRouter 的特殊请求头处理
    model_kwargs = {}
    if config.service_type.lower() == "openrouter":
        model_kwargs = {
            "extra_headers": {
                "HTTP-Referer": "https://localhost:8000",
                "X-Title": "XF-AI-Agent"
            }
        }

    log.info(f"初始化 OpenAI 兼容模型: {config.model} | BaseURL: {config.model_url}")
    transport_kwargs = _common_transport_kwargs(config)
    return ChatOpenAI(
        model=config.model,
        api_key=api_key,
        base_url=config.model_url or None,
        max_retries=transport_kwargs["max_retries"],
        timeout=transport_kwargs["timeout"],
        model_kwargs=model_kwargs,
        **_common_generation_kwargs(config),
    )


def load_openai_embeddings(config: ModelConfig) -> OpenAIEmbeddings:
    """加载兼容 OpenAI 接口协议的嵌入模型 (Embeddings)"""
    api_key = SecretStr(config.embedding_model_key) if config.embedding_model_key else None
    transport_kwargs = _common_transport_kwargs(config)
    
    log.info(f"初始化 OpenAI 兼容 Embedding 模型: {config.embedding_model} | BaseURL: {config.model_url}")
    return OpenAIEmbeddings(
        model=config.embedding_model,
        api_key=api_key,
        base_url=config.model_url or None,
        timeout=transport_kwargs["timeout"],
        max_retries=transport_kwargs["max_retries"],
    )


def load_tongyi_model(config: ModelConfig) -> ChatTongyi:
    """单独加载通义千问 (由于其 SDK 包依赖不同)"""
    log.info(f"初始化通义千问模型: {config.model}")
    return ChatTongyi(
        model=config.model,
        api_key=SecretStr(config.model_key),
        **_common_generation_kwargs(config)
    )


def load_gemini_model(config: ModelConfig) -> ChatGoogleGenerativeAI:
    """单独加载 Gemini 模型"""
    log.info(f"初始化 Gemini 模型: {config.model}")
    kwargs = _common_generation_kwargs(config)
    return ChatGoogleGenerativeAI(
        model=config.model,
        api_key=SecretStr(config.model_key),
        temperature=kwargs["temperature"],
        top_p=kwargs["top_p"],
        max_output_tokens=kwargs["max_tokens"],
    )
