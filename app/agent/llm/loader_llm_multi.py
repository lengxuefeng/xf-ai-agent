import os
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from agent.llm.model_config import ModelConfig
from utils.custom_logger import get_logger

log = get_logger(__name__)
load_dotenv(verbose=True)


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
    return ChatOpenAI(
        model=config.model,
        api_key=api_key,
        base_url=config.model_url or None,
        max_retries=2,
        timeout=60,
        model_kwargs=model_kwargs,
        **_common_generation_kwargs(config),
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