import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatZhipuAI, ChatTongyi
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from modelscope.hub.api import ModelScopeConfig
from pydantic import SecretStr

from agent.llm.model_config import ModelConfig

load_dotenv(verbose=True)


def load_zhipu_model(config: ModelConfig) -> ChatZhipuAI:
    """
    加载ZhipuAI模型

    Args:
        config: 模型配置对象

    Returns:
        ChatZhipuAI: 加载的ZhipuAI模型实例
    """
    return ChatZhipuAI(
        model=config.model,
        api_key=config.model_key,
    )


def load_tongyi_model(config: ModelConfig) -> ChatTongyi:
    """
    加载通义千问模型

    Args:
        config: 模型配置对象

    Returns:
        ChatTongyi: 加载的通义千问模型实例
    """
    return ChatTongyi(
        model=config.model,
        api_key=SecretStr(config.model_key),
    )


def load_gemini_model(config: ModelConfig) -> ChatGoogleGenerativeAI:
    """
    加载Gemini模型

    Args:
        config: 模型配置对象

    Returns:
        ChatGoogleGenerativeAI: 加载的Gemini模型实例
    """
    print(f"🔧 Gemini配置: model={config.model}, api_key={config.model_key}")
    return ChatGoogleGenerativeAI(
        model=config.model,
        api_key=SecretStr(config.model_key),
        # temperature=0.7,  # 控制输出随机性（0-1）
        # max_output_tokens=1024  # 最大输出token数
    )


def load_openai_model(config: ModelConfig):
    """加载OpenAI模型"""
    print(f"🔧 OpenAI配置: model={config.model}, api_key={config.model_key}")
    return ChatOpenAI(
        model=config.model,
        api_key=SecretStr(config.model_key),
        # temperature=0.7,
        # max_tokens=1024
    )


def load_open_router(config: ModelConfig) -> ChatOpenAI:
    """
    加载OpenRouter模型

    Args:
        config: 模型配置对象

    Returns:
        ChatOpenAI: 加载的OpenRouter模型实例
    """
    api_key = SecretStr(config.model_key)
    base_url = config.model_url

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY 环境变量未设置")

    if not base_url:
        raise ValueError("OPENROUTER_API_BASE 环境变量未设置")

    print(f"🔧 OpenRouter配置: model={config.model}, base_url={base_url}")

    return ChatOpenAI(
        model=config.model,
        api_key=api_key,
        base_url=base_url,
        max_retries=2,
        timeout=30,
        model_kwargs={
            "extra_headers": {
                "HTTP-Referer": "https://localhost:8000",
                "X-Title": "XF-AI-Agent"
            }
        }
    )


def load_silicon_flow(config: ModelConfig) -> ChatOpenAI:
    """
    加载硅基流动模型

    Args:
        config: 模型配置对象

    Returns:
        ChatOpenAI: 加载的OpenRouter模型实例
    """
    return ChatOpenAI(
        model=config.model,
        api_key=SecretStr(config.model_key),
        base_url=config.model_url,
    )


def load_chat_model(
        model: str,
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0.7,
        **kwargs,
):
    """
    使用 langchain 的 init_chat_model 加载大模型

    Args:
        model (str): 模型名称（如 gpt-4、glm-4、qwen-turbo）
        api_key (str): 可选，API 密钥
        base_url (str): 可选，自定义 base_url（如 Ollama）
        temperature (float): 模型温度
        **kwargs: 其他参数

    Returns:
        ChatModel 实例
    """
    try:
        return init_chat_model(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            **kwargs
        )
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {model}\n{e}")

if __name__ == '__main__':
    # config = ModelConfig(
    #     model="deepseek/deepseek-chat-v3.1:free",
    #     model_key="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",
    #     model_url="https://openrouter.ai/api/v1",
    #     model_service="openrouter",
    #     service_type="openrouter",
    # )
    # llm = load_open_router(config)
    # config = ModelConfig(
    #     model="GLM-4.5-Flash",
    #     model_key="053f87263c043e57187968338c63384a.FFLbsjLb9yU3NP9a",
    #     model_service="zhipu",
    #     service_type="zhipu",
    # )
    # llm = load_zhipu_model(config)

    config = ModelConfig(
        model="GLM-4.5-Flash",
        model_key="053f87263c043e57187968338c63384a.FFLbsjLb9yU3NP9a",
        model_url="https://openrouter.ai/api/v1",
        model_service="zhipu",
        service_type="zhipu",
    )
    llm = load_zhipu_model(config)
    message = llm.invoke("你好")
    if isinstance(message, AIMessage):
        print("模型回复：", message.content)
    else:
        print("模型未返回 AIMessage:", message)
