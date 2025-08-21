import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatZhipuAI, ChatTongyi
from langchain_openai import ChatOpenAI

load_dotenv(verbose=True)


def load_zhipu_model(model_name: str) -> ChatZhipuAI:
    """
    加载ZhipuAI模型

    Args:
        model_name: 模型名称，如 GLM-4-Flash

    Returns:
        ChatZhipuAI: 加载的ZhipuAI模型实例
    """
    return ChatZhipuAI(
        model=model_name,
        api_key=os.getenv("ZHIPUAI_API_KEY"),
    )


def load_tongyi_model(model_name: str) -> ChatTongyi:
    """
    加载ZhipuAI模型

    Args:
        model_name: 模型名称，如 GLM-4-Flash

    Returns:
        ChatZhipuAI: 加载的ZhipuAI模型实例
    """
    return ChatTongyi(
        model=model_name,
        api_key=os.getenv("ZHIPUAI_API_KEY"),
    )


def load_open_router(model_name: str) -> ChatOpenAI:
    """
    加载OpenRouter模型

    Args:
        model_name: 模型名称，如 gpt-4、gpt-4o

    Returns:
        ChatOpenAI: 加载的OpenRouter模型实例
    """
    return ChatOpenAI(
        model=model_name,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_API_BASE"),
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

# if __name__ == '__main__':
#     llm = load_open_router("qwen/qwen3-coder:free")
#     message = llm.invoke("你好")
#     if isinstance(message, AIMessage):
#         print("模型回复：", message.content)
#     else:
#         print("模型未返回 AIMessage:", message)
