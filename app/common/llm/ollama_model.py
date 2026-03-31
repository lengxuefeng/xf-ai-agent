from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings

from common.utils.config import Settings

"""
根据提供的模型名称加载相应的Ollama模型
"""


def load_ollama_model(model_name: str) -> ChatOllama:
    """加载ollama模型

    Args:
        model_name: 模型名称

    Returns:
        ChatOllama: 加载的ollama模型

    """
    return ChatOllama(model=model_name)


def load_ollama_embeddings(model_name: str) -> OllamaEmbeddings:
    """加载ollama嵌入模型
    嵌入模型解决 “如何将数据转换为有意义的向量”。
    例如，文本可以转换为向量，以便计算机可以理解和处理。

    Args:
        model_name: 模型名称

    Returns:
        OllamaEmbeddings: 加载的ollama嵌入模型

    """
    return OllamaEmbeddings(model=model_name, base_url=Settings.EMBEDDING_BASE_URL)


def load_ollama_vector_storr(model_name: str) -> InMemoryVectorStore:
    """创建内存向量存储模型
    向量存储模型解决 “如何存储和检索向量”。
    例如，文本可以转换为向量，然后可以存储在内存中，以便快速检索。
    向量存储模型可以用于各种应用，如文档检索、图像搜索等。

    Args:
        model_name: 用于创建嵌入模型的模型名称

    Returns:
        InMemoryVectorStore: 内存向量存储实例，用于存储和查询嵌入向量的文本

    """
    embeddings = load_ollama_embeddings(model_name)
    return InMemoryVectorStore(embedding=embeddings)
