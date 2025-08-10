import os

from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()


class Settings:
    """
    项目配置类，从环境变量中加载配置。
    """
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")  # 提供默认空字符串

    # 1. 文件路径
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "faiss_index")
    # 对话历史文件
    HISTORY_FILE: str = os.getenv("HISTORY_FILE", "chat_history.json")

    # 2. 模型配置
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "qwen3:8b")
    AVAILABLE_MODELS: list[str] = ["deepseek-r1:8b", "qwen3:8b", "gemma2:2b", "llama3.1:latest"]

    # embedding 模型名称
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
    AVAILABLE_EMBEDDING_MODELS: list[str] = ["bge-m3:latest", "nomic-embed-text:latest", "mxbai-embed-large:latest",
                                             "bge-large-en-v1.5:latest", "bge-large-zh-v1.5:latest"]
    # embedding 模型基础url,默认ollama地址
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434")

    # 3. RAG配置
    DEFAULT_SIMILARITY_THRESHOLD: float = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", 0.7))
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", 1000))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 200))
    MAX_RETRIEVED_DOCS: int = int(os.getenv("MAX_RETRIEVED_DOCS", 3))

    # 4. 高德地图MCP key
    AMAP_API_KEY: str = os.getenv("AMAP_API_KEY", "d3099bc08ff2f92edad48d2537984adc")

    # 5. LangChain配置
    # 文本分割器chunk大小
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    # 文本分割器重叠长度
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))
    # 文本分割器分隔符
    SEPARATORS: list[str] = os.getenv("SEPARATORS", ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]).split(",")

    # 6. 对话历史配置
    MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", 5))

    # 向量存储目录
    VECTORSTORE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "vectorstore")
    # 知识库数据目录
    KNOWLEDGE_DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

    # 目前项目支持的模型类型
    SUPPORTED_MODEL_TYPES: list[str] = os.getenv("SUPPORTED_MODEL_TYPES",
                                                 [["ollama", "openRouter", "chat", "zhipu", "qwen"]])


settings = Settings()
