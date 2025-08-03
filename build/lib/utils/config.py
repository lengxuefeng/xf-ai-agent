from dotenv import load_dotenv
import os

# 加载 .env 文件中的环境变量
load_dotenv()

class Settings:
    """
    项目配置类，从环境变量中加载配置。
    """
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "") # 提供默认空字符串

    # 向量存储目录
    VECTORSTORE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "vectorstore")
    # 知识库数据目录
    KNOWLEDGE_DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

settings = Settings()
