import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database
import logging

load_dotenv()
logger = logging.getLogger(__name__)


class MongoDatabase:
    """
    MongoDB数据库连接管理的单例类。
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_db()
        return cls._instance

    def _init_db(self):
        """
        初始化MongoDB客户端和数据库对象。
        从环境变量中读取MONGODB_URL和MONGODB_DB_NAME。
        """
        mongo_url = os.getenv("MONGODB_URL")
        db_name = os.getenv("MONGODB_DB_NAME")

        if not mongo_url or not db_name:
            raise ValueError("环境变量 MONGODB_URL 和 MONGODB_DB_NAME 必须被设置。")

        # 创建 MongoDB 客户端，从环境变量读取连接池参数
        self.client = MongoClient(
            mongo_url,
            maxPoolSize=int(os.getenv("MONGODB_MAX_POOL_SIZE", 50)),
            minPoolSize=int(os.getenv("MONGODB_MIN_POOL_SIZE", 10)),
            socketTimeoutMS=int(os.getenv("MONGODB_SOCKET_TIMEOUT_MS", 60000)),
            maxIdleTimeMS=int(os.getenv("MONGODB_MAX_IDLE_TIME_MS", 60000)),
            heartbeatFrequencyMS=int(os.getenv("MONGODB_HEARTBEAT_FREQUENCY_MS", 20000)),
            serverSelectionTimeoutMS=int(os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", 5000)),
            connectTimeoutMS=int(os.getenv("MONGODB_CONNECT_TIMEOUT_MS", 10000)),
        )

        self.db: Database = self.client[db_name]

        # 连接测试
        try:
            # 通过 ping 命令测试连接
            self.client.admin.command('ismaster')
            logger.info(f"MongoDB 连接成功: {mongo_url} -> {db_name}")
        except Exception as e:
            logger.error(f"MongoDB 连接失败: {e}")
            raise

    def get_db(self) -> Database:
        """
        返回数据库对象实例。
        """
        return self.db

    def get_client(self) -> MongoClient:
        """
        返回 MongoDB 客户端实例。
        """
        return self.client

    def close_connection(self):
        """
        关闭 MongoDB 连接。
        """
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("MongoDB 连接已关闭")


# 全局数据库实例
mongodb_db = MongoDatabase()
