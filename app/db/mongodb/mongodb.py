import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

load_dotenv()


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

        self.client = MongoClient(mongo_url)
        self.db: Database = self.client[db_name]

    def get_db(self) -> Database:
        """
        返回数据库对象实例。
        """
        return self.db


# 全局数据库实例
mongodb_db = MongoDatabase()
