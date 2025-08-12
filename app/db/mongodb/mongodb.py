import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

load_dotenv()


class MongoDB:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_db()
        return cls._instance

    def _init_db(self):
        self.client = MongoClient(os.getenv("MONGODB_URL"))
        self.db: Database = self.client[os.getenv("MONGODB_DB_NAME")]

    def get_collection(self, collection_name: str) -> Collection:
        """获取集合（表）对象"""
        return self.db[collection_name]

    async def insert_one(self, collection_name: str, data: Dict) -> str:
        """插入单条数据，返回插入的_id"""
        collection = self.get_collection(collection_name)
        result = collection.insert_one(data)
        return str(result.inserted_id)

    async def insert_many(self, collection_name: str, data: List[Dict]) -> List[str]:
        """批量插入数据，返回插入的_id列表"""
        collection = self.get_collection(collection_name)
        result = collection.insert_many(data)
        return [str(_id) for _id in result.inserted_ids]

    async def find_one(self, collection_name: str, query: Dict) -> Optional[Dict]:
        """查询单条数据"""
        collection = self.get_collection(collection_name)
        data = collection.find_one(query)
        if data:
            data["_id"] = str(data["_id"])  # 将ObjectId转为字符串
        return data

    async def find_many(
            self,
            collection_name: str,
            query: Dict = {},
            skip: int = 0,
            limit: int = 10,
            sort: List[tuple] = None
    ) -> List[Dict]:
        """分页查询数据"""
        collection = self.get_collection(collection_name)
        cursor = collection.find(query).skip(skip).limit(limit)
        if sort:
            cursor = cursor.sort(sort)
        return [{**item, "_id": str(item["_id"])} for item in cursor]

    async def update_one(
            self,
            collection_name: str,
            query: Dict,
            update_data: Dict,
            upsert: bool = False
    ) -> int:
        """更新单条数据，返回匹配的文档数"""
        collection = self.get_collection(collection_name)
        result = collection.update_one(query, {"$set": update_data}, upsert=upsert)
        return result.modified_count

    async def delete_one(self, collection_name: str, query: Dict) -> int:
        """删除单条数据，返回删除的文档数"""
        collection = self.get_collection(collection_name)
        result = collection.delete_one(query)
        return result.deleted_count

    async def count_documents(self, collection_name: str, query: Dict = {}) -> int:
        """统计文档数量"""
        collection = self.get_collection(collection_name)
        return collection.count_documents(query)


# 全局单例
mongodb = MongoDB()
