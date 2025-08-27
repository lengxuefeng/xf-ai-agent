from typing import List, Dict, Any, Optional, Tuple
from bson import ObjectId
from datetime import datetime

from db.mongodb.mongodb_base import MongoCRUDBase
from schemas.chat_history_schemas import ChatHistory, ChatHistoryCreate, ChatHistoryUpdate


class CRUDChatHistory(MongoCRUDBase[ChatHistory, ChatHistoryCreate, ChatHistoryUpdate]):
    """
    聊天历史数据库访问层
    """
    
    def __init__(self):
        super().__init__(collection_name="chat_history", model=ChatHistory)
        self._create_indexes()
    
    def _create_indexes(self):
        """
        为聊天历史集合创建索引
        """
        existing_indexes = [index['name'] for index in self.collection.list_indexes()]
        
        indexes_to_create = {
            "user_id_1": [("user_id", 1)],
            "session_id_1": [("session_id", 1)],
            "created_at_-1": [("created_at", -1)],
            "user_id_1_session_id_1": [("user_id", 1), ("session_id", 1)],
            "user_id_1_created_at_-1": [("user_id", 1), ("created_at", -1)],
            "is_deleted_1": [("is_deleted", 1)]
        }
        
        for index_name, index_keys in indexes_to_create.items():
            if index_name not in existing_indexes:
                self.collection.create_index(index_keys, name=index_name)
    
    def get_by_user_and_id(self, user_id: int, chat_id: str) -> Optional[ChatHistory]:
        """
        根据用户ID和聊天ID获取单条记录
        """
        if not ObjectId.is_valid(chat_id):
            return None
        
        query = {
            "_id": ObjectId(chat_id),
            "user_id": user_id,
            "is_deleted": 0
        }
        return self.get(query=query)
    
    def get_user_chat_list(
        self, 
        user_id: int, 
        session_id: Optional[str] = None,
        page: int = 1, 
        size: int = 10
    ) -> Tuple[List[ChatHistory], int, int, int]:
        """
        获取用户聊天列表（分页）
        """
        query = {"user_id": user_id, "is_deleted": 0}
        if session_id:
            query["session_id"] = session_id
        
        sort = [("created_at", -1)]  # 按创建时间降序
        return self.paginate(query=query, page=page, size=size, sort=sort)
    
    def search_chat_history(
        self,
        user_id: int,
        keyword: str,
        session_id: Optional[str] = None,
        page: int = 1,
        size: int = 10
    ) -> Tuple[List[ChatHistory], int, int, int]:
        """
        搜索聊天历史
        """
        query = {
            "user_id": user_id,
            "is_deleted": 0,
            "$or": [
                {"user_content": {"$regex": keyword, "$options": "i"}},
                {"model_content": {"$regex": keyword, "$options": "i"}},
                {"title": {"$regex": keyword, "$options": "i"}}
            ]
        }
        
        if session_id:
            query["session_id"] = session_id
        
        sort = [("created_at", -1)]
        return self.paginate(query=query, page=page, size=size, sort=sort)
    
    def update_by_user_and_id(
        self, 
        user_id: int, 
        chat_id: str, 
        update_data: ChatHistoryUpdate
    ) -> Optional[ChatHistory]:
        """
        更新指定用户的聊天记录
        """
        if not ObjectId.is_valid(chat_id):
            return None
        
        query = {
            "_id": ObjectId(chat_id),
            "user_id": user_id,
            "is_deleted": 0
        }
        return self.update(query=query, obj_in=update_data)
    
    def soft_delete_by_user_and_id(self, user_id: int, chat_id: str) -> bool:
        """
        软删除指定用户的聊天记录
        """
        if not ObjectId.is_valid(chat_id):
            return False
        
        query = {
            "_id": ObjectId(chat_id),
            "user_id": user_id,
            "is_deleted": 0
        }
        
        result = self.collection.update_one(query, {"$set": {"is_deleted": 1}})
        return result.modified_count > 0
    
    def hard_delete_by_user_and_id(self, user_id: int, chat_id: str) -> bool:
        """
        硬删除指定用户的聊天记录
        """
        if not ObjectId.is_valid(chat_id):
            return False
        
        query = {
            "_id": ObjectId(chat_id),
            "user_id": user_id
        }
        return self.delete(query=query)
    
    def delete_session(self, user_id: int, session_id: str, hard_delete: bool = False) -> int:
        """
        删除整个会话
        """
        if hard_delete:
            query = {"session_id": session_id, "user_id": user_id}
            result = self.collection.delete_many(query)
            return result.deleted_count
        else:
            query = {"session_id": session_id, "user_id": user_id, "is_deleted": 0}
            result = self.collection.update_many(query, {"$set": {"is_deleted": 1}})
            return result.modified_count
    
    def get_user_sessions(self, user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """
        获取用户的会话列表
        """
        pipeline = [
            {"$match": {"user_id": user_id, "is_deleted": 0}},
            {"$sort": {"created_at": -1}},
            {
                "$group": {
                    "_id": "$session_id",
                    "title": {"$first": "$title"},
                    "latest_message": {"$first": "$user_content"},
                    "latest_response": {"$first": "$model_content"},
                    "created_at": {"$first": "$created_at"},
                    "updated_at": {"$max": "$created_at"},
                    "message_count": {"$sum": 1}
                }
            },
            {"$sort": {"updated_at": -1}},
            {"$limit": limit}
        ]
        
        return list(self.collection.aggregate(pipeline))
    
    def get_user_statistics(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """
        获取用户聊天统计信息
        """
        from datetime import timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        pipeline = [
            {
                "$match": {
                    "user_id": user_id,
                    "is_deleted": 0,
                    "created_at": {"$gte": start_date, "$lte": end_date}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_messages": {"$sum": 1},
                    "total_tokens": {"$sum": "$tokens"},
                    "total_sessions": {"$addToSet": "$session_id"},
                    "avg_latency": {"$avg": "$latency_ms"},
                    "models_used": {"$addToSet": "$model"}
                }
            }
        ]
        
        stats = list(self.collection.aggregate(pipeline))
        return stats[0] if stats else {}
    
    def create_with_user_id(self, user_id: int, obj_in: ChatHistoryCreate) -> Optional[ChatHistory]:
        """
        创建聊天记录（自动设置用户ID）
        """
        chat_data = obj_in.model_dump(exclude_unset=True)
        chat_data['user_id'] = user_id
        chat_data['is_deleted'] = 0
        chat_data['created_at'] = datetime.now()
        
        # 直接插入数据库
        result = self.collection.insert_one(chat_data)
        inserted_doc = self.collection.find_one({"_id": result.inserted_id})
        return self._to_model(inserted_doc)
    
    def get_by_session_id(self, *, session_id: str, limit: int = 100) -> List[ChatHistory]:
        """
        根据 session_id 获取聊天记录。
        """
        return self.get_multi(query={"session_id": session_id}, limit=limit, sort=[("created_at", 1)])


# 创建全局实例
chat_history_db = CRUDChatHistory()