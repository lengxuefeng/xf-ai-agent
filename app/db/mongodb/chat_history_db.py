from typing import List, Optional, Tuple

from db.mongodb.mongodb_base import MongoCRUDBase
from schemas.chat_history_schemas import (
    ChatSession, ChatSessionCreate, ChatSessionUpdate,
    ChatMessage, ChatMessageCreate, ChatMessageUpdate
)


class CRUDChatSession(MongoCRUDBase[ChatSession, ChatSessionCreate, ChatSessionUpdate]):
    """
    聊天会话数据库访问层
    """
    def __init__(self):
        super().__init__(collection_name="chat_sessions", model=ChatSession)
        self._create_indexes()

    def _create_indexes(self):
        existing_indexes = [index['name'] for index in self.collection.list_indexes()]
        indexes_to_create = {
            "user_id_1": [("user_id", 1)],
            "session_id_1": [("session_id", 1)],
            "user_id_1_updated_at_-1": [("user_id", 1), ("updated_at", -1)],
        }
        for index_name, index_keys in indexes_to_create.items():
            if index_name not in existing_indexes:
                self.collection.create_index(index_keys, name=index_name)

    def create_session(self, user_id: int, session_id: str, title: str) -> ChatSession:
        session_data = ChatSessionCreate(
            user_id=user_id,
            session_id=session_id,
            title=title
        )
        return self.create(session_data)

    def get_user_sessions(self, user_id: int, page: int = 1, size: int = 20) -> Tuple[List[ChatSession], int, int, int]:
        query = {"user_id": user_id, "is_deleted": 0}
        sort = [("updated_at", -1)]
        return self.paginate(query=query, page=page, size=size, sort=sort)

    def update_session_title(self, user_id: int, session_id: str, new_title: str) -> int:
        query = {"user_id": user_id, "session_id": session_id, "is_deleted": 0}
        update_data = ChatSessionUpdate(title=new_title)
        result = self.collection.update_one(query, {"$set": update_data.model_dump(exclude_unset=True)})
        return result.modified_count

    def delete_session(self, user_id: int, session_id: str, hard_delete: bool = False) -> int:
        query = {"user_id": user_id, "session_id": session_id}
        if hard_delete:
            result = self.collection.delete_one(query)
            return result.deleted_count
        else:
            result = self.collection.update_one(query, {"$set": {"is_deleted": 1}})
            return result.modified_count


class CRUDChatMessage(MongoCRUDBase[ChatMessage, ChatMessageCreate, ChatMessageUpdate]):
    """
    聊天消息数据库访问层
    """
    def __init__(self):
        super().__init__(collection_name="chat_messages", model=ChatMessage)
        self._create_indexes()

    def _create_indexes(self):
        existing_indexes = [index['name'] for index in self.collection.list_indexes()]
        indexes_to_create = {
            "user_id_1": [("user_id", 1)],
            "session_id_1": [("session_id", 1)],
            "created_at_-1": [("created_at", -1)],
            "user_id_1_session_id_1": [("user_id", 1), ("session_id", 1)],
        }
        for index_name, index_keys in indexes_to_create.items():
            if index_name not in existing_indexes:
                self.collection.create_index(index_keys, name=index_name)

    def create_with_user_id(self, user_id: int, obj_in: ChatMessageCreate) -> Optional[ChatMessage]:
        chat_data = obj_in.model_dump(exclude_unset=True)
        chat_data['user_id'] = user_id
        return self.create(chat_data)

    def get_session_messages(self, user_id: int, session_id: str, page: int = 1, size: int = 50) -> Tuple[List[ChatMessage], int, int, int]:
        query = {"user_id": user_id, "session_id": session_id, "is_deleted": 0}
        sort = [("created_at", 1)]
        return self.paginate(query=query, page=page, size=size, sort=sort)
        
    def delete_messages_by_session(self, user_id: int, session_id: str, hard_delete: bool = False) -> int:
        query = {"user_id": user_id, "session_id": session_id}
        if hard_delete:
            result = self.collection.delete_many(query)
            return result.deleted_count
        else:
            query["is_deleted"] = 0
            result = self.collection.update_many(query, {"$set": {"is_deleted": 1}})
            return result.modified_count


# 创建全局实例
chat_session_db = CRUDChatSession()
chat_message_db = CRUDChatMessage()
