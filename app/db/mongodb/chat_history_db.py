from typing import List

from app.db.mongodb.mongodb_base import MongoCRUDBase
from app.schemas.chat_history_schemas import ChatHistory, ChatHistoryCreate, ChatHistoryUpdate


class CRUDChatHistory(MongoCRUDBase[ChatHistory, ChatHistoryCreate, ChatHistoryUpdate]):
    def __init__(self):
        """
        聊天记录的CRUD操作类。
        """
        super().__init__(collection_name="chat_history", model=ChatHistory)
        self._create_indexes()

    def _create_indexes(self):
        """
        如果索引不存在，则为chat_history集合创建索引。
        这使得索引创建操作是幂等的。
        """
        existing_indexes = [index['name'] for index in self.collection.list_indexes()]

        indexes_to_create = {
            "session_id_1": [("session_id", 1)],
            "user_id_1": [("user_id", 1)],
            "created_at_-1": [("created_at", -1)],
            "user_id_1_created_at_-1": [("user_id", 1), ("created_at", -1)],
        }

        for index_name, index_keys in indexes_to_create.items():
            if index_name not in existing_indexes:
                self.collection.create_index(index_keys, name=index_name)

    def get_by_session_id(self, *, session_id: str, limit: int = 100) -> List[ChatHistory]:
        """
        根据 session_id 获取聊天记录。

        :param session_id: 会话ID。
        :param limit: 返回的最大记录数。
        :return: 聊天记录列表。
        """
        return self.get_multi(query={"session_id": session_id}, limit=limit, sort=[("created_at", 1)])


# CRUDChatHistory的单例
chat_history_db = CRUDChatHistory()
