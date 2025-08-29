from typing import Dict, Any, Optional

from fastapi import HTTPException

from db.mongodb.chat_history_db import chat_session_db, chat_message_db
from schemas.chat_history_schemas import (
    ChatSession, ChatSessionCreate, ChatMessage, ChatMessageCreate
)
from utils.chat_utils import ChatUtils


class ChatHistoryService:
    """
    聊天历史服务层
    """

    def __init__(self):
        self.session_db = chat_session_db
        self.message_db = chat_message_db

    def create_session(self, user_id: int, title: Optional[str] = None) -> ChatSession:
        """
        创建一个新的聊天会话
        """
        session_id = ChatUtils.generate_session_id()
        if not title:
            title = "新对话"

        session_create = ChatSessionCreate(
            user_id=user_id,
            session_id=session_id,
            title=title
        )
        return self.session_db.create(session_create)

    def create_chat_message(self, user_id: int, chat_data: ChatMessageCreate) -> ChatMessage:
        """
        创建聊天记录, 并在需要时更新会话标题
        """
        # 检查会话是否存在
        session = self.session_db.get(query={"user_id": user_id, "session_id": chat_data.session_id})
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")

        # 如果是新对话，且用户提供了足够内容，则更新标题
        if session.title == "新对话" and len(chat_data.user_content) > 5:
            new_title = chat_data.user_content[:50]  # 使用用户输入的前50个字符作为标题
            self.session_db.update_session_title(user_id, chat_data.session_id, new_title)

        return self.message_db.create_with_user_id(user_id, chat_data)

    def get_user_sessions(self, user_id: int, page: int = 1, size: int = 20) -> Dict[str, Any]:
        """
        获取用户会话列表（分页）
        """
        items, total, current_page, pages = self.session_db.get_user_sessions(user_id, page, size)

        session_list = [item.model_dump(exclude={'_id'}) for item in items]
        for session in session_list:
            session['id'] = str(items[session_list.index(session)].id)

        return {
            "items": session_list,
            "total": total,
            "page": current_page,
            "size": size,
            "pages": pages
        }

    def get_session_messages(self, user_id: int, session_id: str, page: int = 1, size: int = 50) -> Dict[str, Any]:
        """
        获取指定会话的聊天记录详情
        """
        session = self.session_db.get(query={"user_id": user_id, "session_id": session_id})
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")

        items, total, current_page, pages = self.message_db.get_session_messages(user_id, session_id, page, size)

        message_list = [item.model_dump(exclude={'_id'}) for item in items]
        for msg in message_list:
            msg['id'] = str(items[message_list.index(msg)].id)

        return {
            "session_id": session_id,
            "title": session.title,
            "messages": message_list,
            "total": total,
            "page": current_page,
            "size": size,
            "pages": pages
        }

    def delete_session(self, user_id: int, session_id: str, hard_delete: bool = False) -> Dict[str, Any]:
        """
        删除整个会话及相关消息
        """
        # TODO: 考虑使用事务
        session_deleted_count = self.session_db.delete_session(user_id, session_id, hard_delete)

        if session_deleted_count == 0:
            raise HTTPException(status_code=404, detail="会话不存在或已被删除")

        messages_deleted_count = self.message_db.delete_messages_by_session(user_id, session_id, hard_delete)

        return {
            "session_deleted": session_deleted_count > 0,
            "messages_deleted_count": messages_deleted_count,
            "hard_delete": hard_delete
        }

    def update_session_title(self, user_id: int, session_id: str, new_title: str) -> Dict[str, Any]:
        """
        更新会话标题
        """
        updated_count = self.session_db.update_session_title(user_id, session_id, new_title)
        if updated_count == 0:
            raise HTTPException(status_code=404, detail="会话不存在或标题更新失败")

        return {"updated_count": updated_count, "new_title": new_title}


# 创建全局实例
chat_history_service = ChatHistoryService()
