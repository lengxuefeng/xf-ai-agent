# -*- coding: utf-8 -*-
import uuid
from sqlalchemy.orm import Session
from fastapi import HTTPException
from db.crud import chat_session_db, chat_message_db
from schemas.chat_history_schemas import ChatSessionCreate, ChatSessionIn, ChatSessionUpdate, ChatMessageCreate


class ChatHistoryService:
    def get_or_create_session(self, db: Session, user_id: int, session_id: str, title: str):
        session = chat_session_db.get_by_session_id(db, session_id)
        if not session:
            obj_in = ChatSessionCreate(user_id=user_id, session_id=session_id, title=title[:50])
            session = chat_session_db.create(db, obj_in=obj_in)
        return session

    def create_session(self, db: Session, user_id: int, req: ChatSessionIn):
        s_id = req.session_id or str(uuid.uuid4())
        existing = chat_session_db.get_by_session_id(db, s_id)
        if existing: return existing
        obj_in = ChatSessionCreate(user_id=user_id, session_id=s_id, title=req.title or "新对话")
        return chat_session_db.create(db, obj_in=obj_in)

    def get_user_sessions(self, db: Session, user_id: int, page: int = 1, size: int = 20):
        return chat_session_db.get_by_user_id(db, user_id, (page - 1) * size, size)

    def get_session_messages(self, db: Session, user_id: int, session_id: str, page: int = 1, size: int = 50):
        # 先校验归属权
        sess = chat_session_db.get_by_session_id(db, session_id)
        if not sess or sess.user_id != user_id:
            return {"messages": []}
        messages = chat_message_db.get_by_session_id(db, session_id, (page - 1) * size, size)
        return {"messages": messages}

    def create_chat_message(self, db: Session, user_id: int, req: ChatMessageCreate):
        # 确保会话存在
        self.get_or_create_session(db, user_id, req.session_id, req.user_content[:20])
        data = req.model_dump()
        data['user_id'] = user_id
        # 兼容处理：Schema 里的 model 转为 DB 里的 model_name
        if 'model' in data and data['model']:
            data['model_name'] = data.pop('model')
        return chat_message_db.create(db, obj_in=data)


chat_history_service = ChatHistoryService()
