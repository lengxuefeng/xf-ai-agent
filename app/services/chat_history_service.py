# -*- coding: utf-8 -*-
import uuid
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from db.crud import chat_session_db, chat_message_db
from models.schemas.chat_history_schemas import ChatSessionCreate, ChatSessionIn, ChatSessionUpdate, ChatMessageCreate


class ChatHistoryService:
    def get_or_create_session(self, db: Session, user_id: int, session_id: str, title: str):
        """
        按会话 ID 获取或创建会话。

        这里保留“幂等创建”能力，避免流式首条消息和显式创建接口并发时生成重复会话。
        """
        session = chat_session_db.get_by_session_id(db, session_id)
        if not session:
            safe_title = (title or "新对话").strip()[:50] or "新对话"
            obj_in = ChatSessionCreate(user_id=user_id, session_id=session_id, title=safe_title)
            session = chat_session_db.create(db, obj_in=obj_in)
        return session

    def create_session(self, db: Session, user_id: int, req: ChatSessionIn):
        """创建新会话；若前端传入的 session_id 已存在，则直接复用。"""
        s_id = req.session_id or str(uuid.uuid4())
        existing = chat_session_db.get_by_session_id(db, s_id)
        if existing:
            return existing
        safe_title = (req.title or "新对话").strip() or "新对话"
        obj_in = ChatSessionCreate(user_id=user_id, session_id=s_id, title=safe_title)
        return chat_session_db.create(db, obj_in=obj_in)

    def get_user_sessions(self, db: Session, user_id: int, page: int = 1, size: int = 20):
        """分页获取用户会话列表，并返回总数供前端分页控件使用。"""
        skip = (page - 1) * size
        items = chat_session_db.get_by_user_id(db, user_id, skip, size)
        total = chat_session_db.count_by_user_id(db, user_id)
        return {
            "items": items,
            "total": total,
            "page": page,
            "size": size,
        }

    def get_session_messages(
            self,
            db: Session,
            user_id: int,
            session_id: str,
            page: int = 1,
            size: int = 50,
            order: str = "asc"
    ):
        # 先校验归属权
        sess = chat_session_db.get_by_session_id(db, session_id)
        if not sess or sess.user_id != user_id:
            return {"messages": [], "total": 0, "page": page, "size": size, "has_more": False, "order": order}

        normalized_order = (order or "asc").lower()
        order_desc = normalized_order == "desc"
        offset = (page - 1) * size
        total = chat_message_db.count_by_session_id(db, session_id)
        messages = chat_message_db.get_by_session_id(
            db=db,
            session_id=session_id,
            skip=offset,
            limit=size,
            order_desc=order_desc
        )

        has_more = offset + len(messages) < total
        return {
            "messages": messages,
            "total": total,
            "page": page,
            "size": size,
            "has_more": has_more,
            "order": "desc" if order_desc else "asc"
        }

    def rename_session(self, db: Session, user_id: int, session_id: str, title: str):
        """
        重命名指定会话。

        只允许修改当前用户自己的未删除会话，避免不同账号之间串改标题。
        """
        session = self._get_owned_session(db, user_id, session_id)
        safe_title = (title or "新对话").strip()[:200] or "新对话"
        return chat_session_db.update(db, session, ChatSessionUpdate(title=safe_title))

    def delete_session(self, db: Session, user_id: int, session_id: str):
        """
        软删除会话，并同步软删除其下全部消息。

        这里走软删除是为了兼容后续回收站、审计与恢复能力，不直接物理清除。
        """
        session = self._get_owned_session(db, user_id, session_id)
        updated_session = chat_session_db.update(db, session, ChatSessionUpdate(is_deleted=True))
        (
            db.query(chat_message_db.model)
            .filter(
                chat_message_db.model.session_id == session_id,
                chat_message_db.model.user_id == user_id,
                chat_message_db.model.is_deleted == False,
            )
            .update({"is_deleted": True}, synchronize_session=False)
        )
        db.flush()
        db.refresh(updated_session)
        return updated_session

    def delete_message(self, db: Session, user_id: int, message_id: int):
        """软删除单条聊天消息，前提是该消息属于当前用户。"""
        message = chat_message_db.get(db, message_id)
        if not message or message.user_id != user_id or message.is_deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="消息不存在")
        return chat_message_db.update(db, message, {"is_deleted": True})

    def get_recent_session_messages(
        self,
        db: Session,
        user_id: int,
        session_id: str,
        limit: int = 30,
    ):
        """
        流式链路专用的轻量历史读取：
        1) 仅查询最近 N 条；
        2) 不做 count，避免额外一次聚合查询；
        3) 最终按时间正序返回，便于直接拼接上下文。
        """
        sess = chat_session_db.get_by_session_id(db, session_id)
        if not sess or sess.user_id != user_id:
            return []

        safe_limit = max(1, min(int(limit or 30), 200))
        rows = chat_message_db.get_by_session_id(
            db=db,
            session_id=session_id,
            skip=0,
            limit=safe_limit,
            order_desc=True,
        )
        # get_by_session_id(order_desc=True) 返回倒序，这里翻转为正序
        return list(reversed(rows))

    def create_chat_message(
        self,
        db: Session,
        user_id: int,
        req: ChatMessageCreate,
        *,
        ensure_session: bool = True,
    ):
        """创建一轮消息记录，并兼容旧字段 model -> model_name 的映射。"""
        # 确保会话存在
        if ensure_session:
            self.get_or_create_session(db, user_id, req.session_id, req.user_content[:20])
        data = req.model_dump()
        data['user_id'] = user_id
        # 兼容处理：Schema 里的 model 转为 DB 里的 model_name
        if 'model' in data and data['model']:
            data['model_name'] = data.pop('model')
        return chat_message_db.create(db, obj_in=data)

    def _get_owned_session(self, db: Session, user_id: int, session_id: str):
        """读取当前用户拥有的会话，不满足条件时统一抛 404。"""
        session = chat_session_db.get_by_session_id(db, session_id)
        if not session or session.user_id != user_id or session.is_deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
        return session


chat_history_service = ChatHistoryService()
