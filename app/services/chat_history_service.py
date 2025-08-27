from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from db.mongodb.chat_history_db import chat_history_db
from schemas.chat_history_schemas import ChatHistory, ChatHistoryCreate, ChatHistoryUpdate
from services.base_service import BaseService


class ChatHistoryService:
    """
    聊天历史服务层
    """
    
    def __init__(self):
        self.db = chat_history_db
    
    def create_chat(self, user_id: int, chat_data: ChatHistoryCreate) -> ChatHistory:
        """
        创建聊天记录
        """
        return self.db.create_with_user_id(user_id, chat_data)
    
    def get_chat_by_id(self, user_id: int, chat_id: str) -> Optional[ChatHistory]:
        """
        获取单条聊天记录
        """
        return self.db.get_by_user_and_id(user_id, chat_id)
    
    def get_chat_list(
        self, 
        user_id: int, 
        session_id: Optional[str] = None, 
        page: int = 1, 
        size: int = 10
    ) -> Dict[str, Any]:
        """
        获取聊天列表（分页）
        """
        items, total, current_page, pages = self.db.get_user_chat_list(
            user_id=user_id, 
            session_id=session_id, 
            page=page, 
            size=size
        )
        
        # 转换 ObjectId 为字符串
        chat_list = []
        for chat in items:
            chat_dict = chat.model_dump() if hasattr(chat, 'model_dump') else chat.__dict__
            if '_id' in chat_dict:
                chat_dict['_id'] = str(chat_dict['_id'])
            if chat_dict.get('parent_message_id'):
                chat_dict['parent_message_id'] = str(chat_dict['parent_message_id'])
            chat_list.append(chat_dict)
        
        return {
            "items": chat_list,
            "total": total,
            "page": current_page,
            "size": size,
            "pages": pages
        }
    
    def search_chats(
        self, 
        user_id: int, 
        keyword: str, 
        session_id: Optional[str] = None, 
        page: int = 1, 
        size: int = 10
    ) -> Dict[str, Any]:
        """
        搜索聊天记录
        """
        items, total, current_page, pages = self.db.search_chat_history(
            user_id=user_id,
            keyword=keyword,
            session_id=session_id,
            page=page,
            size=size
        )
        
        # 转换 ObjectId 为字符串
        chat_list = []
        for chat in items:
            chat_dict = chat.model_dump() if hasattr(chat, 'model_dump') else chat.__dict__
            if '_id' in chat_dict:
                chat_dict['_id'] = str(chat_dict['_id'])
            if chat_dict.get('parent_message_id'):
                chat_dict['parent_message_id'] = str(chat_dict['parent_message_id'])
            chat_list.append(chat_dict)
        
        return {
            "items": chat_list,
            "total": total,
            "page": current_page,
            "size": size,
            "pages": pages,
            "keyword": keyword
        }
    
    def update_chat(self, user_id: int, chat_id: str, update_data: ChatHistoryUpdate) -> Optional[ChatHistory]:
        """
        更新聊天记录
        """
        return self.db.update_by_user_and_id(user_id, chat_id, update_data)
    
    def delete_chat(self, user_id: int, chat_id: str, hard_delete: bool = False) -> bool:
        """
        删除聊天记录
        """
        if hard_delete:
            return self.db.hard_delete_by_user_and_id(user_id, chat_id)
        else:
            return self.db.soft_delete_by_user_and_id(user_id, chat_id)
    
    def delete_session(self, user_id: int, session_id: str, hard_delete: bool = False) -> int:
        """
        删除整个会话
        """
        return self.db.delete_session(user_id, session_id, hard_delete)
    
    def get_user_sessions(self, user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """
        获取用户会话列表
        """
        sessions = self.db.get_user_sessions(user_id, limit)
        
        # 格式化返回数据
        result = []
        for session in sessions:
            result.append({
                "session_id": session["_id"],
                "title": session["title"],
                "latest_message": session["latest_message"],
                "latest_response": session["latest_response"],
                "created_at": session["created_at"],
                "updated_at": session["updated_at"],
                "message_count": session["message_count"]
            })
        
        return result
    
    def get_user_statistics(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """
        获取用户统计信息
        """
        stats = self.db.get_user_statistics(user_id, days)
        
        if stats:
            from datetime import timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            return {
                "total_messages": stats["total_messages"],
                "total_tokens": stats["total_tokens"] or 0,
                "total_sessions": len(stats["total_sessions"]),
                "avg_latency_ms": round(stats["avg_latency"] or 0, 2),
                "models_used": [model for model in stats["models_used"] if model],
                "period_days": days,
                "start_date": start_date,
                "end_date": end_date
            }
        else:
            from datetime import timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            return {
                "total_messages": 0,
                "total_tokens": 0,
                "total_sessions": 0,
                "avg_latency_ms": 0,
                "models_used": [],
                "period_days": days,
                "start_date": start_date,
                "end_date": end_date
            }


# 创建全局实例
chat_history_service = ChatHistoryService()