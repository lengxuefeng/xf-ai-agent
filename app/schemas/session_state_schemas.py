# -*- coding: utf-8 -*-
from typing import Optional, Dict, Any

from schemas.base import BaseSchema


class SessionStateCreate(BaseSchema):
    """创建会话状态请求体"""

    user_id: int
    session_id: str
    slots: Optional[Dict[str, Any]] = None
    summary_text: Optional[str] = None
    last_route: Optional[Dict[str, Any]] = None
    turn_count: int = 0
    is_deleted: bool = False


class SessionStateUpdate(BaseSchema):
    """更新会话状态请求体"""

    slots: Optional[Dict[str, Any]] = None
    summary_text: Optional[str] = None
    last_route: Optional[Dict[str, Any]] = None
    turn_count: Optional[int] = None
    is_deleted: Optional[bool] = None

