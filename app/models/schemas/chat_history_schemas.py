# -*- coding: utf-8 -*-
from datetime import datetime
from typing import Optional, Any, Dict

from pydantic import BaseModel, Field

from models.schemas.base import BaseSchema


# --- Chat Session Schemas ---

class ChatSessionBase(BaseSchema):
    """会话基础模型"""
    user_id: int = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话唯一标识")
    title: str = Field(default="新对话", description="会话标题")
    is_deleted: bool = Field(default=False, description="删除标记: False 未删除, True 已删除")


class ChatSessionCreate(ChatSessionBase):
    """创建会话模型"""
    pass


class ChatSessionUpdate(BaseModel):
    """更新会话模型"""
    title: Optional[str] = None
    is_deleted: Optional[bool] = None


class ChatSession(ChatSessionBase):
    """会话响应模型 (返回给前端的数据)"""
    id: int = Field(..., description="PgSQL的自增ID")
    create_time: datetime = Field(..., description="会话创建时间")
    update_time: datetime = Field(..., description="会话更新时间")


class ChatSessionIn(BaseModel):
    """前端传入的创建会话参数"""
    session_id: Optional[str] = None
    title: Optional[str] = None


# --- Chat Message Schemas ---

class ChatMessageBase(BaseSchema):
    """聊天记录基础模型"""
    session_id: str = Field(..., description="会话唯一标识")
    user_content: str = Field(..., description="用户输入内容")
    model_content: str = Field(..., description="模型响应内容")
    model_name: Optional[str] = Field(default=None, description="使用的模型名称")
    tokens: int = Field(default=0, description="消耗的tokens")
    latency_ms: int = Field(default=0, description="响应延迟毫秒数")
    is_deleted: bool = Field(default=False, description="删除标记: False 未删除, True 已删除")

    # 【PgSQL JSONB 杀手锏】以前 Mongo 里的 parent_message_id, tool_calls 等都可以塞这里
    extra_data: Optional[Dict[str, Any]] = Field(default=None, description="附加结构化数据")


class ChatMessageCreate(ChatMessageBase):
    """创建聊天记录的模型"""
    pass


class ChatMessageUpdate(BaseModel):
    """更新聊天记录的模型"""
    is_deleted: Optional[bool] = None
    extra_data: Optional[Dict[str, Any]] = None


class ChatMessage(ChatMessageBase):
    """聊天记录响应模型 (返回给前端的数据)"""
    id: int = Field(..., description="PgSQL的自增主键ID")
    user_id: int = Field(..., description="用户ID")
    create_time: datetime = Field(..., description="消息创建时间")