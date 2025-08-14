from datetime import datetime
from typing import Optional

from bson import ObjectId
from pydantic import BaseModel, Field

from schemas.base import MongodbBaseSchema


class PyObjectId(ObjectId):
    """
    自定义的ObjectId类型，用于Pydantic模型。
    使其能够通过字符串进行校验，并在JSON schema中表现为字符串。
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("无效的 ObjectId")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class ChatHistoryBase(MongodbBaseSchema):
    """聊天记录基础模型"""
    parent_message_id: Optional[PyObjectId] = Field(default=None, description="上一条消息的_id（用于构建对话树）")
    user_id: int = Field(..., description="用户ID")
    title: str = Field(..., description="会话标题")
    session_id: str = Field(..., description="会话唯一标识")
    user_content: str = Field(..., description="用户输入内容")
    model_content: Optional[str] = Field(default=None, description="模型响应")
    tokens: Optional[int] = Field(default=None, description="消耗的tokens")
    model: Optional[str] = Field(default=None, description="使用的模型")
    latency_ms: Optional[int] = Field(default=None, description="响应延迟毫秒数")
    is_deleted: int = Field(default=0, description="删除标记, 0:未删除, 1:已删除")


class ChatHistoryCreate(ChatHistoryBase):
    """创建聊天记录的模型"""
    created_at: datetime = Field(default_factory=datetime.now, description="会话创建时间")


class ChatHistoryUpdate(BaseModel):
    """更新聊天记录的模型"""
    title: Optional[str] = None
    model_content: Optional[str] = None
    tokens: Optional[int] = None
    is_deleted: Optional[int] = None


class ChatHistory(ChatHistoryBase):
    """从数据库读取的聊天记录模型"""
    id: PyObjectId = Field(alias="_id", description="MongoDB 文档ID")
    created_at: datetime = Field(..., description="会话创建时间")
