from datetime import datetime
from typing import Optional, Any

from bson import ObjectId
from pydantic import BaseModel, Field
from pydantic_core import core_schema

from schemas.base import MongodbBaseSchema


class PyObjectId(ObjectId):
    """
    自定义的ObjectId类型，用于Pydantic模型。
    使其能够通过字符串进行校验，并在JSON schema中表现为字符串。
    """

    @classmethod
    def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: Any
    ) -> core_schema.CoreSchema:
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema([
                core_schema.is_instance_schema(ObjectId),
                core_schema.chain_schema([
                    core_schema.str_schema(),
                    core_schema.no_info_plain_validator_function(cls.validate),
                ])
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: str(x)
            ),
        )

    @classmethod
    def validate(cls, v: Any) -> ObjectId:
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str) and ObjectId.is_valid(v):
            return ObjectId(v)
        raise ValueError("无效的 ObjectId")


# --- Chat Session Schemas ---

class ChatSessionBase(MongodbBaseSchema):
    """会话基础模型"""
    user_id: int = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话唯一标识")
    title: str = Field(..., description="会話标题")
    is_deleted: int = Field(default=0, description="删除标记, 0:未删除, 1:已删除")


class ChatSessionIn(BaseModel):
    """API请求用的会话创建模型"""
    title: str = Field(..., description="会话标题")
    session_id: Optional[str] = Field(None, description="会话唯一标识(可选)")


class ChatSessionCreate(ChatSessionBase):
    """创建会话的模型"""
    created_at: datetime = Field(default_factory=datetime.now, description="会话创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="会话更新时间")


class ChatSessionUpdate(BaseModel):
    """更新会话的模型"""
    title: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.now)


class ChatSession(ChatSessionBase):
    """从数据库读取的会话模型"""
    id: PyObjectId = Field(alias="_id", description="MongoDB 文档ID")
    created_at: datetime = Field(default_factory=datetime.now, description="会话创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="会话更新时间")


# --- Chat Message Schemas ---

class ChatMessageBase(MongodbBaseSchema):
    """聊天记录基础模型"""
    parent_message_id: Optional[PyObjectId] = Field(default=None, description="上一条消息的_id（用于构建对话树）")
    user_id: Optional[int] = Field(default=None, description="用户ID")
    session_id: str = Field(..., description="会话唯一标识")
    user_content: str = Field(..., description="用户输入内容")
    model_content: Optional[str] = Field(default=None, description="模型响应")
    tokens: Optional[int] = Field(default=None, description="消耗的tokens")
    model: Optional[str] = Field(default=None, description="使用的模型")
    latency_ms: Optional[int] = Field(default=None, description="响应延迟毫秒数")
    is_deleted: int = Field(default=0, description="删除标记, 0:未删除, 1:已删除")


class ChatMessageCreate(ChatMessageBase):
    """创建聊天记录的模型"""
    created_at: datetime = Field(default_factory=datetime.now, description="消息创建时间")


class ChatMessageUpdate(BaseModel):
    """更新聊天记录的模型"""
    model_content: Optional[str] = None
    tokens: Optional[int] = None
    is_deleted: Optional[int] = None


class ChatMessage(ChatMessageBase):
    """从数据库读取的聊天记录模型"""
    id: PyObjectId = Field(alias="_id", description="MongoDB 文档ID")
    user_id: int = Field(..., description="用户ID")
    created_at: datetime = Field(default_factory=datetime.now, description="消息创建时间")
