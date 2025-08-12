from pydantic import BaseModel, Field
from typing import Optional
from bson import ObjectId
from datetime import datetime

from schemas.base import MongodbBaseSchema


# class PyObjectId(ObjectId):
#     @classmethod
#     def __get_validators__(cls):
#         yield cls.validate
#
#     @classmethod
#     def validate(cls, v):
#         if not ObjectId.is_valid(v):
#             raise ValueError("无效的 ObjectId")
#         return ObjectId(v)
#
#     @classmethod
#     def __modify_schema__(cls, field_schema):
#         field_schema.update(type="string", description="MongoDB 文档ID")


# class ChatHistory(MongodbBaseSchema):
#     id: Optional[PyObjectId] = Field(alias="_id")
#     # 上一条消息的_id（用于构建对话树）
#     parent_message_id: Optional[PyObjectId] = None
#     # 用户ID
#     user_id: int
#     # 会话标题
#     title: str
#     # 会话唯一标识（如用户ID+时间戳哈希）
#     session_id: str
#     # user
#     user_content: str
#     # 模型响应
#     model_content: Optional[str] = None
#     # tokens
#     tokens: Optional[int] = None
#     # model
#     model: Optional[str] = None
#     # 响应延迟毫秒数（性能监控）
#     latency_ms: Optional[int] = None
#     # 删除标记
#     is_deleted: Optional[int] = 0
#     # 会话创建时间
#     created_at: Optional[datetime] = datetime.now()

