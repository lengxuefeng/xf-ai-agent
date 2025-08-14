# app/schemas/base.py

import datetime

from pydantic import BaseModel, ConfigDict
from bson.objectid import ObjectId


class BaseSchema(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,  # 允许从 ORM 对象读取属性
        json_encoders={datetime.datetime: lambda v: v.strftime('%Y-%m-%d %H:%M:%S')}  # 日期格式化
    )


class MongodbBaseSchema(BaseModel):
    """
    Mongodb基础模型 - 包含id字段
    """
    class Config:
        # 日期格式化
        json_encoders = {ObjectId: str, datetime.datetime: lambda v: v.strftime('%Y-%m-%d %H:%M:%S')}
        # 允许任意类型
        arbitrary_types_allowed = True
        # 允许通过字段名填充
        populate_by_name = True
        schema_extra = {
            "example": {
                "user_id": 123,
                "session_id": "sess_abc",
                "user_content": "你是谁",
                "model_content": "我是一个AI助手",
                "tokens": 42,
            "model": "gpt-4"
        }
    }
