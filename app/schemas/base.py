# app/schemas/base.py
"""
Pydantic 基础模型定义

本模块提供了不同用途的基础模型类：

1. BaseSchema - 标准基础模型
   - 用于常规的数据模型（如用户信息、配置等）
   - 支持 ORM 对象属性读取
   - 提供日期格式化

2. ArbitraryTypesBaseSchema - 支持任意类型的基础模型
   - 用于包含复杂对象的场景（如 LLM 模型、工具类等）
   - 适用于 LangChain 相关的数据结构
   - 继承了 BaseSchema 的所有功能

3. MongodbBaseSchema - MongoDB 专用基础模型
   - 用于 MongoDB 文档模型
   - 支持 ObjectId 处理
   - 包含 MongoDB 特有的配置

使用建议：
- 普通数据模型 -> 继承 BaseSchema
- 包含 LLM/工具对象 -> 继承 ArbitraryTypesBaseSchema  
- MongoDB 文档 -> 继承 MongodbBaseSchema
"""

import datetime

from pydantic import BaseModel, ConfigDict
from bson.objectid import ObjectId


class BaseSchema(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,  # 允许从 ORM 对象读取属性
        json_encoders={datetime.datetime: lambda v: v.strftime('%Y-%m-%d %H:%M:%S')}  # 日期格式化
    )


class ArbitraryTypesBaseSchema(BaseModel):
    """
    支持任意类型的基础模型 - 用于包含复杂对象（如 LLM 模型）的场景
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # 允许任意类型
        from_attributes=True,  # 允许从 ORM 对象读取属性
        json_encoders={datetime.datetime: lambda v: v.strftime('%Y-%m-%d %H:%M:%S')}  # 日期格式化
    )


class MongodbBaseSchema(BaseModel):
    """
    Mongodb基础模型 - 包含id字段
    """
    model_config = ConfigDict(
        # 日期格式化
        json_encoders={ObjectId: str, datetime.datetime: lambda v: v.strftime('%Y-%m-%d %H:%M:%S')},
        # 允许任意类型
        arbitrary_types_allowed=True,
        # 允许通过字段名填充
        populate_by_name=True,
        # schema_extra 改为 json_schema_extra
        json_schema_extra={
            "example": {
                "user_id": 123,
                "session_id": "sess_abc",
                "user_content": "你是谁",
                "model_content": "我是一个AI助手",
                "tokens": 42,
                "model": "gpt-4"
            }
        }
    )
