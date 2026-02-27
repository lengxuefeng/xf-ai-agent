# -*- coding: utf-8 -*-
"""
Pydantic 基础模型定义

本模块提供了不同用途的基础模型类：

1. BaseSchema - 标准基础模型
   - 用于常规的数据模型（如用户信息、配置、聊天记录等）
   - 支持 ORM 对象属性读取
   - 提供日期格式化

2. ArbitraryTypesBaseSchema - 支持任意类型的基础模型
   - 用于包含复杂对象的场景（如 LLM 模型、工具类等）
   - 适用于 LangChain 相关的数据结构
   - 继承了 BaseSchema 的所有功能
"""

import datetime
from pydantic import BaseModel, ConfigDict

class BaseSchema(BaseModel):
    """标准基础模型"""
    model_config = ConfigDict(
        from_attributes=True,  # 允许从 SQLAlchemy ORM 对象读取属性 (极其重要)
        json_encoders={datetime.datetime: lambda v: v.strftime('%Y-%m-%d %H:%M:%S')}  # 统一日期格式化
    )

class ArbitraryTypesBaseSchema(BaseModel):
    """
    支持任意类型的基础模型 - 用于包含复杂对象（如 LangChain 组件）的场景
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # 允许非 Pydantic 原生类型
        from_attributes=True,
        json_encoders={datetime.datetime: lambda v: v.strftime('%Y-%m-%d %H:%M:%S')}
    )