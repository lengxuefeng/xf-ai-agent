# -*- coding: utf-8 -*-
"""
Pydantic 基础模型定义（Base Schemas）。

定义了系统中所有Pydantic模型的基础类。
这些基础模型提供了通用的配置和功能，其他模型通过继承获得这些能力。

设计要点：
1. 统一的日期格式化
2. 支持ORM对象属性读取
3. 支持任意类型的扩展模型
4. 灵活的配置选项

使用场景：
- 所有Schema模型都应该继承这些基础类
- ORM对象的自动序列化
- 复杂对象（如LangChain组件）的处理
"""
import datetime
from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    """
    标准基础模型。

    核心职责：
    1. 提供统一的日期格式化
    2. 支持从SQLAlchemy ORM对象读取属性
    3. 提供模型级别的配置

    设计理由：
    1. 统一的日期格式化：所有datetime字段自动格式化为'YYYY-MM-DD HH:MM:SS'
    2. ORM支持：可以直接从SQLAlchemy模型对象创建Pydantic模型
    3. 灵活性：所有子模型自动继承这些配置

    重要配置说明：
    - from_attributes=True: 允许从ORM对象读取属性
      场景：数据库模型直接转换为Pydantic模型，无需手动映射
      注意：极其重要，否则ORM对象无法转换为Pydantic模型

    - json_encoders: 统一日期格式化
      格式：'%Y-%m-%d %H:%M:%S'
      场景：API响应中的时间字段统一格式

    使用场景：
    - 用户信息模型
    - 配置模型
    - 聊天记录模型
    - 审批记录模型
    - 所有API请求和响应模型
    """
    model_config = ConfigDict(
        from_attributes=True,  # 允许从 SQLAlchemy ORM 对象读取属性 (极其重要)
        json_encoders={datetime.datetime: lambda v: v.strftime('%Y-%m-%d %H:%M:%S')}  # 统一日期格式化
    )


class ArbitraryTypesBaseSchema(BaseModel):
    """
    支持任意类型的基础模型。

    核心职责：
    1. 支持非Pydantic原生类型（如LangChain组件）
    2. 继承BaseSchema的所有能力
    3. 提供最大的灵活性

    设计理由：
    1. 某些场景需要处理复杂的Python对象
    2. LangChain组件经常包含复杂的嵌套结构
    3. 提供了Pydantic V2的完整灵活性

    重要配置说明：
    - arbitrary_types_allowed=True: 允许任意Python类型
      场景：存储和序列化LangChain组件、自定义类等
      注意：可能导致性能问题，只在需要时使用

    - from_attributes=True: 兼许从ORM对象读取属性
      场景：ORM对象包含Pydantic模型或组件

    - json_encoders: 统一日期格式化
      格式：'%Y-%m-%d %H:%M:%S'
      场景：API响应中的时间字段统一格式

    使用场景：
    - 包含LangChain组件的配置模型
    - 包含自定义工具的配置模型
    - 需要序列化复杂对象的场景
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # 允许非 Pydantic 原生类型
        from_attributes=True,
        json_encoders={datetime.datetime: lambda v: v.strftime('%Y-%m-%d %H:%M:%S')}
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