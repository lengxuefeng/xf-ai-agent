# -*- coding: utf-8 -*-
"""
会话状态持久化模型（Session State Model）。

这个模型用于存储会话级别的结构化上下文信息，如用户画像、槽位、路由快照等。
会话状态是系统"记忆"用户信息的重要机制，可以减少重复追问和误路由。

设计目标：
1. 结构化存储：将城市、用户画像等信息存储为结构化数据
2. 减少重复：避免每次都从历史文本中猜测信息
3. 统一访问：路由层、Agent层都能访问同一份状态
4. 持久化保证：状态保存在数据库中，重启不丢失

使用场景：
- 存储用户所在城市，天气查询时不需要每次都问
- 存储用户画像（姓名、年龄等），相关问题时自动使用
- 记录最近一次路由快照，用于排障和分析
- 累计会话轮次，用于策略判断

槽位说明：
- city: 城市
- name: 姓名
- age: 年龄
- gender: 性别
- height_cm: 身高（厘米）
- weight_kg: 体重（千克）
- last_topic: 最近主题
- key_facts: 关键事实列表
"""
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import String, DateTime, func, BigInteger, Boolean, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from db import Base


class SessionStateModel(Base):
    """
    会话状态表（每个 session_id 一条）。

    核心职责：
    1. 存储会话级的结构化槽位信息
    2. 生成人类可读的摘要，用于系统提示注入
    3. 记录最近一次路由快照
    4. 累计会话轮次，用于策略判断

    设计理由：
    1. 结构化存储比从历史文本提取更可靠
    2. 避免重复追问，提升用户体验
    3. 减少token浪费，提高响应效率
    4. 支持复杂的上下文策略

    字段说明：
    - id: 主键，自增
    - user_id: 用户ID
    - session_id: 会话ID，唯一
    - slots: 槽位字典，JSONB格式
    - summary_text: 人类可读摘要
    - last_route: 最近一次路由快照
    - turn_count: 会话轮次计数
    - is_deleted: 软删除标记
    - create_time: 创建时间
    - update_time: 更新时间

    槽位字典结构：
    {
        "city": "郑州",
        "name": "张三",
        "age": 30,
        "gender": "male",
        "height_cm": 175,
        "weight_kg": 70.5,
        "last_topic": "weather",
        "key_facts": ["需求=查询天气", "时间=今天"]
    }

    使用流程：
    1. 用户请求进来时，读取会话状态
    2. 从用户输入中提取新的槽位信息
    3. 合并新旧槽位，更新会话状态
    4. 生成摘要，注入到系统提示中
    5. 使用路由快照，优化路由决策
    """
    __tablename__ = "t_session_state"

    # 主键
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # 标识字段
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)

    # 槽位信息
    # 槽位字典：如 city/name/age/gender/height_cm/weight_kg 等
    # 使用JSONB类型，支持复杂的嵌套结构
    slots: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    # 人类可读摘要，便于快速注入系统上下文
    # 示例：【会话关键上下文】- 当前城市: 郑州 - 用户画像: 姓名=张三，年龄=30岁
    summary_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 保存最近一次路由快照，用于排障和策略分析
    # 记录路由决策、Agent选择、执行结果等
    last_route: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    # 统计信息
    # 累积轮次，便于后续策略（如摘要刷新阈值）扩展
    turn_count: Mapped[int] = mapped_column(BigInteger, default=0)

    # 软删除标记
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    # 时间戳
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    update_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

