# -*- coding: utf-8 -*-
"""Agent 工具层统一入参 Schema。"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field

from models.schemas.base import BaseSchema


class TavilySearchToolInput(BaseSchema):
    query: str = Field(default="", description="需要联网检索的查询关键词或问题。")
    topic: Literal["general", "news", "finance"] = Field(
        default="general",
        description="搜索主题，可选 general、news、finance。",
    )


class GetWeathersToolInput(BaseSchema):
    city_names: List[str] = Field(default_factory=list, description="需要查询天气的城市名称列表。")
    max_threads: int = Field(default=4, ge=1, description="兼容旧接口保留参数，表示历史线程并发上限。")


class HolterListToolInput(BaseSchema):
    startUseDay: str = Field(default="", description="开始日期，格式为 YYYY-MM-DD。")
    endUseDay: str = Field(default="", description="结束日期，格式为 YYYY-MM-DD。")
    isUploaded: Optional[int] = Field(default=None, description="数据是否上传完成，0 表示否，1 表示是。")
    reportStatus: Optional[int] = Field(
        default=None,
        description="报告审核状态，0 待审核，1 审核中，2 人工审核完成，3 自动审核完成。",
    )
    holterType: Optional[int] = Field(
        default=None,
        description="Holter 类型，0 为 24 小时，1 为 2 小时，2 为夜间 24 小时，3 为 48 小时。",
    )


class HolterDateRangeToolInput(BaseSchema):
    startUseDay: str = Field(default="", description="开始日期，格式为 YYYY-MM-DD。")
    endUseDay: str = Field(default="", description="结束日期，格式为 YYYY-MM-DD。")


class HolterRecentDbToolInput(BaseSchema):
    limit: int = Field(default=5, ge=1, description="返回记录条数上限。")
    order: Literal["asc", "desc"] = Field(default="desc", description="排序方向，只允许 asc 或 desc。")
    startUseDay: Optional[str] = Field(default=None, description="开始日期，格式为 YYYY-MM-DD，可选。")
    endUseDay: Optional[str] = Field(default=None, description="结束日期，格式为 YYYY-MM-DD，可选。")


class HolterLogInfoToolInput(BaseSchema):
    user_id: str = Field(default="", description="要查询日志的用户 ID。")


class ExecuteSqlToolInput(BaseSchema):
    sql: str = Field(default="", description="需要审批并执行的只读 SQL 语句。")


class EmptyToolInput(BaseSchema):
    """无入参工具的占位 Schema。"""


__all__ = [
    "EmptyToolInput",
    "ExecuteSqlToolInput",
    "GetWeathersToolInput",
    "HolterDateRangeToolInput",
    "HolterListToolInput",
    "HolterLogInfoToolInput",
    "HolterRecentDbToolInput",
    "TavilySearchToolInput",
]
