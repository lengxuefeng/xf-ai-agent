# -*- coding: utf-8 -*-
"""Yunyou Agent 关键词与字段常量。"""
from enum import Enum
from typing import Dict, Tuple


class YunyouKeywordGroup(str, Enum):
    """云柚业务关键词分组。"""

    SAME_DAY = "same_day"
    RANGE_7_DAYS = "range_7_days"
    RANGE_30_DAYS = "range_30_days"
    RANGE_90_DAYS = "range_90_days"
    RANGE_180_DAYS = "range_180_days"
    RANGE_365_DAYS = "range_365_days"
    ORDER_ASC = "order_asc"
    ORDER_DESC = "order_desc"
    HOLTER_DOMAIN = "holter_domain"
    STATS_BLOCK = "stats_block"
    LIST_INTENT = "list_intent"
    ORDER_INTENT = "order_intent"
    FOLLOWUP_DATE = "followup_date"
    FOLLOWUP_LIST_HISTORY = "followup_list_history"
    SQL_EXAMPLE_INTENT = "sql_example_intent"
    UPLOADED_YES = "uploaded_yes"
    UPLOADED_NO = "uploaded_no"
    HOLTER_2H = "holter_2h"
    HOLTER_48H = "holter_48h"
    HOLTER_NIGHT = "holter_night"
    HOLTER_24H = "holter_24h"


YUNYOU_KEYWORDS: Dict[YunyouKeywordGroup, Tuple[str, ...]] = {
    YunyouKeywordGroup.SAME_DAY: ("当天", "当日", "今天"),
    YunyouKeywordGroup.RANGE_7_DAYS: ("最近一周", "近一周", "最近7天", "近7天"),
    YunyouKeywordGroup.RANGE_30_DAYS: ("最近30天", "近30天", "最近一个月", "近一个月"),
    YunyouKeywordGroup.RANGE_90_DAYS: ("最近90天", "近90天", "最近三个月", "近三个月"),
    YunyouKeywordGroup.RANGE_180_DAYS: ("最近半年", "近半年"),
    YunyouKeywordGroup.RANGE_365_DAYS: ("最近一年", "近一年"),
    YunyouKeywordGroup.ORDER_ASC: ("升序", "asc", "最早"),
    YunyouKeywordGroup.ORDER_DESC: ("倒序", "倒叙", "降序", "desc", "最近", "最新", "最后"),
    YunyouKeywordGroup.HOLTER_DOMAIN: ("holter", "云柚", "动态心电", "贴片", "心电"),
    YunyouKeywordGroup.STATS_BLOCK: ("类型统计", "报告统计", "报告状态统计"),
    YunyouKeywordGroup.LIST_INTENT: (
        "列表",
        "明细",
        "记录",
        "用户",
        "有哪些",
        "最近",
        "最新",
        "最后",
        "按id",
        "按 id",
        "根据id",
        "根据 id",
        "id倒",
        "倒序",
        "倒叙",
        "limit",
        "top",
        "前",
        "有人使用",
        "有使用",
        "有没有人使用",
        "今天有人用",
        "是否有人使用",
    ),
    YunyouKeywordGroup.ORDER_INTENT: ("按id", "按 id", "根据id", "根据 id", "倒序", "倒叙", "降序", "limit", "top", "前"),
    YunyouKeywordGroup.FOLLOWUP_DATE: ("今天", "昨天", "本周", "上周", "本月", "上月", "最近", "近"),
    YunyouKeywordGroup.FOLLOWUP_LIST_HISTORY: ("列表", "记录", "用户", "按id", "按 id", "倒序", "倒叙", "limit", "top", "前"),
    YunyouKeywordGroup.SQL_EXAMPLE_INTENT: (
        "sql",
        "sql怎么写",
        "sql怎样写",
        "sql示例",
        "查询语句",
        "select",
        "语句怎么写",
    ),
    YunyouKeywordGroup.UPLOADED_YES: ("已上传", "上传完成", "上传完"),
    YunyouKeywordGroup.UPLOADED_NO: ("未上传", "没上传"),
    YunyouKeywordGroup.HOLTER_2H: ("2小时", "两小时"),
    YunyouKeywordGroup.HOLTER_48H: ("48小时",),
    YunyouKeywordGroup.HOLTER_NIGHT: ("夜间", "24小时（夜间）", "24小时(夜间)"),
    YunyouKeywordGroup.HOLTER_24H: ("24小时",),
}


YUNYOU_LIMIT_PATTERNS: Tuple[str, ...] = (
    r"(?:前|最后|最新)\s*(\d{1,3})\s*条",
    r"\btop\s*(\d{1,3})\b",
    r"\blimit\s*(\d{1,3})\b",
)


YUNYOU_RECORD_KEYS: Tuple[str, ...] = ("records", "list", "rows", "items", "result", "data")


YUNYOU_HOLTER_TYPE_MAP = {0: "24小时", 1: "2小时", 2: "24小时(夜间)", 3: "48小时"}
YUNYOU_REPORT_STATUS_MAP = {-1: "无数据", 0: "待审核", 1: "审核中", 2: "人工审核完成", 3: "自动审核完成"}
YUNYOU_UPLOAD_STATUS_MAP = {-1: "无数据", 0: "未上传", 1: "已上传"}

YUNYOU_RELATIVE_DATE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "yesterday": ("昨天",),
    "today": ("今天", "当日"),
    "this_week": ("本周", "这周"),
    "last_week": ("上周",),
    "this_month": ("本月", "这个月"),
    "last_month": ("上月", "上个月"),
}

YUNYOU_REPORT_STATUS_FILTERS: Dict[int, Tuple[str, ...]] = {
    0: ("待审核",),
    1: ("审核中",),
    2: ("人工审核完成",),
    3: ("自动审核完成",),
}

YUNYOU_HOLTER_TYPE_FILTERS: Dict[int, Tuple[str, ...]] = {
    1: YUNYOU_KEYWORDS[YunyouKeywordGroup.HOLTER_2H],
    3: YUNYOU_KEYWORDS[YunyouKeywordGroup.HOLTER_48H],
    2: YUNYOU_KEYWORDS[YunyouKeywordGroup.HOLTER_NIGHT],
    0: YUNYOU_KEYWORDS[YunyouKeywordGroup.HOLTER_24H],
}

YUNYOU_HOLTER_TABLE_COLUMNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("ID", ("id",)),
    ("用户ID", ("user_id", "userId")),
    ("用户名", ("nick_name", "nickName", "user_name", "userName")),
    ("使用日期", ("use_day", "useDay")),
    ("开始时间", ("begin_date_time", "beginDateTime")),
    ("结束时间", ("end_date_time", "endDateTime")),
    ("上传状态", ("is_uploaded", "isUploaded")),
    ("报告状态", ("report_status", "reportStatus")),
    ("Holter类型", ("holter_type", "holterType")),
)
