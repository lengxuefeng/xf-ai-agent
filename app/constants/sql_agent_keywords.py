# -*- coding: utf-8 -*-
"""SQL Agent 关键词与字段常量。"""
from enum import Enum
from typing import Dict, Tuple


class SqlAgentKeywordGroup(str, Enum):
    """SQL Agent 关键词分组。"""

    HOLTER_DOMAIN = "holter_domain"
    ORDER_HINT = "order_hint"
    HOLTER_TABLE_HINT = "holter_table_hint"
    HOLTER_TIME_COLUMNS = "holter_time_columns"
    HOLTER_TABLE_NAME_HINT = "holter_table_name_hint"
    HOLTER_SELECT_COLUMNS = "holter_select_columns"
    HOLTER_BAD_TABLES = "holter_bad_tables"


SQL_AGENT_KEYWORDS: Dict[SqlAgentKeywordGroup, Tuple[str, ...]] = {
    SqlAgentKeywordGroup.HOLTER_DOMAIN: ("holter", "动态心电", "云柚"),
    SqlAgentKeywordGroup.ORDER_HINT: ("倒序", "降序", "id", "前", "limit", "order by"),
    SqlAgentKeywordGroup.HOLTER_TABLE_HINT: ("holter",),
    SqlAgentKeywordGroup.HOLTER_TIME_COLUMNS: ("report_time", "usage_date", "create_time", "start_time", "update_time"),
    SqlAgentKeywordGroup.HOLTER_TABLE_NAME_HINT: ("usage", "record", "report"),
    SqlAgentKeywordGroup.HOLTER_SELECT_COLUMNS: ("id", "user_id", "user_name", "nick_name"),
    SqlAgentKeywordGroup.HOLTER_BAD_TABLES: ("t_chat_message", "t_chat_session", "chat_message", "chat_session"),
}


SQL_AGENT_ORDER_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "id",
    "report_time",
    "usage_date",
    "create_time",
    "start_time",
    "update_time",
)


SQL_AGENT_LIMIT_PATTERNS: Tuple[str, ...] = (
    r"前\s*(\d+)\s*条",
    r"limit\s+(\d+)",
)
