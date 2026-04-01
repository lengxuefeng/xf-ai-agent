# -*- coding: utf-8 -*-
"""SQL 策略中心常量。"""
from typing import Set


SQL_WRITE_KEYWORDS: Set[str] = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "create",
    "grant",
    "revoke",
    "merge",
}


DEFAULT_LOCAL_TABLES = "t_user_info,t_chat_session,t_chat_message,t_model_setting,t_user_model,t_user_mcp,t_user_skill,t_interrupt_approval"
DEFAULT_YUNYOU_TABLES = "t_holter_use_record"
