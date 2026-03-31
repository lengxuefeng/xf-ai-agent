# -*- coding: utf-8 -*-
"""SQL 工具敏感字段常量。"""
from typing import Tuple


SQL_SENSITIVE_COLUMN_KEYWORDS: Tuple[str, ...] = (
    "token",
    "password",
    "passwd",
    "secret",
    "api_key",
    "access_key",
    "refresh_token",
)


SQL_PHONE_COLUMN_KEYWORDS: Tuple[str, ...] = ("phone", "mobile", "tel")

