# -*- coding: utf-8 -*-
import os
import re
from dataclasses import dataclass
from typing import Optional, Set

from config.constants.sql_policy_constants import DEFAULT_LOCAL_TABLES, DEFAULT_YUNYOU_TABLES, SQL_WRITE_KEYWORDS


def _split_csv_env(value: Optional[str], default: str) -> Set[str]:
    """将逗号分隔白名单配置转为小写集合。"""
    raw = value if value is not None else default
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


@dataclass(frozen=True)
class SqlPolicyResult:
    """SQL 策略验证结果。"""

    ok: bool
    reason: str = ""


class SqlPolicyEngine:
    """
    SQL 策略中心。

    能力：
    1. 只读语句约束（阻止 DDL/DML）。
    2. 表级白名单（按数据域区分）。
    """

    WRITE_KEYWORDS = SQL_WRITE_KEYWORDS
    TABLE_PATTERN = re.compile(r"\b(?:from|join|update|into)\s+([a-zA-Z_][a-zA-Z0-9_\.]*)", re.IGNORECASE)

    def __init__(self):
        """初始化本地库与云柚库的表白名单。"""
        self.local_whitelist = _split_csv_env(
            os.getenv("SQL_LOCAL_TABLE_WHITELIST"),
            DEFAULT_LOCAL_TABLES,
        )
        self.yunyou_whitelist = _split_csv_env(
            os.getenv("SQL_YUNYOU_TABLE_WHITELIST"),
            DEFAULT_YUNYOU_TABLES,
        )

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        """清理 Markdown 包裹后的 SQL 文本。"""
        if not sql:
            return ""
        cleaned = sql.strip().replace("```sql", "").replace("```", "").strip()
        return cleaned

    @classmethod
    def extract_tables(cls, sql: str) -> Set[str]:
        """提取 SQL 中出现的表名（from/join/update/into）。"""
        t = cls._normalize_sql(sql)
        found = set()
        for table in cls.TABLE_PATTERN.findall(t):
            found.add(table.split(".")[-1].lower())
        return found

    @classmethod
    def _first_keyword(cls, sql: str) -> str:
        """获取 SQL 首关键字，用于判断读写类型。"""
        normalized = cls._normalize_sql(sql)
        if not normalized:
            return ""
        return re.split(r"\s+", normalized, maxsplit=1)[0].lower()

    @classmethod
    def _is_write_sql(cls, sql: str) -> bool:
        """判断是否为写操作 SQL。"""
        first = cls._first_keyword(sql)
        return first in cls.WRITE_KEYWORDS

    def get_whitelist(self, domain: str) -> Set[str]:
        """按数据域获取表白名单。"""
        d = (domain or "LOCAL_DB").upper()
        if d == "YUNYOU_DB":
            return self.yunyou_whitelist
        return self.local_whitelist

    def validate(self, sql: str, domain: str = "LOCAL_DB") -> SqlPolicyResult:
        """执行 SQL 策略校验并返回结果对象。"""
        normalized = self._normalize_sql(sql)
        if not normalized:
            return SqlPolicyResult(ok=False, reason="SQL 为空")

        if self._is_write_sql(normalized):
            return SqlPolicyResult(ok=False, reason="策略禁止写操作，仅允许只读 SQL")

        tables = self.extract_tables(normalized)
        if not tables:
            return SqlPolicyResult(ok=True, reason="")

        whitelist = self.get_whitelist(domain)
        disallowed = sorted([t for t in tables if t not in whitelist])
        if disallowed:
            return SqlPolicyResult(
                ok=False,
                reason=f"命中未授权表: {', '.join(disallowed)}。当前数据域白名单: {', '.join(sorted(whitelist))}",
            )

        return SqlPolicyResult(ok=True, reason="")

    def enforce(self, sql: str, domain: str = "LOCAL_DB"):
        """策略不通过时抛出异常，供工具层快速阻断。"""
        result = self.validate(sql, domain=domain)
        if not result.ok:
            raise PermissionError(result.reason)


sql_policy_engine = SqlPolicyEngine()
