# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy import text

from agent.policy.sql_policy_engine import sql_policy_engine
from db import get_db_context
from services.semantic_cache_service import semantic_cache_service
from utils.custom_logger import get_logger, LogTarget

log = get_logger(__name__)


def _normalize_sql(query: str) -> str:
    """清理模型输出，去掉 markdown 包裹。"""
    if not query:
        return ""
    cleaned = query.strip()
    cleaned = cleaned.replace("```sql", "").replace("```", "").strip()
    return cleaned


def execute_sql(query: str, domain: str = "LOCAL_DB") -> str:
    """
    执行 SQL 查询并返回结构化文本结果。
    支持：
    - SELECT / WITH / EXPLAIN（返回查询结果）
    - INSERT / UPDATE / DELETE（返回影响行数）
    """
    sql = _normalize_sql(query)
    if not sql:
        return "SQL 为空，无法执行。"

    try:
        # 表级白名单与只读策略校验
        sql_policy_engine.enforce(sql, domain=domain)

        cache_key = semantic_cache_service.build_key(domain=domain, text=sql)
        cached = semantic_cache_service.get(cache_key)
        if cached is not None:
            log.info(f"SQL 语义缓存命中: domain={domain}", target=LogTarget.LOG)
            return cached

        with get_db_context() as db:
            first_keyword = re.split(r"\s+", sql, maxsplit=1)[0].upper()

            if first_keyword in {"SELECT", "WITH", "EXPLAIN"}:
                result = db.execute(text(sql))
                rows = result.fetchall()
                if not rows:
                    no_data = "查询成功，但没有返回结果。"
                    semantic_cache_service.set(cache_key, no_data, ttl_seconds=30)
                    return no_data

                # 优先返回 JSON，方便后续模型总结时稳定解析
                as_dict = [dict(row._mapping) for row in rows]
                payload = json.dumps(as_dict, ensure_ascii=False, default=str)
                semantic_cache_service.set(cache_key, payload, ttl_seconds=120)
                return payload

            result = db.execute(text(sql))
            affected = result.rowcount if result.rowcount is not None else 0
            outcome = f"操作成功，影响行数: {affected}"
            semantic_cache_service.set(cache_key, outcome, ttl_seconds=30)
            return outcome

    except Exception as e:
        return f"执行 SQL 时发生错误: {e}"


def _escape_cell(val: Any, max_len: int = 80) -> str:
    s = "" if val is None else str(val)
    s = s.replace("\n", " ").replace("|", "\\|").strip()
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _is_sensitive_column(col: str) -> bool:
    c = (col or "").lower()
    keywords = ("token", "password", "passwd", "secret", "api_key", "access_key", "refresh_token")
    return any(k in c for k in keywords)


def _mask_phone_if_needed(col: str, val: Any) -> Any:
    c = (col or "").lower()
    if not any(k in c for k in ("phone", "mobile", "tel")):
        return val
    s = str(val) if val is not None else ""
    digits = re.sub(r"\D", "", s)
    if len(digits) == 11:
        return f"{digits[:3]}****{digits[-4:]}"
    return val


def format_sql_result_for_user(sql: str, raw_result: str, max_rows: int = 10) -> str:
    """
    将 SQL 原始执行结果格式化为用户可读文本。
    - 查询结果(JSON 数组) -> Markdown 表格 + 行数摘要
    - 影响行数/无结果/错误 -> 友好文案
    """
    normalized_sql = _normalize_sql(sql)
    result_text = (raw_result or "").strip()

    if not result_text:
        return "查询完成，但没有可展示的结果。"

    if result_text.startswith("执行 SQL 时发生错误"):
        return f"❌ SQL 执行失败\n\n{result_text}"

    if result_text.startswith("SQL 为空"):
        return f"❌ {result_text}"

    if result_text.startswith("操作成功，影响行数"):
        return "\n".join([
            "✅ SQL 执行成功",
            f"- 语句: `{normalized_sql}`" if normalized_sql else "- 语句: （未提供）",
            f"- {result_text}",
        ])

    if result_text.startswith("查询成功，但没有返回结果"):
        return "\n".join([
            "✅ 查询执行成功",
            f"- 语句: `{normalized_sql}`" if normalized_sql else "- 语句: （未提供）",
            "- 返回 0 行数据",
        ])

    try:
        payload = json.loads(result_text)
    except Exception:
        return "\n".join([
            "✅ SQL 已执行",
            f"- 语句: `{normalized_sql}`" if normalized_sql else "- 语句: （未提供）",
            "",
            "结果：",
            result_text,
        ])

    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or (payload and not isinstance(payload[0], dict)):
        return "\n".join([
            "✅ SQL 已执行",
            f"- 语句: `{normalized_sql}`" if normalized_sql else "- 语句: （未提供）",
            "",
            "结果：",
            result_text,
        ])

    total = len(payload)
    preview = payload[:max_rows]
    if not preview:
        return "\n".join([
            "✅ 查询执行成功",
            f"- 语句: `{normalized_sql}`" if normalized_sql else "- 语句: （未提供）",
            "- 返回 0 行数据",
        ])

    all_cols = list(preview[0].keys())
    for row in preview[1:]:
        for k in row.keys():
            if k not in all_cols:
                all_cols.append(k)

    hidden_cols = [c for c in all_cols if _is_sensitive_column(c)]
    visible_cols = [c for c in all_cols if c not in hidden_cols]
    if not visible_cols:
        visible_cols = all_cols[: min(5, len(all_cols))]
        hidden_cols = [c for c in all_cols if c not in visible_cols]

    header = "| " + " | ".join(visible_cols) + " |"
    sep = "| " + " | ".join(["---"] * len(visible_cols)) + " |"
    rows = []
    for row in preview:
        cells = []
        for col in visible_cols:
            masked = _mask_phone_if_needed(col, row.get(col))
            cells.append(_escape_cell(masked))
        rows.append("| " + " | ".join(cells) + " |")

    out = [
        "✅ 查询执行成功",
        f"- 语句: `{normalized_sql}`" if normalized_sql else "- 语句: （未提供）",
        f"- 返回行数: {total}",
        "",
        f"数据预览（前 {len(preview)} 行）：",
        header,
        sep,
        *rows,
    ]
    if hidden_cols:
        out.extend([
            "",
            f"说明：已自动隐藏敏感字段：{', '.join(hidden_cols)}",
        ])
    if total > len(preview):
        out.append(f"说明：仅展示前 {len(preview)} 行。")
    return "\n".join(out)


def get_schema() -> str:
    """
    获取 PostgreSQL public schema 下的表结构信息。
    """
    try:
        with get_db_context() as db:
            table_sql = text(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name;
                """
            )
            tables = [row[0] for row in db.execute(table_sql).fetchall()]
            if not tables:
                return "当前数据库没有可用业务表。"

            schema_lines: list[str] = ["数据库表结构如下："]
            for table_name in tables:
                cols_sql = text(
                    """
                    SELECT
                        column_name,
                        data_type,
                        is_nullable,
                        column_default
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = :table_name
                    ORDER BY ordinal_position;
                    """
                )
                columns = db.execute(cols_sql, {"table_name": table_name}).fetchall()

                schema_lines.append(f"\n表名: {table_name}")
                schema_lines.append("字段名 | 类型 | 允许为空 | 默认值")
                schema_lines.append("---|---|---|---")
                for col in columns:
                    schema_lines.append(f"{col[0]} | {col[1]} | {col[2]} | {col[3]}")

            return "\n".join(schema_lines)
    except Exception as e:
        return f"获取表结构时发生错误: {e}"
