import os
import re
import copy
import hashlib
import json
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
from contextlib import contextmanager

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from config.runtime_settings import (
    YUNYOU_DB_POOL_CONFIG,
    YUNYOU_HTTP_CONFIG,
    YUNYOU_TABLE_DISCOVERY_CACHE_TTL_SECONDS,
)
from services.request_cancellation_service import request_cancellation_service
from utils.custom_logger import get_logger

log = get_logger(__name__)

load_dotenv()


class YunYouTools:
    """云柚HTTP工具封装"""

    _state_lock = threading.RLock()
    _circuit_state: Dict[str, Dict[str, float]] = {}
    _response_cache: Dict[str, Tuple[float, float, Dict[str, Any]]] = {}

    def __init__(self):
        """初始化云柚服务基础地址"""
        self.base_url = os.getenv("YY_BASE_URL")

    @staticmethod
    def _build_endpoint_key(url: str) -> str:
        return str(url or "").strip().lstrip("/") or "/"

    @staticmethod
    def _build_cache_key(endpoint: str, params: Dict[str, Any]) -> str:
        stable_payload = json.dumps(
            {"endpoint": endpoint, "params": params},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.md5(stable_payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _ensure_not_cancelled(full_url: str) -> None:
        if request_cancellation_service.is_cancelled():
            raise ValueError(f"请求已取消：跳过调用 {full_url}")

    @classmethod
    def _get_cached_response(cls, cache_key: str, *, allow_stale: bool = False) -> Optional[Dict[str, Any]]:
        now = time.time()
        with cls._state_lock:
            cache_entry = cls._response_cache.get(cache_key)
            if not cache_entry:
                return None
            fresh_until, stale_until, cached_data = cache_entry
            if now <= fresh_until:
                return copy.deepcopy(cached_data)
            if allow_stale and now <= stale_until:
                return copy.deepcopy(cached_data)
            if now > stale_until:
                cls._response_cache.pop(cache_key, None)
        return None

    @classmethod
    def _set_cached_response(cls, cache_key: str, payload: Dict[str, Any]) -> None:
        now = time.time()
        fresh_until = now + int(YUNYOU_HTTP_CONFIG.cache_ttl_seconds)
        stale_until = fresh_until + int(YUNYOU_HTTP_CONFIG.cache_stale_seconds)
        with cls._state_lock:
            cls._response_cache[cache_key] = (fresh_until, stale_until, copy.deepcopy(payload))

    @classmethod
    def _is_circuit_open(cls, endpoint: str) -> bool:
        now = time.time()
        with cls._state_lock:
            state = cls._circuit_state.get(endpoint)
            if not state:
                return False
            opened_until = float(state.get("opened_until") or 0.0)
            if opened_until <= 0:
                return False
            if opened_until <= now:
                # 熔断窗口结束，自动进入半开态（重置失败计数）
                state["opened_until"] = 0.0
                state["failures"] = 0.0
                return False
            return True

    @classmethod
    def _record_success(cls, endpoint: str) -> None:
        with cls._state_lock:
            cls._circuit_state[endpoint] = {"failures": 0.0, "opened_until": 0.0}

    @classmethod
    def _record_failure(cls, endpoint: str, *, reason: str) -> None:
        now = time.time()
        threshold = max(1, int(YUNYOU_HTTP_CONFIG.circuit_breaker_threshold))
        open_seconds = max(1, int(YUNYOU_HTTP_CONFIG.circuit_open_seconds))
        with cls._state_lock:
            state = cls._circuit_state.setdefault(endpoint, {"failures": 0.0, "opened_until": 0.0})
            failures = int(state.get("failures") or 0) + 1
            state["failures"] = float(failures)
            if failures >= threshold:
                state["opened_until"] = now + open_seconds
                state["failures"] = 0.0
                log.warning(
                    f"云柚接口触发熔断: endpoint={endpoint}, open_seconds={open_seconds}, reason={reason}"
                )

    def common_post(self, url: str, params: Dict) -> Dict:
        """执行云柚POST请求"""
        if not self.base_url:
            raise ValueError("未配置 YY_BASE_URL，无法调用云柚 API。")
        headers = {
            "Content-Type": "application/json",
            # 增加通用请求头，避免服务端因头信息不全拒绝
            "User-Agent": YUNYOU_HTTP_CONFIG.user_agent,
            "Connection": YUNYOU_HTTP_CONFIG.connection_header,  # 关闭长连接，避免连接池问题
        }

        filtered_params = {k: v for k, v in params.items() if v is not None}

        # 拼接URL（强制处理末尾/，避免拼接成 http://xxx:8089/holter//list 这类错误）
        endpoint = self._build_endpoint_key(url)
        full_url = f"{self.base_url.rstrip('/')}/{endpoint}"
        proxies = {"http": None, "https": None} if YUNYOU_HTTP_CONFIG.disable_proxy else None
        # 连接超时使用更短预算，读超时沿用总预算，避免长时间“假死”。
        connect_timeout = max(1, min(5, int(YUNYOU_HTTP_CONFIG.timeout_seconds / 2)))
        read_timeout = max(connect_timeout, int(YUNYOU_HTTP_CONFIG.timeout_seconds))

        cache_key = self._build_cache_key(endpoint, filtered_params)
        fresh_cache = self._get_cached_response(cache_key, allow_stale=False)
        if fresh_cache is not None:
            log.info(f"云柚接口命中缓存: endpoint={endpoint}")
            return fresh_cache

        if self._is_circuit_open(endpoint):
            stale_cache = self._get_cached_response(cache_key, allow_stale=True)
            if stale_cache is not None:
                log.warning(f"云柚接口处于熔断窗口，返回陈旧缓存: endpoint={endpoint}")
                return stale_cache
            raise ValueError(f"云柚服务暂时不可用（熔断中）：{full_url}")

        attempts = max(1, int(YUNYOU_HTTP_CONFIG.retry_attempts))
        last_error: Optional[ValueError] = None
        should_retry = True
        for attempt_idx in range(1, attempts + 1):
            self._ensure_not_cancelled(full_url)
            response = None
            should_retry = True
            try:
                # 禁用重定向+超时+关闭代理+验证SSL（内网可关）
                response = requests.post(
                    full_url,
                    json=filtered_params,
                    headers=headers,
                    proxies=proxies,
                    allow_redirects=YUNYOU_HTTP_CONFIG.allow_redirects,
                    timeout=(connect_timeout, read_timeout),
                    verify=YUNYOU_HTTP_CONFIG.verify_ssl,
                )
                response.raise_for_status()
                data = response.json()
                payload = data.get("data", {})
                self._set_cached_response(cache_key, payload)
                self._record_success(endpoint)
                return payload
            except requests.exceptions.HTTPError as http_err:
                status_code = int(getattr(response, "status_code", 0) or 0)
                err_preview = ((response.text or "")[:YUNYOU_HTTP_CONFIG.error_preview_chars]) if response is not None else ""
                last_error = ValueError(f"HTTP错误 {status_code}: {http_err}\n响应内容: {err_preview}")
                # 4xx（除 429）通常是请求参数问题，不做重试。
                should_retry = not (400 <= status_code < 500 and status_code != 429)
            except requests.exceptions.ConnectTimeout:
                last_error = ValueError(f"连接超时：无法访问 {full_url}，请检查服务器是否在线")
                should_retry = True
            except requests.exceptions.ReadTimeout:
                last_error = ValueError(f"读取超时：{full_url} 响应过慢，请稍后重试")
                should_retry = True
            except requests.exceptions.ConnectionError:
                last_error = ValueError(f"连接失败：无法连接到 {full_url}，请检查IP/端口是否正确")
                should_retry = True
            except Exception as e:
                last_error = ValueError(f"查询时发生未知错误: {str(e)}")
                should_retry = True

            if attempt_idx < attempts and should_retry:
                backoff_sec = (YUNYOU_HTTP_CONFIG.retry_backoff_ms / 1000.0) * attempt_idx
                log.warning(
                    f"云柚接口调用失败，准备重试: endpoint={endpoint}, attempt={attempt_idx}/{attempts}, "
                    f"backoff={backoff_sec:.2f}s, error={last_error}"
                )
                if request_cancellation_service.is_cancelled():
                    raise ValueError(f"请求已取消：跳过重试 {full_url}")
                time.sleep(backoff_sec)
                continue
            break

        self._record_failure(endpoint, reason=str(last_error or "unknown"))
        stale_cache = self._get_cached_response(cache_key, allow_stale=True)
        if stale_cache is not None:
            log.warning(f"云柚接口失败，已回退陈旧缓存: endpoint={endpoint}, error={last_error}")
            return stale_cache

        raise last_error or ValueError(f"查询失败：{full_url}")


class YunyouDbTools:
    """云柚业务库查询工具（只读）"""

    _engine = None
    _session_factory = None
    _table_cache_lock = threading.RLock()
    _holter_table_cache: Dict[str, Any] = {"expire_at": 0.0, "tables": []}
    # 查询 Holter 列表时要求最少具备的字段
    _MIN_REQUIRED_COLUMNS = {"id", "user_id", "use_day"}
    # 用于展示/过滤的常用字段
    _COMMON_HOLTER_COLUMNS = {
        "id",
        "user_id",
        "use_day",
        "begin_date_time",
        "end_date_time",
        "is_uploaded",
        "report_status",
        "holter_type",
        "add_time",
        "update_time",
    }

    @classmethod
    def _get_db_url(cls) -> str:
        """获取云柚数据库连接URL"""
        # 推荐显式配置云柚业务库地址
        db_url = (
            os.getenv("YUNYOU_DB_URL")
            or os.getenv("YY_DB_URL")
            or os.getenv("YUNYOU_DATABASE_URL")
        )
        if db_url:
            return db_url

        # 兜底：复用主库配置（仅用于开发验证）
        pg_host = os.getenv("POSTGRES_HOST")
        pg_port = os.getenv("POSTGRES_PORT")
        pg_user = os.getenv("POSTGRES_USER")
        pg_pwd = os.getenv("POSTGRES_PASSWORD")
        pg_db = os.getenv("POSTGRES_DB")
        if all([pg_host, pg_port, pg_user, pg_db]):
            return f"postgresql+psycopg2://{pg_user}:{pg_pwd}@{pg_host}:{pg_port}/{pg_db}"

        raise ValueError("未配置云柚数据库连接。请设置 YUNYOU_DB_URL（推荐）。")

    @classmethod
    def _init_engine(cls):
        """初始化数据库连接池（单例）"""
        if cls._engine is not None and cls._session_factory is not None:
            return
        db_url = cls._get_db_url()
        cls._engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_recycle=YUNYOU_DB_POOL_CONFIG.pool_recycle_seconds,
            pool_size=YUNYOU_DB_POOL_CONFIG.pool_size,
            max_overflow=YUNYOU_DB_POOL_CONFIG.max_overflow,
        )
        cls._session_factory = sessionmaker(autocommit=False, autoflush=False, bind=cls._engine)

    @classmethod
    @contextmanager
    def _session_scope(cls):
        """数据库会话上下文管理器"""
        cls._init_engine()
        session = cls._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _normalize_date(date_text: Optional[str]) -> Optional[str]:
        """将日期格式标准化为YYYY-MM-DD"""
        if not date_text:
            return None
        cleaned = (
            str(date_text).strip()
            .replace("年", "-")
            .replace("月", "-")
            .replace("日", "")
            .replace("/", "-")
            .replace(".", "-")
        )
        try:
            return datetime.strptime(cleaned, "%Y-%m-%d").strftime("%Y-%m-%d")
        except Exception:
            raise ValueError(f"日期格式非法: {date_text}，期望 YYYY-MM-DD")

    @staticmethod
    def _split_table_identifier(table_identifier: str) -> Tuple[Optional[str], str]:
        """拆分 schema.table 标识。"""
        cleaned = str(table_identifier or "").strip()
        if "." not in cleaned:
            return None, cleaned
        schema_name, table_name = cleaned.split(".", 1)
        return schema_name.strip() or None, table_name.strip()

    @staticmethod
    def _safe_quote_identifier(identifier: str) -> str:
        """对 SQL 标识符进行安全校验并加引号。"""
        name = str(identifier or "").strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"非法标识符: {identifier}")
        return f'"{name}"'

    @classmethod
    def _safe_quote_table(cls, table_identifier: str) -> str:
        """将表名（可带 schema）转换为安全可执行 SQL 片段。"""
        schema_name, table_name = cls._split_table_identifier(table_identifier)
        quoted_table = cls._safe_quote_identifier(table_name)
        if schema_name:
            return f"{cls._safe_quote_identifier(schema_name)}.{quoted_table}"
        return quoted_table

    @staticmethod
    def _extract_error_text(exc: Exception) -> str:
        """统一提取异常文本，便于降级判断。"""
        return str(exc or "").strip()

    @classmethod
    def _is_missing_table_error(cls, exc: Exception) -> bool:
        """判断是否为表不存在错误。"""
        msg = cls._extract_error_text(exc).lower()
        return ("undefinedtable" in msg) or ("relation" in msg and "does not exist" in msg)

    @classmethod
    def _is_missing_column_error(cls, exc: Exception) -> bool:
        """判断是否为字段不存在错误。"""
        msg = cls._extract_error_text(exc).lower()
        return ("undefinedcolumn" in msg) or ("column" in msg and "does not exist" in msg)

    @classmethod
    def _discover_holter_tables(cls, session) -> List[str]:
        """
        自动发现云柚库中可用的 Holter 业务表。

        选择策略：
        1. 仅选择表名包含 holter 的业务表；
        2. 至少具备 id/user_id/use_day；
        3. 按“常用字段命中数”从高到低排序。
        """
        sql = text(
            """
            SELECT
                c.table_schema,
                c.table_name,
                lower(c.column_name) AS column_name
            FROM information_schema.columns c
            JOIN information_schema.tables t
              ON c.table_schema = t.table_schema
             AND c.table_name = t.table_name
            WHERE t.table_type = 'BASE TABLE'
              AND c.table_schema NOT IN ('pg_catalog', 'information_schema')
              AND lower(c.table_name) LIKE :table_pattern
            """
        )
        rows = session.execute(sql, {"table_pattern": "%holter%"}).fetchall()
        if not rows:
            return []

        columns_by_table: Dict[Tuple[str, str], set[str]] = {}
        for row in rows:
            mapping = row._mapping
            key = (str(mapping["table_schema"]), str(mapping["table_name"]))
            columns_by_table.setdefault(key, set()).add(str(mapping["column_name"]))

        ranked_tables: List[Tuple[int, str]] = []
        for (schema_name, table_name), columns in columns_by_table.items():
            if not cls._MIN_REQUIRED_COLUMNS.issubset(columns):
                continue
            score = len(columns & cls._COMMON_HOLTER_COLUMNS)
            full_name = f"{schema_name}.{table_name}" if schema_name and schema_name != "public" else table_name
            ranked_tables.append((score, full_name))

        ranked_tables.sort(key=lambda item: (-item[0], item[1]))
        return [table for _, table in ranked_tables]

    @classmethod
    def _get_cached_holter_tables(cls) -> List[str]:
        now = time.time()
        with cls._table_cache_lock:
            expire_at = float(cls._holter_table_cache.get("expire_at") or 0.0)
            if now >= expire_at:
                return []
            cached = cls._holter_table_cache.get("tables") or []
            return [str(x) for x in cached if str(x).strip()]

    @classmethod
    def _set_cached_holter_tables(cls, tables: Sequence[str]) -> None:
        valid_tables = [str(x).strip() for x in (tables or []) if str(x).strip()]
        if not valid_tables:
            return
        with cls._table_cache_lock:
            cls._holter_table_cache = {
                "expire_at": time.time() + int(YUNYOU_TABLE_DISCOVERY_CACHE_TTL_SECONDS),
                "tables": list(valid_tables),
            }

    @classmethod
    def _invalidate_holter_table_cache(cls) -> None:
        with cls._table_cache_lock:
            cls._holter_table_cache = {"expire_at": 0.0, "tables": []}

    @classmethod
    def _build_query_sql(
        cls,
        table_identifier: str,
        where_parts: Sequence[str],
        order_sql: str,
    ) -> str:
        """拼装 Holter 查询 SQL。"""
        quoted_table = cls._safe_quote_table(table_identifier)
        sql = f"""
            SELECT
                id,
                user_id,
                use_day,
                begin_date_time,
                end_date_time,
                is_uploaded,
                report_status,
                holter_type,
                add_time,
                update_time
            FROM {quoted_table}
        """
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)
        sql += f" ORDER BY id {order_sql} LIMIT :limit"
        return sql

    @classmethod
    def query_holter_recent(
        cls,
        limit: int = 5,
        order_desc: bool = True,
        start_use_day: Optional[str] = None,
        end_use_day: Optional[str] = None,
    ) -> Dict:
        """查询最近的Holter记录"""
        if request_cancellation_service.is_cancelled():
            raise ValueError("请求已取消：跳过云柚数据库查询。")
        safe_limit = max(1, min(int(limit or YUNYOU_DB_POOL_CONFIG.holter_default_limit), YUNYOU_DB_POOL_CONFIG.holter_max_limit))
        start_day = cls._normalize_date(start_use_day)
        end_day = cls._normalize_date(end_use_day)

        where_parts = []
        params: Dict[str, object] = {"limit": safe_limit}
        if start_day and end_day:
            where_parts.append("use_day BETWEEN :start_use_day AND :end_use_day")
            params["start_use_day"] = start_day
            params["end_use_day"] = end_day
        elif start_day:
            where_parts.append("use_day >= :start_use_day")
            params["start_use_day"] = start_day
        elif end_day:
            where_parts.append("use_day <= :end_use_day")
            params["end_use_day"] = end_day

        order_sql = "DESC" if order_desc else "ASC"
        with cls._session_scope() as session:
            configured_table = str(YUNYOU_DB_POOL_CONFIG.holter_table_name or "").strip()
            candidate_tables: List[str] = []
            if configured_table:
                candidate_tables.append(configured_table)
            discovered_tables = cls._get_cached_holter_tables()
            if not discovered_tables:
                try:
                    discovered_tables = cls._discover_holter_tables(session)
                    cls._set_cached_holter_tables(discovered_tables)
                except Exception as discover_exc:
                    cls._invalidate_holter_table_cache()
                    log.warning(f"自动发现 Holter 表失败，继续使用配置表名。原因: {discover_exc}")
                    discovered_tables = []
            for table_name in discovered_tables:
                if table_name not in candidate_tables:
                    candidate_tables.append(table_name)

            if not candidate_tables:
                raise ValueError("未发现可用的 Holter 业务表，请检查 YUNYOU_DB_URL 是否连接到云柚业务库。")

            last_error: Optional[str] = None
            for table_name in candidate_tables:
                if request_cancellation_service.is_cancelled():
                    raise ValueError("请求已取消：终止云柚数据库查询。")
                try:
                    sql = cls._build_query_sql(table_name, where_parts, order_sql)
                    result = session.execute(text(sql), params)
                    rows = [dict(row._mapping) for row in result.fetchall()]
                    return {
                        "source": "yunyou_db",
                        "table": table_name,
                        "query_mode": "direct_db",
                        "limit": safe_limit,
                        "order": order_sql,
                        "startUseDay": start_day,
                        "endUseDay": end_day,
                        "rows": rows,
                    }
                except Exception as query_exc:
                    last_error = cls._extract_error_text(query_exc)
                    if cls._is_missing_table_error(query_exc) or cls._is_missing_column_error(query_exc):
                        cls._invalidate_holter_table_cache()
                        log.warning(f"Holter 表候选不可用，继续尝试下一个表: {table_name} | {last_error}")
                        continue
                    # 非“表/字段不存在”错误不再继续尝试，直接抛出。
                    raise

            table_preview = ", ".join(candidate_tables[:5])
            raise ValueError(
                "未在当前数据库找到可查询的 Holter 业务表。"
                f"已尝试表: {table_preview}。"
                f"最后错误: {last_error or 'unknown'}"
            )


@tool
def holter_list(startUseDay: str, endUseDay: str, isUploaded: Optional[int] = None, reportStatus: Optional[int] = None,
                holterType: Optional[int] = None) -> Dict:
    """
    云柚 holter 数据列表，根据具体参数查询holter信息

    Args:
        startUseDay (str): 开始日期，格式如 "2020-07-30"
        endUseDay (str): 结束日期，格式如 "2020-07-30"
        isUploaded (Optional[int]): 数据是否传完 0：否 1：是。如果不需要则忽略。
        reportStatus (Optional[int]): 报告审核状态 0:待审核 1:审核中 2:人工审核完成 3:自动审核完成。如果不需要则忽略。
        holterType (Optional[int]): holter类型 0:24小时 1:2小时 2:24小时（夜间）3:48小时。如果不需要则忽略。
    """
    params = {
        "startUseDay": startUseDay,
        "endUseDay": endUseDay,
        "isUploaded": isUploaded,
        "reportStatus": reportStatus,
        "holterType": holterType,
    }
    return YunYouTools().common_post("holter/list", params)


@tool
def holter_type_count(startUseDay: str, endUseDay: str) -> Dict:
    """
    云柚 获取holter类型统计，根据时间范围查询holter类型统计
    Args:
        startUseDay (str): 开始日期，格式如 "2020-07-30"
        endUseDay (str): 结束日期，格式如 "2020-07-30"
    Returns:
        Dict: 响应数据，包含holter报告状态统计
    """
    params = {
        "startUseDay": startUseDay,
        "endUseDay": endUseDay,
    }
    log.info(f"holter_type_count params startUseDay={startUseDay}, endUseDay={endUseDay}")
    return YunYouTools().common_post("/holter/holterTypeCount", params)


@tool
def holter_report_count(startUseDay: str, endUseDay: str) -> Dict:
    """
    云柚 获取holter报告状态统计，根据时间范围查询holter报告状态统计
    Args:
        startUseDay (str): 开始日期，格式如 "2020-07-30"
        endUseDay (str): 结束日期，格式如 "2020-07-30"
    Returns:
        Dict: 响应数据，包含holter报告状态统计
    """
    params = {
        "startUseDay": startUseDay,
        "endUseDay": endUseDay,
    }
    return YunYouTools().common_post("/holter/holterReportCount", params)


@tool
def holter_recent_db(limit: int = 5, order: str = "desc", startUseDay: Optional[str] = None, endUseDay: Optional[str] = None) -> Dict:
    """
    直接从云柚业务数据库查询 holter 最近记录（t_holter_use_record）。
    该工具支持无日期参数，按 id 排序返回最近 N 条。

    Args:
        limit (int): 返回条数上限，默认 5，最大 200。
        order (str): 排序方向，支持 "desc"/"asc"，默认 "desc"。
        startUseDay (Optional[str]): 开始日期（YYYY-MM-DD），可选。
        endUseDay (Optional[str]): 结束日期（YYYY-MM-DD），可选。
    """
    order_desc = (order or "desc").lower() != "asc"
    return YunyouDbTools.query_holter_recent(
        limit=limit,
        order_desc=order_desc,
        start_use_day=startUseDay,
        end_use_day=endUseDay,
    )
