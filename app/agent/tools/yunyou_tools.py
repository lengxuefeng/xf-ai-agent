import os
from datetime import datetime
from typing import Dict, Optional
from contextlib import contextmanager

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from agent.prompts.yunyou_prompt import YunyouPrompt
from utils.custom_logger import get_logger

log = get_logger(__name__)

load_dotenv()


class YunYouTools:
    def __init__(self):
        self.base_url = os.getenv("YY_BASE_URL")

    def common_post(self, url: str, params: Dict) -> Dict:
        """执行 API 请求并处理错误"""
        try:
            headers = {
                "Content-Type": "application/json",
                # 增加通用请求头，避免服务端因头信息不全拒绝
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Connection": "close"  # 关闭长连接，避免连接池问题
            }

            filtered_params = {k: v for k, v in params.items() if v is not None}

            # 拼接URL（强制处理末尾/，避免拼接成 http://xxx:8089/holter//list 这类错误）
            full_url = f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"
            # 禁用重定向+超时+关闭代理+验证SSL（内网可关）
            response = requests.post(
                full_url,
                json=filtered_params,
                headers=headers,
                proxies={"http": None, "https": None},  # 强制禁用代理
                allow_redirects=False,  # 禁止重定向（502常因重定向异常）
                timeout=30,  # 超时时间，避免卡壳
                verify=False  # 内网服务关闭SSL验证（如果是http可忽略，https需开启）
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", {})

        except requests.exceptions.HTTPError as http_err:
            # 细化错误提示，增加响应内容
            err_msg = f"HTTP错误 {response.status_code}: {http_err}\n响应内容: {response.text[:500]}"
            raise ValueError(err_msg)
        except requests.exceptions.ConnectTimeout:
            raise ValueError(f"连接超时：无法访问 {full_url}，请检查服务器是否在线")
        except requests.exceptions.ConnectionError:
            raise ValueError(f"连接失败：无法连接到 {full_url}，请检查IP/端口是否正确")
        except Exception as e:
            raise ValueError(f"查询时发生未知错误: {str(e)}")


class YunyouDbTools:
    """云柚业务库查询工具（只读）。"""

    _engine = None
    _session_factory = None

    @classmethod
    def _get_db_url(cls) -> str:
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
        if cls._engine is not None and cls._session_factory is not None:
            return
        db_url = cls._get_db_url()
        cls._engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=10,
            max_overflow=10,
        )
        cls._session_factory = sessionmaker(autocommit=False, autoflush=False, bind=cls._engine)

    @classmethod
    @contextmanager
    def _session_scope(cls):
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

    @classmethod
    def query_holter_recent(
        cls,
        limit: int = 5,
        order_desc: bool = True,
        start_use_day: Optional[str] = None,
        end_use_day: Optional[str] = None,
    ) -> Dict:
        """
        直接查询云柚 holter 记录表（t_holter_use_record），支持无日期查询。
        """
        safe_limit = max(1, min(int(limit or 5), 200))
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
        sql = """
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
            FROM t_holter_use_record
        """
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)
        sql += f" ORDER BY id {order_sql} LIMIT :limit"

        with cls._session_scope() as session:
            result = session.execute(text(sql), params)
            rows = [dict(row._mapping) for row in result.fetchall()]
            return {
                "source": "yunyou_db",
                "table": "t_holter_use_record",
                "query_mode": "direct_db",
                "limit": safe_limit,
                "order": order_sql,
                "startUseDay": start_day,
                "endUseDay": end_day,
                "rows": rows,
            }


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
    print(f"holter_type_count params startUseDay: {startUseDay}, endUseDay: {endUseDay}")
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
