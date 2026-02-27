# -*- coding: utf-8 -*-
# app/agent/graphs/checkpointer.py
import os
import atexit
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from utils.custom_logger import get_logger

log = get_logger(__name__)

"""
【架构大一统 - 最终修复版】
LangGraph 记忆核心 (Checkpointer)。
通过 atexit 注册确保全局连接池在生命周期内保持活跃。
"""


def _create_checkpointer() -> Any:
    """
    初始化存储后端。
    """
    backend = os.getenv("CHECKPOINTER_BACKEND", "postgres").strip().lower()

    if backend == "memory":
        log.info("Checkpointer: 正在使用内存存储 (InMemorySaver)")
        return InMemorySaver()

    if backend == "postgres":
        pg_user = os.getenv("POSTGRES_USER", "postgres")
        pg_pwd = os.getenv("POSTGRES_PASSWORD", "xiaoleng")
        pg_host = os.getenv("POSTGRES_HOST", "192.168.1.10")
        pg_port = os.getenv("POSTGRES_PORT", "5432")
        pg_db = os.getenv("POSTGRES_DB", "xf_ai_agent")

        # 构造连接串
        postgres_uri = f"postgresql://{pg_user}:{pg_pwd}@{pg_host}:{pg_port}/{pg_db}"

        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            log.info(f"Checkpointer: 正在建立 PostgreSQL 全局连接池...")

            # 1. 建立同步连接池上下文管理器
            # 增加 conn_kwargs 优化连接稳定性
            context_manager = PostgresSaver.from_conn_string(postgres_uri)

            # 2. 进入上下文并获取真正的 saver 实例
            # 将其设为全局引用，防止被 Python 垃圾回收关闭
            saver = context_manager.__enter__()

            # 3. 注册退出钩子：确保 FastAPI 关闭时，数据库连接池能正常优雅释放
            atexit.register(lambda: context_manager.__exit__(None, None, None))

            # 4. 执行表结构自动初始化
            # 新版 setup 会自动处理内部事务，不需要传 conn
            saver.setup()

            log.info("Checkpointer: PostgreSQL 状态表校验成功，连接已锁定。")
            return saver

        except ImportError:
            log.error("缺失依赖: uv add langgraph-checkpoint-postgres psycopg[binary]")
            raise
        except Exception as exc:
            log.error(f"Postgres Checkpointer 初始化失败: {exc}")
            raise

    return InMemorySaver()


# 导出单例对象，供 supervisor.py 和 xf_graph.py 引用
checkpointer = _create_checkpointer()