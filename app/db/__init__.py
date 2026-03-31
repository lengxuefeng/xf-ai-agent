# -*- coding: utf-8 -*-
import os
from typing import Generator
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from common.utils.custom_logger import get_logger

log = get_logger(__name__)
load_dotenv()

"""
【学习笔记】PgSQL 数据库引擎与会话管理
在此处统一使用 psycopg2 驱动连接 PostgreSQL。
"""

# 读取 PgSQL 配置 (请确保 .env 中已配置这些变量)
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")
PG_USER = os.getenv("POSTGRES_USER", "postgres")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
PG_DATABASE = os.getenv("POSTGRES_DB", "xf_ai_agent")

SQLALCHEMY_DATABASE_URL = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

# 创建企业级 PgSQL 引擎
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,  # 每次借出连接前 ping 一下数据库，防止断连报错
    pool_recycle=3600,   # 1小时强制回收，保持连接新鲜度
    pool_size=20,        # 常驻连接池大小
    max_overflow=30      # 流量突发时允许额外溢出的连接数
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 所有 SQLAlchemy 模型的基类
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    【场景 A：FastAPI 路由专用】
    用法：@app.get(...) def endpoint(db: Session = Depends(get_db)):
    """
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        log.error(f"数据库事务异常回滚: {e}")
        raise
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    【场景 B：后台任务/Service 独立调用专用上下文】
    用法：with get_db_context() as db: ...
    """
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        log.error(f"后台上下文事务异常回滚: {e}")
        raise
    finally:
        db.close()