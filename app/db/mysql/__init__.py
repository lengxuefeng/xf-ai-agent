# 数据库连接配置（实际使用时替换为真实地址）
import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

from exceptions.business_exception import BusinessException
load_dotenv()
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "your_mysql_password_here")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "xf-ai-agent")

SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"

# 创建引擎
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,  # 连接前检查可用性
    pool_recycle=3600  # 连接超时回收（秒）
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 全局唯一的 Base 实例
Base = declarative_base()


# @contextmanager  # 添加此装饰器，使生成器支持 with 语句
def get_db():
    """
    获取数据库 Session，并统一管理事务：
    - 正常结束自动 commit
    - 出现异常自动 rollback
    - 最终关闭 Session
    """
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        raise BusinessException(message=f"{e}")
    finally:
        db.close()
