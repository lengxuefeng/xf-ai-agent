from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
from sqlalchemy.orm import Session

from exceptions.business_exception import BusinessException

# 数据库连接配置（实际使用时替换为真实地址）
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:xiaoleng@localhost:3306/xf-ai-agent?charset=utf8mb4"

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


@contextmanager  # 装饰器，可以更方便地创建自定义的上下文管理器。
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
