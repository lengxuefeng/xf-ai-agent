from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

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


# 依赖注入：获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
