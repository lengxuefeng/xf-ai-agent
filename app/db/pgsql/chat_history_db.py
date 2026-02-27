# -*- coding: utf-8 -*-
# migrate_to_pgsql.py
import sys
import os
import json
import redis
from pymongo import MongoClient
from sqlalchemy import create_engine, text, MetaData

# 把 app 目录加入系统路径，防止找不到包
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

print("初始化全能数据迁徙引擎...")

# ================= 数据库配置区 =================
MYSQL_URI = "mysql+pymysql://root:Xiaoleng123!@192.168.1.10:3306/xf-ai-agent"
MONGO_URI = "mongodb://root:xiaoleng@192.168.1.10:27017/xf-ai-agent?authSource=admin"
REDIS_HOST = "192.168.1.10"
REDIS_PORT = 6379
REDIS_PWD = "xiaoleng"
REDIS_DB = 0
PG_URI = "postgresql+psycopg2://postgres:xiaoleng@192.168.1.10:5432/xf_ai_agent"
# =====================================================================

mysql_engine = create_engine(MYSQL_URI)
pg_engine = create_engine(PG_URI)
mongo_client = MongoClient(MONGO_URI)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PWD, db=REDIS_DB, decode_responses=True)
mongo_db = mongo_client["xf_ai_agent"]

# ================= 核心：暴力自动建表 =================
try:
    import models.user_info
    import models.model_setting
    import models.user_model
    import models.user_mcp

    print("🐘 正在 PostgreSQL 中自动创建关系型表结构...")
    models.user_info.UserInfo.metadata.create_all(bind=pg_engine)
    print("✅ 表结构初始化完成！\n")

except Exception as e:
    print(f"⚠️ 导入模型或建表失败，报错详情: {e}")
    sys.exit(1)


def reset_pg_sequence(pg_conn, table_name, pk_column="id"):
    """修复 PgSQL 手动插入数据后的自增序列不同步问题"""
    try:
        sql = f"SELECT setval(pg_get_serial_sequence('{table_name}', '{pk_column}'), COALESCE(MAX({pk_column}), 1), MAX({pk_column}) IS NOT NULL) FROM {table_name};"
        pg_conn.execute(text(sql))
    except Exception:
        pass


def migrate_mysql():
    print("\n🚀 [1/3] 开始迁移 MySQL 核心业务数据...")
    mysql_meta = MetaData()
    mysql_meta.reflect(bind=mysql_engine)

    tables_to_migrate = ['t_user_info', 't_model_setting', 't_user_model', 't_user_mcp']

    with pg_engine.begin() as pg_conn:
        with mysql_engine.connect() as mysql_conn:
            for table_name in tables_to_migrate:
                if table_name not in mysql_meta.tables:
                    print(f"⚠️ 跳过 {table_name}: MySQL 中不存在此表")
                    continue

                print(f"   -> 正在迁移表: {table_name}")
                result = mysql_conn.execute(text(f"SELECT * FROM {table_name}"))

                rows = []
                for row in result:
                    row_dict = dict(row._mapping)
                    for k, v in row_dict.items():
                        # 【核心修复补丁】处理 MySQL TINYINT 到 PgSQL BOOLEAN 的转换
                        if k.startswith('is_') and v in [0, 1]:
                            row_dict[k] = bool(v)
                    rows.append(row_dict)

                if not rows:
                    print(f"      (表为空，跳过)")
                    continue

                columns = list(rows[0].keys())
                cols_str = ", ".join(columns)
                vals_str = ", ".join([f":{c}" for c in columns])

                # PgSQL 的 ON CONFLICT 必须指定主键字段
                pk_field = "id" if "id" in columns else columns[0]
                insert_sql = text(
                    f"INSERT INTO {table_name} ({cols_str}) VALUES ({vals_str}) ON CONFLICT ({pk_field}) DO NOTHING")

                pg_conn.execute(insert_sql, rows)

                # 同步 PgSQL 的序列生成器
                if "id" in columns:
                    reset_pg_sequence(pg_conn, table_name)

                print(f"      ✅ 成功迁移 {len(rows)} 条记录")


def migrate_mongodb():
    print("\n🚀 [2/3] 开始迁移 MongoDB 聊天记忆库...")
    with pg_engine.begin() as pg_conn:
        # 【终极防弹】强行建表保底
        pg_conn.execute(text("""
            CREATE TABLE IF NOT EXISTS t_chat_session (
                id BIGSERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                session_id VARCHAR(100) UNIQUE NOT NULL,
                title VARCHAR(200) NOT NULL DEFAULT '新对话',
                is_deleted BOOLEAN DEFAULT FALSE,
                create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        pg_conn.execute(text("""
            CREATE TABLE IF NOT EXISTS t_chat_message (
                id BIGSERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                session_id VARCHAR(100) NOT NULL,
                user_content TEXT NOT NULL,
                model_content TEXT NOT NULL,
                model_name VARCHAR(100),
                tokens BIGINT DEFAULT 0,
                latency_ms BIGINT DEFAULT 0,
                is_deleted BOOLEAN DEFAULT FALSE,
                extra_data JSONB,
                create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # 1. 迁移会话
        sessions = list(mongo_db["chat_sessions"].find())
        if sessions:
            session_data = []
            for doc in sessions:
                session_data.append({
                    "user_id": doc.get("user_id"),
                    "session_id": doc.get("session_id"),
                    "title": doc.get("title", "历史对话"),
                    "is_deleted": doc.get("is_deleted", 0) == 1,
                    "create_time": doc.get("created_at") or doc.get("updated_at")
                })
            insert_sess_sql = text("""
                INSERT INTO t_chat_session (user_id, session_id, title, is_deleted, create_time) 
                VALUES (:user_id, :session_id, :title, :is_deleted, :create_time)
                ON CONFLICT (session_id) DO NOTHING
            """)
            pg_conn.execute(insert_sess_sql, session_data)
            reset_pg_sequence(pg_conn, "t_chat_session")
            print(f"      ✅ 成功迁移 {len(sessions)} 个聊天会话")

        # 2. 迁移消息
        messages = list(mongo_db["chat_messages"].find())
        if messages:
            message_data = []
            for doc in messages:
                extra_data = {"old_mongo_id": str(doc["_id"]), "tool_calls": doc.get("tool_calls", [])}
                message_data.append({
                    "user_id": doc.get("user_id"),
                    "session_id": doc.get("session_id"),
                    "user_content": doc.get("user_content", ""),
                    "model_content": doc.get("model_content", ""),
                    "model_name": doc.get("model", ""),
                    "tokens": doc.get("tokens", 0),
                    "latency_ms": doc.get("latency_ms", 0),
                    "is_deleted": doc.get("is_deleted", 0) == 1,
                    "extra_data": json.dumps(extra_data),
                    "create_time": doc.get("created_at")
                })
            insert_msg_sql = text("""
                INSERT INTO t_chat_message 
                (user_id, session_id, user_content, model_content, model_name, tokens, latency_ms, is_deleted, extra_data, create_time) 
                VALUES 
                (:user_id, :session_id, :user_content, :model_content, :model_name, :tokens, :latency_ms, :is_deleted, CAST(:extra_data AS JSONB), :create_time)
            """)
            pg_conn.execute(insert_msg_sql, message_data)
            reset_pg_sequence(pg_conn, "t_chat_message")
            print(f"      ✅ 成功迁移 {len(messages)} 条聊天记录")


def migrate_redis():
    print("\n🚀 [3/3] 开始迁移 Redis 缓存池...")
    try:
        keys = redis_client.keys("*")
        if not keys:
            print("   -> Redis 为空，无需迁移")
            return

        with pg_engine.begin() as pg_conn:
            pg_conn.execute(text("""
                CREATE UNLOGGED TABLE IF NOT EXISTS t_sys_cache (
                    cache_key VARCHAR(255) PRIMARY KEY,
                    cache_value TEXT NOT NULL,
                    expire_time TIMESTAMP NULL
                )
            """))
            cache_data = []
            for key in keys:
                if redis_client.type(key) == 'string':
                    val = redis_client.get(key)
                    ttl = redis_client.ttl(key)
                    if ttl > 0 or ttl == -1:
                        cache_data.append({"cache_key": key, "cache_value": val})

            if cache_data:
                insert_cache_sql = text("""
                    INSERT INTO t_sys_cache (cache_key, cache_value) 
                    VALUES (:cache_key, :cache_value)
                    ON CONFLICT (cache_key) DO UPDATE SET cache_value = EXCLUDED.cache_value
                """)
                pg_conn.execute(insert_cache_sql, cache_data)
                print(f"      ✅ 成功迁移 {len(cache_data)} 个缓存键值对")
    except Exception as e:
        print(f"   ⚠️ 忽略 Redis: {e}")


if __name__ == "__main__":
    try:
        migrate_mysql()
        migrate_mongodb()
        migrate_redis()
        print("\n🎉🎉🎉 [大满贯] 所有数据已完美迁入 PostgreSQL！")
    except Exception as e:
        print(f"\n❌ 迁移发生错误: {e}")
