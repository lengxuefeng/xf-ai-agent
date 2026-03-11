# -*- coding: utf-8 -*-
"""
历史数据迁移脚本：将 MySQL / MongoDB / Redis 数据迁移到 PostgreSQL。

说明：
1. 本文件仅作为离线运维脚本使用，不参与运行时主链路。
2. 所有连接信息均从环境变量读取，避免硬编码账号与地址。
3. 严禁在模块 import 时建立连接，避免副作用。
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import redis
from pymongo import MongoClient
from sqlalchemy import MetaData, create_engine, text


@dataclass
class MigrationContext:
    """迁移上下文，统一持有多数据源连接对象。"""

    mysql_engine: Any
    pg_engine: Any
    mongo_db: Any
    redis_client: redis.Redis


def _read_env(name: str, default: str) -> str:
    """读取环境变量（空值回退默认值）。"""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def build_context_from_env() -> MigrationContext:
    """根据环境变量构建迁移上下文。"""
    mysql_uri = _read_env("MIGRATE_MYSQL_URI", "mysql+pymysql://root:root@127.0.0.1:3306/xf_ai_agent")
    mongo_uri = _read_env("MIGRATE_MONGO_URI", "mongodb://127.0.0.1:27017")
    mongo_db_name = _read_env("MIGRATE_MONGO_DB", "xf_ai_agent")
    redis_host = _read_env("MIGRATE_REDIS_HOST", "127.0.0.1")
    redis_port = int(_read_env("MIGRATE_REDIS_PORT", "6379"))
    redis_pwd = _read_env("MIGRATE_REDIS_PASSWORD", "")
    redis_db = int(_read_env("MIGRATE_REDIS_DB", "0"))
    pg_uri = _read_env("MIGRATE_PG_URI", "postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/xf_ai_agent")

    mysql_engine = create_engine(mysql_uri)
    pg_engine = create_engine(pg_uri)
    mongo_client = MongoClient(mongo_uri)
    redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        password=redis_pwd or None,
        db=redis_db,
        decode_responses=True,
    )
    mongo_db = mongo_client[mongo_db_name]
    return MigrationContext(
        mysql_engine=mysql_engine,
        pg_engine=pg_engine,
        mongo_db=mongo_db,
        redis_client=redis_client,
    )


def reset_pg_sequence(pg_conn, table_name: str, pk_column: str = "id"):
    """修复 PgSQL 手动插入数据后的自增序列不同步问题。"""
    try:
        sql = (
            f"SELECT setval(pg_get_serial_sequence('{table_name}', '{pk_column}'), "
            f"COALESCE(MAX({pk_column}), 1), MAX({pk_column}) IS NOT NULL) FROM {table_name};"
        )
        pg_conn.execute(text(sql))
    except Exception:
        # 序列修复失败不影响主迁移流程
        pass


def migrate_mysql(ctx: MigrationContext):
    """迁移 MySQL 核心业务表。"""
    print("\n🚀 [1/3] 开始迁移 MySQL 核心业务数据...")
    mysql_meta = MetaData()
    mysql_meta.reflect(bind=ctx.mysql_engine)
    tables_to_migrate = ["t_user_info", "t_model_setting", "t_user_model", "t_user_mcp"]

    with ctx.pg_engine.begin() as pg_conn:
        with ctx.mysql_engine.connect() as mysql_conn:
            for table_name in tables_to_migrate:
                if table_name not in mysql_meta.tables:
                    print(f"⚠️ 跳过 {table_name}: MySQL 中不存在此表")
                    continue

                print(f"   -> 正在迁移表: {table_name}")
                result = mysql_conn.execute(text(f"SELECT * FROM {table_name}"))

                rows: List[Dict[str, Any]] = []
                for row in result:
                    row_dict = dict(row._mapping)
                    for k, v in row_dict.items():
                        # 处理 MySQL TINYINT 到 PgSQL BOOLEAN 的转换
                        if k.startswith("is_") and v in [0, 1]:
                            row_dict[k] = bool(v)
                    rows.append(row_dict)

                if not rows:
                    print("      (表为空，跳过)")
                    continue

                columns = list(rows[0].keys())
                cols_str = ", ".join(columns)
                vals_str = ", ".join([f":{c}" for c in columns])
                pk_field = "id" if "id" in columns else columns[0]

                insert_sql = text(
                    f"INSERT INTO {table_name} ({cols_str}) VALUES ({vals_str}) "
                    f"ON CONFLICT ({pk_field}) DO NOTHING"
                )
                pg_conn.execute(insert_sql, rows)

                if "id" in columns:
                    reset_pg_sequence(pg_conn, table_name)

                print(f"      ✅ 成功迁移 {len(rows)} 条记录")


def migrate_mongodb(ctx: MigrationContext):
    """迁移 MongoDB 聊天会话与消息。"""
    print("\n🚀 [2/3] 开始迁移 MongoDB 聊天记忆库...")
    with ctx.pg_engine.begin() as pg_conn:
        pg_conn.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS t_chat_session (
                id BIGSERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                session_id VARCHAR(100) UNIQUE NOT NULL,
                title VARCHAR(200) NOT NULL DEFAULT '新对话',
                is_deleted BOOLEAN DEFAULT FALSE,
                create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            )
        )
        pg_conn.execute(
            text(
                """
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
            """
            )
        )

        sessions = list(ctx.mongo_db["chat_sessions"].find())
        if sessions:
            session_data = []
            for doc in sessions:
                session_data.append(
                    {
                        "user_id": doc.get("user_id"),
                        "session_id": doc.get("session_id"),
                        "title": doc.get("title", "历史对话"),
                        "is_deleted": doc.get("is_deleted", 0) == 1,
                        "create_time": doc.get("created_at") or doc.get("updated_at"),
                    }
                )
            insert_sess_sql = text(
                """
                INSERT INTO t_chat_session (user_id, session_id, title, is_deleted, create_time)
                VALUES (:user_id, :session_id, :title, :is_deleted, :create_time)
                ON CONFLICT (session_id) DO NOTHING
                """
            )
            pg_conn.execute(insert_sess_sql, session_data)
            reset_pg_sequence(pg_conn, "t_chat_session")
            print(f"      ✅ 成功迁移 {len(sessions)} 个聊天会话")

        messages = list(ctx.mongo_db["chat_messages"].find())
        if messages:
            message_data = []
            for doc in messages:
                extra_data = {"old_mongo_id": str(doc["_id"]), "tool_calls": doc.get("tool_calls", [])}
                message_data.append(
                    {
                        "user_id": doc.get("user_id"),
                        "session_id": doc.get("session_id"),
                        "user_content": doc.get("user_content", ""),
                        "model_content": doc.get("model_content", ""),
                        "model_name": doc.get("model", ""),
                        "tokens": doc.get("tokens", 0),
                        "latency_ms": doc.get("latency_ms", 0),
                        "is_deleted": doc.get("is_deleted", 0) == 1,
                        "extra_data": json.dumps(extra_data, ensure_ascii=False),
                        "create_time": doc.get("created_at"),
                    }
                )
            insert_msg_sql = text(
                """
                INSERT INTO t_chat_message
                (user_id, session_id, user_content, model_content, model_name, tokens, latency_ms, is_deleted, extra_data, create_time)
                VALUES
                (:user_id, :session_id, :user_content, :model_content, :model_name, :tokens, :latency_ms, :is_deleted, CAST(:extra_data AS JSONB), :create_time)
                """
            )
            pg_conn.execute(insert_msg_sql, message_data)
            reset_pg_sequence(pg_conn, "t_chat_message")
            print(f"      ✅ 成功迁移 {len(messages)} 条聊天记录")


def migrate_redis(ctx: MigrationContext):
    """迁移 Redis 字符串缓存到 PostgreSQL。"""
    print("\n🚀 [3/3] 开始迁移 Redis 缓存池...")
    try:
        keys = ctx.redis_client.keys("*")
        if not keys:
            print("   -> Redis 为空，无需迁移")
            return

        with ctx.pg_engine.begin() as pg_conn:
            pg_conn.execute(
                text(
                    """
                CREATE UNLOGGED TABLE IF NOT EXISTS t_sys_cache (
                    cache_key VARCHAR(255) PRIMARY KEY,
                    cache_value TEXT NOT NULL,
                    expire_time TIMESTAMP NULL
                )
                """
                )
            )
            cache_data = []
            for key in keys:
                if ctx.redis_client.type(key) == "string":
                    val = ctx.redis_client.get(key)
                    ttl = ctx.redis_client.ttl(key)
                    if ttl > 0 or ttl == -1:
                        cache_data.append({"cache_key": key, "cache_value": val})

            if cache_data:
                insert_cache_sql = text(
                    """
                    INSERT INTO t_sys_cache (cache_key, cache_value)
                    VALUES (:cache_key, :cache_value)
                    ON CONFLICT (cache_key) DO UPDATE SET cache_value = EXCLUDED.cache_value
                    """
                )
                pg_conn.execute(insert_cache_sql, cache_data)
                print(f"      ✅ 成功迁移 {len(cache_data)} 个缓存键值对")
    except Exception as exc:
        print(f"   ⚠️ Redis 迁移跳过: {exc}")


def run_all_migrations():
    """执行全量迁移。"""
    print("初始化全能数据迁移引擎...")
    ctx = build_context_from_env()
    migrate_mysql(ctx)
    migrate_mongodb(ctx)
    migrate_redis(ctx)
    print("\n🎉 [完成] 所有可迁移数据已处理。")


if __name__ == "__main__":
    try:
        run_all_migrations()
    except Exception as exc:
        print(f"\n❌ 迁移发生错误: {exc}")

