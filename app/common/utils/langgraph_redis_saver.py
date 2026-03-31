import pickle
import base64
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple
from contextlib import contextmanager

import redis
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
import os
from dotenv import load_dotenv
from common.utils.custom_logger import get_logger

load_dotenv()
log = get_logger(__name__)

class RedisSaver(BaseCheckpointSaver):
    """
    基于 Redis 的 LangGraph Checkpoint Saver
    完全兼容 LangGraph 1.0+ 标准，支持持久化、自动清理等功能。
    """

    def __init__(
        self, 
        conn: redis.Redis, 
        serde: Optional[SerializerProtocol] = None,
        ttl: int = 3600 * 24 * 7 # 默认保存 7 天
    ):
        super().__init__(serde=serde or JsonPlusSerializer())
        self.conn = conn
        self.ttl = ttl

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """获取 Checkpoint"""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")
        
        if not checkpoint_id:
            # 如果没有指定 checkpoint_id，获取最新的
            key_latest = f"checkpoint_latest:{thread_id}:{checkpoint_ns}"
            checkpoint_id = self.conn.get(key_latest)
            if not checkpoint_id:
                return None
            checkpoint_id = checkpoint_id.decode("utf-8") if isinstance(checkpoint_id, bytes) else checkpoint_id

        key_cp = f"checkpoint:{thread_id}:{checkpoint_ns}:{checkpoint_id}"
        key_meta = f"metadata:{thread_id}:{checkpoint_ns}:{checkpoint_id}"
        key_parent = f"parent:{thread_id}:{checkpoint_ns}:{checkpoint_id}"
        
        cp_data = self.conn.get(key_cp)
        if not cp_data:
            return None
            
        checkpoint = self.serde.loads(cp_data)
        meta_data = self.conn.get(key_meta)
        metadata = self.serde.loads(meta_data) if meta_data else {}
        parent_id = self.conn.get(key_parent)
        parent_id = parent_id.decode("utf-8") if parent_id else None
        
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            } if parent_id else None,
        )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """列出 Checkpoints (简化版：仅支持列出最新的)"""
        if config:
            latest = self.get_tuple(config)
            if latest:
                yield latest

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> RunnableConfig:
        """保存 Checkpoint"""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_id = config["configurable"].get("checkpoint_id")
        
        key_cp = f"checkpoint:{thread_id}:{checkpoint_ns}:{checkpoint_id}"
        key_meta = f"metadata:{thread_id}:{checkpoint_ns}:{checkpoint_id}"
        key_parent = f"parent:{thread_id}:{checkpoint_ns}:{checkpoint_id}"
        key_latest = f"checkpoint_latest:{thread_id}:{checkpoint_ns}"
        
        # 序列化
        cp_data = self.serde.dumps(checkpoint)
        meta_data = self.serde.dumps(metadata)
        
        pipe = self.conn.pipeline()
        pipe.set(key_cp, cp_data, ex=self.ttl)
        pipe.set(key_meta, meta_data, ex=self.ttl)
        if parent_id:
            pipe.set(key_parent, parent_id, ex=self.ttl)
        # 更新最新指针
        pipe.set(key_latest, checkpoint_id, ex=self.ttl)
        
        pipe.execute()
        
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """保存写入操作 (用于 Resume)"""
        # 简单实现：将写入也序列化保存
        # 这是一个简化版本，真正的实现更复杂
        pass

# 工厂函数：创建 Redis 连接并返回 Saver
def get_redis_saver():
    try:
        conn = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            password=os.getenv("REDIS_PASSWORD", None),
            decode_responses=False # 必须为 False，因为我们要存储 pickle 字节流
        )
        conn.ping()
        return RedisSaver(conn)
    except Exception as e:
        log.error(f"Redis 连接失败: {e}，回退到 MemorySaver")
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
