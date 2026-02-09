import json
import logging
import os

import redis
from typing import Optional, List, Dict, Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import messages_to_dict, messages_from_dict
from utils.custom_logger import get_logger, LogTarget

load_dotenv()
logger = get_logger(__name__)


class RedisManager:
    """Redis 会话管理器，支持活跃会话列表"""

    def __init__(self, active_set_key: str = "code_agent:active_sessions"):
        self.r = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=int(os.getenv("REDIS_PORT")),
            db=int(os.getenv("REDIS_DB")),
            decode_responses=bool(os.getenv("REDIS_DECODE_RESPONSES")),
            password=os.getenv("REDIS_PASSWORD"),
            socket_timeout=int(os.getenv("REDIS_SOCKET_TIMEOUT", 5)),
            socket_connect_timeout=int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", 5)),
            socket_keepalive=True,
            health_check_interval=30,
        )
        self.ttl = int(os.getenv("REDIS_TTL", 3600))
        self.active_set_key = active_set_key
        try:
            self.r.ping()  # 测试连接
            logger.info("Redis 连接成功")
        except redis.RedisError as e:
            logger.warning(f"Redis 连接失败: {str(e)}")
            # 不抛出异常，允许系统继续运行

    def _key(self, thread_id: str) -> str:
        return f"{self.active_set_key}:{thread_id}"

    def load(self, thread_id: str) -> Optional[TypedDict]:
        """从 Redis 加载会话"""
        raw = self.r.get(self._key(thread_id))
        if raw:
            data = json.loads(raw)
            data["messages"] = messages_from_dict(data["messages"])
            return data
        return None

    def save(self, thread_id: str, state: TypedDict):
        """保存会话到 Redis，并设置过期时间"""
        data = {
            "messages": messages_to_dict(state["messages"]),
            "interrupt": state.get("interrupt", "")
        }
        self.r.setex(self._key(thread_id), self.ttl, json.dumps(data))
        # 加入活跃会话集合
        self.r.sadd(self.active_set_key, thread_id)

    def delete(self, thread_id: str):
        """删除会话"""
        self.r.delete(self._key(thread_id))
        self.r.srem(self.active_set_key, thread_id)

    def exists(self, thread_id: str) -> bool:
        """检查会话是否存在"""
        return self.r.exists(self._key(thread_id)) > 0

    def list_active(self) -> List[str]:
        """列出所有活跃的会话 thread_id"""
        return list(self.r.smembers(self.active_set_key))

    def cleanup_inactive(self):
        """
        清理活跃列表中已过期的会话
        （防止 set 里残留 thread_id）
        """
        active_ids = self.r.smembers(self.active_set_key)
        for tid in active_ids:
            if not self.exists(tid):
                self.r.srem(self.active_set_key, tid)

    def _make_key(self, session_id: str, subgraph_id: Optional[str] = None) -> str:
        return f"{subgraph_id}:{session_id}" if subgraph_id else session_id

    def save_graph_state(self, state: Dict[str, Any], session_id: str, subgraph_id: str = None) -> None:
        """
        保存子图状态到 Redis，仅保存最新快照
        """
        try:
            key = self._make_key(session_id, subgraph_id)

            # 确保 messages 始终是 list
            if "messages" in state and not isinstance(state["messages"], list):
                state["messages"] = list(state["messages"])

            serialized_state = json.dumps(state, default=lambda o: o.__dict__)
            self.r.set(key, serialized_state)

            logger.info(f"[StateManager] 状态已保存，key={key}, size={len(serialized_state)} bytes")
        except Exception as e:
            logger.error(f"[StateManager] 保存状态失败: {str(e)}")
            raise

    def load_graph_state(self, session_id: str, subgraph_id: str = None) -> Optional[Dict[str, Any]]:
        """
        从 Redis 加载最新子图状态
        """
        key = self._make_key(session_id, subgraph_id)
        try:
            state_data = self.r.get(key)
            if not state_data:
                logger.info(f"[StateManager] 未找到状态，key={key}")
                return None

            state = json.loads(state_data)

            # 确保 messages 字段可用
            if "messages" not in state or state["messages"] is None:
                state["messages"] = []

            return state
        except Exception as e:
            logger.error(f"[StateManager] 加载状态失败, key={key}, error={str(e)}")
            raise ValueError(f"加载状态失败，key={key},{str(e)}")

    def _deep_dict(self, obj):
        """递归将 defaultdict 转换为 dict，以去除 lambda factory"""
        if isinstance(obj, dict):
            return {k: self._deep_dict(v) for k, v in obj.items()}
        return obj

    def save_checkpoint(self, saver: Any, session_id: str):
        """保存 LangGraph Checkpoint (InMemorySaver state)"""
        try:
            # 必须递归转换，因为 InMemorySaver 内部嵌套了使用 lambda 的 defaultdict
            data = {
                "storage": self._deep_dict(saver.storage),
                "writes": self._deep_dict(saver.writes),
                "blobs": self._deep_dict(saver.blobs)
            }
            import pickle
            self.r.set(f"checkpoint:{session_id}", pickle.dumps(data))
            logger.info(f"Checkpoint 已保存: {session_id}", target=LogTarget.ALL)
        except Exception as e:
            logger.error(f"保存 Checkpoint 失败: {e}", target=LogTarget.ALL)

    def load_checkpoint(self, saver: Any, session_id: str):
        """加载 LangGraph Checkpoint (InMemorySaver state)"""
        try:
            data = self.r.get(f"checkpoint:{session_id}")
            if data:
                import pickle
                state = pickle.loads(data)
                # 恢复状态
                if "storage" in state:
                    saver.storage.update(state["storage"])
                if "writes" in state:
                    saver.writes.update(state["writes"])
                if "blobs" in state:
                    saver.blobs.update(state["blobs"])
                logger.info(f"Checkpoint 已加载: {session_id}", target=LogTarget.ALL)
        except Exception as e:
            logger.error(f"加载 Checkpoint 失败: {e}", target=LogTarget.ALL)


redis_manager = RedisManager()
