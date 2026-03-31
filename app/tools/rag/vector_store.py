# -*- coding: utf-8 -*-
# app/agent/rag/vector_store.py
import os
import threading
from typing import List, Tuple

from langchain_core.documents import Document

from config.runtime_settings import AGENT_LOOP_CONFIG
from common.utils.custom_logger import get_logger

log = get_logger(__name__)

"""
PGVector 向量库服务（懒加载版）。

设计目标：
1. 避免在模块 import 阶段就连接数据库，防止 RAG 未开启也拖垮主链路。
2. RAG 依赖不可用时自动降级为空结果，不影响聊天主流程。
3. 兼容历史调用：保留 search_documents / get_context 接口。
"""


class VectorStoreService:
    """向量检索服务，按需初始化 PGVector 和 Embedding 模型。"""

    def __init__(self, collection_name: str = "xf_knowledge_base"):
        """初始化服务基础配置，不在构造函数里做远程连接。"""
        self.collection_name = collection_name
        self._lock = threading.RLock()
        self._initialized = False
        self._init_error: str = ""
        self.embeddings = None
        self.vector_store = None

        # 连接参数配置化，默认值使用本地开发友好的保守值。
        pg_user = os.getenv("POSTGRES_USER", "postgres")
        pg_pwd = os.getenv("POSTGRES_PASSWORD", "")
        pg_host = os.getenv("POSTGRES_HOST", "127.0.0.1")
        pg_port = os.getenv("POSTGRES_PORT", "5432")
        pg_db = os.getenv("POSTGRES_DB", "xf_ai_agent")
        self.connection_string = f"postgresql+psycopg://{pg_user}:{pg_pwd}@{pg_host}:{pg_port}/{pg_db}"

        # Embedding 配置统一读取环境变量，避免硬编码。
        self.embedding_model = os.getenv("RAG_EMBEDDING_MODEL", "bge-m3")
        self.embedding_base_url = os.getenv("RAG_EMBEDDING_BASE_URL", "http://127.0.0.1:11434")

        # 默认检索参数
        self.default_top_k = int(os.getenv("RAG_DEFAULT_TOP_K", "4"))
        self.max_context_chars = AGENT_LOOP_CONFIG.context_compress_max_chars

    def _ensure_initialized(self) -> bool:
        """
        确保向量库已初始化。

        返回：
        - True: 可正常检索
        - False: 初始化失败，调用方应降级为空结果
        """
        if self._initialized and self.vector_store is not None:
            return True
        if self._init_error:
            return False

        with self._lock:
            if self._initialized and self.vector_store is not None:
                return True
            if self._init_error:
                return False
            try:
                # 优先使用新版包，避免 LangChain deprecation 告警
                try:
                    from langchain_ollama import OllamaEmbeddings  # type: ignore
                except Exception:
                    from langchain_community.embeddings import OllamaEmbeddings  # type: ignore
                from langchain_postgres import PGVector

                self.embeddings = OllamaEmbeddings(
                    model=self.embedding_model,
                    base_url=self.embedding_base_url,
                )
                self.vector_store = PGVector(
                    embeddings=self.embeddings,
                    collection_name=self.collection_name,
                    connection=self.connection_string,
                    use_jsonb=True,
                )
                self._initialized = True
                log.info(f"PGVector 初始化成功，集合: {self.collection_name}")
                return True
            except Exception as exc:
                self._init_error = str(exc)
                # RAG 初始化失败不抛异常，避免影响主聊天链路
                log.warning(f"PGVector 初始化失败，已降级禁用 RAG: {exc}")
                return False

    def add_documents(self, documents: List[Document]) -> List[str]:
        """将文档切片写入向量库；不可用时返回空列表。"""
        if not self._ensure_initialized() or self.vector_store is None:
            return []
        try:
            ids = self.vector_store.add_documents(documents)
            log.info(f"成功写入 {len(ids)} 条向量文档")
            return ids
        except Exception as exc:
            log.error(f"向量写入失败: {exc}")
            return []

    def similarity_search(self, query: str, k: int = 4, filter: dict = None) -> List[Document]:
        """相似度检索；不可用时返回空列表。"""
        if not self._ensure_initialized() or self.vector_store is None:
            return []
        try:
            return self.vector_store.similarity_search(query, k=k, filter=filter)
        except Exception as exc:
            log.warning(f"相似度检索失败，已降级为空结果: {exc}")
            return []

    def as_retriever(self, search_kwargs: dict = None):
        """转化为 LangChain 标准检索器；不可用时返回 None。"""
        if not self._ensure_initialized() or self.vector_store is None:
            return None
        kwargs = search_kwargs or {"k": self.default_top_k}
        try:
            return self.vector_store.as_retriever(search_kwargs=kwargs)
        except Exception as exc:
            log.warning(f"创建 retriever 失败: {exc}")
            return None

    def search_documents(self, query: str, threshold: float = 0.7, k: int | None = None) -> List[Document]:
        """
        兼容 GraphRunner 的检索接口。

        优先使用带相关度分数的检索；若底层不支持则回退普通相似检索。
        """
        if not self._ensure_initialized() or self.vector_store is None:
            return []

        top_k = max(1, k or self.default_top_k)
        try:
            pairs: List[Tuple[Document, float]] = self.vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=top_k,
            )
            result_docs: List[Document] = []
            for doc, score in pairs:
                if float(score) >= float(threshold):
                    result_docs.append(doc)
            return result_docs
        except Exception:
            # 某些向量后端不支持 relevance_scores，回退普通检索
            return self.similarity_search(query=query, k=top_k)

    def get_context(self, docs: List[Document]) -> str:
        """将检索文档拼装为注入模型的上下文字符串。"""
        if not docs:
            return ""
        parts: List[str] = []
        total_len = 0
        for idx, doc in enumerate(docs, start=1):
            text = (getattr(doc, "page_content", "") or "").strip()
            if not text:
                continue
            source = ""
            metadata = getattr(doc, "metadata", {}) or {}
            if isinstance(metadata, dict):
                source = str(metadata.get("source") or "").strip()
            block = f"[文档{idx}]"
            if source:
                block += f" 来源: {source}"
            block += f"\n{text}"
            if total_len + len(block) > self.max_context_chars:
                remain = max(0, self.max_context_chars - total_len)
                if remain > 0:
                    parts.append(block[:remain])
                break
            parts.append(block)
            total_len += len(block)
        return "\n\n".join(parts)


# 全局服务单例（懒加载，不会在 import 时连接数据库）
vector_store_service = VectorStoreService()
