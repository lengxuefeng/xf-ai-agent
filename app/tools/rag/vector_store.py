# -*- coding: utf-8 -*-
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from common.llm.unified_loader import create_embedding_model_from_config
from common.utils.custom_logger import get_logger
from config.runtime_settings import AGENT_LOOP_CONFIG, is_local_mode

log = get_logger(__name__)

"""
双模式 RAG 检索服务。

cloud:
- 继续使用 PGVector 懒加载链路。

local:
- 基于 workspace_root 扫描 .txt/.md/.pdf
- 使用当前模型配置对应的 Embedding 构建 InMemoryVectorStore
- 通过 workspace_root + 文件指纹缓存进程内索引
"""


class VectorStoreService:
    """向量检索服务，按运行模式自动切换后端。"""

    def __init__(self, collection_name: str = "xf_knowledge_base"):
        self.collection_name = collection_name
        self._lock = threading.RLock()
        self._initialized = False
        self._init_error: str = ""
        self.embeddings = None
        self.vector_store = None
        self._local_index_cache: Dict[str, Dict[str, Any]] = {}

        pg_user = os.getenv("POSTGRES_USER", "postgres")
        pg_pwd = os.getenv("POSTGRES_PASSWORD", "")
        pg_host = os.getenv("POSTGRES_HOST", "127.0.0.1")
        pg_port = os.getenv("POSTGRES_PORT", "5432")
        pg_db = os.getenv("POSTGRES_DB", "xf_ai_agent")
        self.connection_string = f"postgresql+psycopg://{pg_user}:{pg_pwd}@{pg_host}:{pg_port}/{pg_db}"

        self.embedding_model = os.getenv("RAG_EMBEDDING_MODEL", "bge-m3")
        self.embedding_base_url = os.getenv("RAG_EMBEDDING_BASE_URL", "http://127.0.0.1:11434")
        self.default_top_k = int(os.getenv("RAG_DEFAULT_TOP_K", "4"))
        self.max_context_chars = AGENT_LOOP_CONFIG.context_compress_max_chars

    def _ensure_initialized(self) -> bool:
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
                log.warning(f"PGVector 初始化失败，已降级禁用 RAG: {exc}")
                return False

    def add_documents(self, documents: List[Document]) -> List[str]:
        if is_local_mode():
            return []
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
        if is_local_mode():
            return []
        if not self._ensure_initialized() or self.vector_store is None:
            return []
        try:
            return self.vector_store.similarity_search(query, k=k, filter=filter)
        except Exception as exc:
            log.warning(f"相似度检索失败，已降级为空结果: {exc}")
            return []

    def as_retriever(
        self,
        search_kwargs: dict | None = None,
        model_config: Dict[str, Any] | None = None,
    ):
        kwargs = search_kwargs or {"k": self.default_top_k}
        if is_local_mode():
            local_store = self._get_or_build_local_store(model_config=model_config or {})
            if local_store is None:
                return None
            try:
                return local_store.as_retriever(search_kwargs=kwargs)
            except Exception as exc:
                log.warning(f"创建本地 retriever 失败，已降级为空结果: {exc}")
                return None

        if not self._ensure_initialized() or self.vector_store is None:
            return None
        try:
            return self.vector_store.as_retriever(search_kwargs=kwargs)
        except Exception as exc:
            log.warning(f"创建云端 retriever 失败: {exc}")
            return None

    def search_documents(
        self,
        query: str,
        threshold: float = 0.7,
        k: int | None = None,
        model_config: Dict[str, Any] | None = None,
    ) -> List[Document]:
        if is_local_mode():
            return self._search_local_documents(
                query=query,
                threshold=threshold,
                k=k,
                model_config=model_config or {},
            )
        return self._search_cloud_documents(query=query, threshold=threshold, k=k)

    def _search_cloud_documents(self, query: str, threshold: float = 0.7, k: int | None = None) -> List[Document]:
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
            return self.similarity_search(query=query, k=top_k)

    def _search_local_documents(
        self,
        *,
        query: str,
        threshold: float,
        k: int | None,
        model_config: Dict[str, Any],
    ) -> List[Document]:
        local_store = self._get_or_build_local_store(model_config=model_config)
        if local_store is None:
            return []

        top_k = max(1, k or self.default_top_k)
        try:
            pairs: List[Tuple[Document, float]] = local_store.similarity_search_with_relevance_scores(
                query=query,
                k=top_k,
            )
            return [doc for doc, score in pairs if float(score) >= float(threshold)]
        except Exception:
            try:
                return local_store.similarity_search(query=query, k=top_k)
            except Exception as exc:
                log.warning(f"本地 RAG 检索失败，已降级为空结果: {exc}")
                return []

    def _get_or_build_local_store(self, *, model_config: Dict[str, Any]) -> InMemoryVectorStore | None:
        workspace_root = str(model_config.get("workspace_root") or "").strip()
        if not workspace_root:
            return None

        try:
            root_path = Path(workspace_root).expanduser().resolve(strict=True)
        except Exception as exc:
            log.warning(f"本地 RAG 无法解析 workspace_root，已跳过: {exc}")
            return None

        source_files = self._collect_local_source_files(root_path)
        if not source_files:
            return None

        cache_key = self._build_local_cache_key(root_path, model_config)
        fingerprint = self._build_local_fingerprint(root_path, source_files, model_config)

        with self._lock:
            cached = self._local_index_cache.get(cache_key)
            if cached and cached.get("fingerprint") == fingerprint:
                return cached.get("store")

        embeddings = self._build_local_embeddings(model_config)
        if embeddings is None:
            return None

        documents = self._load_local_documents(root_path, source_files)
        if not documents:
            return None

        split_documents = self._split_local_documents(documents)
        if not split_documents:
            return None

        try:
            store = InMemoryVectorStore(embedding=embeddings)
            store.add_documents(split_documents)
        except Exception as exc:
            log.warning(f"本地 RAG 索引构建失败，已跳过: {exc}")
            return None

        with self._lock:
            self._local_index_cache[cache_key] = {
                "fingerprint": fingerprint,
                "store": store,
            }
        return store

    @staticmethod
    def _collect_local_source_files(root_path: Path) -> List[Path]:
        supported_suffixes = {".txt", ".md", ".pdf"}
        return [
            path
            for path in sorted(root_path.rglob("*"))
            if path.is_file() and path.suffix.lower() in supported_suffixes
        ]

    @staticmethod
    def _build_local_cache_key(root_path: Path, model_config: Dict[str, Any]) -> str:
        embedding_signature = "|".join(
            [
                str(model_config.get("service_type") or ""),
                str(model_config.get("model_service") or ""),
                str(model_config.get("embedding_model") or ""),
                str(model_config.get("model_url") or ""),
            ]
        )
        return f"{root_path}:{embedding_signature}"

    @staticmethod
    def _build_local_fingerprint(
        root_path: Path,
        source_files: List[Path],
        model_config: Dict[str, Any],
    ) -> str:
        payload = {
            "workspace_root": str(root_path),
            "embedding_model": str(model_config.get("embedding_model") or ""),
            "service_type": str(model_config.get("service_type") or ""),
            "model_service": str(model_config.get("model_service") or ""),
            "files": [
                {
                    "path": str(path.relative_to(root_path)),
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
                for path in source_files
            ],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _build_local_embeddings(self, model_config: Dict[str, Any]):
        try:
            return create_embedding_model_from_config(
                model=str(model_config.get("model") or ""),
                model_service=str(model_config.get("model_service") or ""),
                service_type=str(model_config.get("service_type") or ""),
                model_size=str(model_config.get("model_size") or "large"),
                deep_thinking_mode=str(model_config.get("deep_thinking_mode") or "auto"),
                rag_enabled=True,
                similarity_threshold=float(model_config.get("similarity_threshold", 0.7)),
                embedding_model=str(model_config.get("embedding_model") or self.embedding_model),
                model_key=str(model_config.get("model_key") or ""),
                model_url=str(model_config.get("model_url") or ""),
                embedding_model_key=str(model_config.get("embedding_model_key") or ""),
            )
        except Exception as exc:
            log.warning(f"本地 RAG Embedding 初始化失败，已降级跳过: {exc}")
            return None

    def _load_local_documents(self, root_path: Path, source_files: List[Path]) -> List[Document]:
        documents: list[Document] = []
        for file_path in source_files:
            try:
                file_docs = self._load_single_local_document(file_path)
            except Exception as exc:
                log.warning(f"本地 RAG 文档读取失败，已跳过: path={file_path}, error={exc}")
                continue
            for doc in file_docs:
                metadata = dict(getattr(doc, "metadata", {}) or {})
                metadata["source"] = str(file_path.relative_to(root_path))
                doc.metadata = metadata
            documents.extend(file_docs)
        return documents

    @staticmethod
    def _load_single_local_document(file_path: Path) -> List[Document]:
        suffix = file_path.suffix.lower()
        if suffix in {".txt", ".md"}:
            try:
                from langchain_community.document_loaders import TextLoader
            except Exception:
                from langchain.document_loaders import TextLoader  # type: ignore
            return TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True).load()

        try:
            from langchain_community.document_loaders import PyPDFLoader
        except Exception:
            from langchain.document_loaders import PyPDFLoader  # type: ignore
        return PyPDFLoader(str(file_path)).load()

    @staticmethod
    def _split_local_documents(documents: List[Document]) -> List[Document]:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except Exception:
            return documents

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=160,
        )
        return splitter.split_documents(documents)

    def get_context(self, docs: List[Document]) -> str:
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


vector_store_service = VectorStoreService()
