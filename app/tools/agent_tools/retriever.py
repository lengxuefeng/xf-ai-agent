# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from tools.rag.vector_store import vector_store_service


class KnowledgeRetriever:
    """统一的双模式知识检索入口。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def search_documents(
        self,
        query: str,
        *,
        threshold: float = 0.7,
        k: Optional[int] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        return vector_store_service.search_documents(
            query=query,
            threshold=threshold,
            k=k,
            model_config=model_config or {},
        )

    def get_context(
        self,
        query: str,
        *,
        threshold: float = 0.7,
        k: Optional[int] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        documents = self.search_documents(
            query=query,
            threshold=threshold,
            k=k,
            model_config=model_config,
        )
        return vector_store_service.get_context(documents)

    def as_retriever(
        self,
        *,
        k: Optional[int] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        search_kwargs = {"k": max(1, int(k or vector_store_service.default_top_k))}
        return vector_store_service.as_retriever(
            search_kwargs=search_kwargs,
            model_config=model_config or {},
        )


knowledge_retriever = KnowledgeRetriever()


__all__ = ["KnowledgeRetriever", "knowledge_retriever"]
