# -*- coding: utf-8 -*-
# app/agent/rag/vector_store.py
import os
from typing import List

from langchain_core.documents import Document
from langchain_postgres import PGVector

from utils.custom_logger import get_logger

log = get_logger(__name__)

"""
【学习笔记】PGVector 向量库服务单例
完美替代旧版 FAISS，支持高并发和 JSONB 混合过滤。
"""


class VectorStoreService:
    def __init__(self, collection_name: str = "xf_knowledge_base"):
        self.collection_name = collection_name

        pg_user = os.getenv("POSTGRES_USER", "postgres")
        pg_pwd = os.getenv("POSTGRES_PASSWORD", "xiaoleng")
        pg_host = os.getenv("POSTGRES_HOST", "192.168.1.10")
        pg_port = os.getenv("POSTGRES_PORT", "5432")
        pg_db = os.getenv("POSTGRES_DB", "xf_ai_agent")

        self.connection_string = f"postgresql+psycopg://{pg_user}:{pg_pwd}@{pg_host}:{pg_port}/{pg_db}"

        # =========================================================
        # ⚠️ 【关键】请在这里初始化你之前使用的 Embedding 模型！
        # 例如：如果你之前用的是智谱，或者是 OpenAI，或者是本地的 BGE 模型
        # =========================================================
        # 示例 (你需要根据你原本的引入来修改这几行):
        # from langchain_openai import OpenAIEmbeddings
        # self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # 暂时的兜底占位 (如果是本地跑的，可能是下面这种):
        # from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        # self.embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")
        from langchain_community.embeddings import OllamaEmbeddings

        self.embeddings = OllamaEmbeddings(
            model="bge-m3",  # 换成最强的 bge-m3
            base_url="http://127.0.0.1:11434"  # 你的 Ollama 本地服务地址
        )
        # =========================================================
        # =========================================================

        # 初始化 PGVector
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            connection=self.connection_string,
            use_jsonb=True,  # 开启 JSONB 存储 metadata，支持极速过滤检索！
        )
        log.info(f"PGVector 初始化成功！知识库集合: {self.collection_name}")

    def add_documents(self, documents: List[Document]) -> List[str]:
        """将切片后的文档灌入数据库"""
        try:
            ids = self.vector_store.add_documents(documents)
            log.info(f"成功将 {len(ids)} 个文档向量化并存入 PgSQL")
            return ids
        except Exception as e:
            log.error(f"灌库失败: {e}")
            raise

    def similarity_search(self, query: str, k: int = 4, filter: dict = None) -> List[Document]:
        """相似度检索"""
        return self.vector_store.similarity_search(query, k=k, filter=filter)

    def as_retriever(self, search_kwargs: dict = None):
        """转化为 LangChain 标准检索器"""
        kwargs = search_kwargs or {"k": 4}
        return self.vector_store.as_retriever(search_kwargs=kwargs)


vector_store_service = VectorStoreService()
