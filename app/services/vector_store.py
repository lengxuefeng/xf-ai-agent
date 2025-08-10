import logging
from pathlib import Path
from typing import Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.config import Settings
from app.utils.decorators import error_handler

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    向量存储服务类，用户管理文档向量存储
    """

    def __init__(self, index_dir: str = "faiss_index"):
        """
        初始化向量存储服务
        Args:
            index_dir:索引目录
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        self.vector_store = None
        self.embeddings = OllamaEmbeddings(model=Settings.EMBEDDING_MODEL, base_url=Settings.EMBEDDING_BASE_URL)
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Settings.CHUNK_SIZE,
            chunk_overlap=Settings.CHUNK_OVERLAP,
            separators=Settings.SEPARATORS
        )

    def upload_embedding_model(self, mode_name: str) -> bool:
        """
        刷新embedding模型
        Args:
            mode_name:模型名称
        Returns:
            bool:刷新成功返回True,否则返回False
        """
        try:
            if self.embeddings.model != mode_name:
                self.embeddings = OllamaEmbeddings(model=Settings.EMBEDDING_MODEL, base_url=Settings.EMBEDDING_BASE_URL)
                logger.info(f"嵌入模型已更新为: {mode_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"刷新嵌入模型失败: {str(e)}")
            return False

    @error_handler()
    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        对文档进行分块处理

        Args:
            documents:原始文档列表
        Returns:
            list:分块后的文档列表
        """
        try:
            # 使用文本分割器进行分块
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"文档分块完成：原始文档数量 {len(documents)}，分块后文档数量 {len(split_docs)}")
            return split_docs
        except Exception as e:
            logger.error(f"文档分割失败: {str(e)}")
            # 分块失败时，返回原始文档列表
            return documents

    @error_handler()
    def create_vector_store(self, documents: list[Document]) -> Optional[FAISS]:
        """
        创建新的向量库实例，会覆盖原有数据

        Args:
            documents: 文档列表
        Returns:
            FAISS:向量存储实例
        """
        if not documents:
            logger.error("没有文档可以创建向量存储")
            return None
        logger.info(f"开始创建向量存储，文档数量: {len(documents)}")

        try:
            # 对文档进行分块
            split_documents = self.split_documents(documents)

            # 使用LangChain的FAISS向量存储
            vector_store = FAISS.from_documents(
                split_documents,
                self.embeddings
            )
            self._save_vector_store(vector_store)
            logger.info(f"向量存储创建成功，包含 {len(split_documents)} 个文档块")
            return self.vector_store

        except Exception as e:
            logger.error(f"向量存储创建失败: {str(e)}")
            return None

    def _save_vector_store(self, vector_store: FAISS):
        """
        保存向量存储实例
        保存向量存储实例到本地文件系统

        Args:
            vector_store: 向量存储实例
        """
        try:
            vector_store.save_local(str(self.index_dir))
            logger.info(f"向量存储已保存到 {self.index_dir}")
        except Exception as e:
            logger.error(f"保存向量存储失败: {str(e)}")

    @error_handler()
    def load_vector_store(self) -> Optional[FAISS]:
        """
        从本地文件系统加载向量存储实例

        Returns:
            FAISS: 向量存储实例
        """
        try:
            # 检查索引文件是否存在
            if (self.index_dir / "index.faiss").exists():
                self.vector_store = FAISS.load_local(
                    str(self.index_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"向量存储已从 {self.index_dir} 加载")
                return self.vector_store
            else:
                logger.warning(f"索引目录 {self.index_dir} 下没有找到向量存储文件")
                return None
        except Exception as e:
            logger.error(f"加载向量存储失败: {str(e)}")
            return None

    @error_handler()
    def search_documents(self, query: str, threshold: float = 0.7) -> list[Document]:
        """
        搜索相关文档

        Args:
            query: 查询字符串
            threshold: 相似度阈值
        Returns:
            list[Document]: 搜索结果文档列表
        """
        if not self.vector_store:
            self.vector_store = self.load_vector_store()
            if not self.vector_store:
                logger.error("向量存储未初始化，无法搜索")
                return []
        try:
            # 执行搜索，使用层方法similarity_search_with_score，直接返回原始分数，适合自定义流程
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query, threshold=threshold
            )
            # 根据阀值过滤结果,只保留文本不要相似度分数
            results = [doc for doc, score in docs_and_scores if score > threshold]
            logger.info(f"搜索到 {len(results)} 个相关文档，相似度阈值: {threshold}")
            return results
        except Exception as e:
            logger.error(f"搜索文档失败: {str(e)}")
            return []

    def get_context(self, docs: list[Document]) -> str:
        """
        从文档列表中提取上下文信息

        Args:
            docs: 文档列表

        Returns:
            str: 合并后的上下文字符串
        """
        if not docs:
            return ""
        return "\n\n".join(doc.page_content for doc in docs)

    def add_document(self, content: str, metadata: dict[str, any] = None) -> bool:
        """
        添加文档到向量存储

        Args:
            content: 文档的实际文本内容（字符串）。
            metadata: 文档的元数据（字典），用于存储额外信息（如来源、页码、时间等）。
        Returns:
            bool: 添加成功返回True,否则返回False
        """
        if not content:
            logger.error("文档内容不能为空，无法添加到向量存储")
            return False

        try:
            # 创建Document 标准化文档格式
            doc = Document(page_content=content, metadata=metadata or {})

            # 对文档进行分块
            split_docs = self.split_documents([doc])

            # 如果向量存储不存在则先初始化
            if not self.vector_store:
                self.vector_store = self.create_vector_store(split_docs)
                if not self.vector_store:
                    # 如果仍不存在则使用当前文档创建新的向量存储
                    self.vector_store = self.create_vector_store([doc])
                    return True

            # 为已存在的向量存储添加文档块
            self.vector_store.add_documents(split_docs)

            # 保存更新后的向量存储
            self._save_vector_store(self.vector_store)

            logger.info(f"成功添加文档，标题: {metadata.get('source', '未知') if metadata else '未知'}，"
                        f"分块数量: {len(split_docs)}")
            return True
        except Exception as e:
            logger.error(f"添加文档到向量存储失败: {str(e)}")
            return False

    def clear_index(self):
        """
        清空向量存储索引（删除所有索引文件）
        """
        try:
            for item in self.index_dir.glob("*"):  # glob("*")匹配目录下的所有文件和子目录
                if item.is_file():
                    item.unlink()  # 删除文件
                elif item.is_dir():
                    item.rmdir()  # 删除空目录（非空目录需递归删除）
            self.vector_store = None
            logger.info("向量存储索引已清空")
        except Exception as e:
            logger.error(f"清空向量存储索引失败: {str(e)}")
            raise


# 单例模式，确保 VectorStore 只有一个实例
vector_store_service = VectorStoreService()
