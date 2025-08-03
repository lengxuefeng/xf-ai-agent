import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from app.utils.config import settings

class KnowledgeRetriever:
    """
    知识检索器，负责加载知识库文档，创建向量存储，并提供检索功能。
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KnowledgeRetriever, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.vectorstore_path = settings.VECTORSTORE_DIR
        self.knowledge_data_path = settings.KNOWLEDGE_DATA_DIR
        if not settings.GOOGLE_API_KEY:
            print("警告: 未设置 GOOGLE_API_KEY，GoogleGenerativeAIEmbeddings 将不会被初始化。")
            self.embeddings = None
        else:
            try:
                self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=settings.GOOGLE_API_KEY)
            except Exception as e:
                print(f"警告: 无法初始化 GoogleGenerativeAIEmbeddings，RAG 功能可能受限: {e}")
                self.embeddings = None
        self.retriever = self._load_or_create_vectorstore()

    def _load_or_create_vectorstore(self) -> VectorStoreRetriever:
        """
        加载或创建 FAISS 向量存储。
        如果向量存储已存在，则直接加载；否则，从知识库文件创建。
        """
        if os.path.exists(self.vectorstore_path) and os.listdir(self.vectorstore_path):
            print("加载现有向量存储...")
            if self.embeddings:
                vectorstore = FAISS.load_local(self.vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)
            else:
                print("警告: 嵌入模型未初始化，无法加载向量存储。")
                return None
        else:
            print("创建新的向量存储...")
            if self.embeddings:
                docs = self._load_documents()
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                texts = text_splitter.split_documents(docs)
                vectorstore = FAISS.from_documents(texts, self.embeddings)
                vectorstore.save_local(self.vectorstore_path)
            else:
                print("警告: 嵌入模型未初始化，无法创建向量存储。")
                return None
        return vectorstore.as_retriever()

    def _load_documents(self):
        """
        从知识库数据目录加载所有文本文件。
        """
        documents = []
        for filename in os.listdir(self.knowledge_data_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.knowledge_data_path, filename)
                loader = TextLoader(file_path)
                documents.extend(loader.load())
        return documents

    def get_retriever(self) -> VectorStoreRetriever:
        """
        获取配置好的向量存储检索器。
        """
        return self.retriever

# 单例模式，确保 KnowledgeRetriever 只有一个实例
knowledge_retriever = KnowledgeRetriever()
