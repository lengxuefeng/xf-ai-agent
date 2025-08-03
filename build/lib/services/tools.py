from langchain.tools import tool
from app.services.retriever import knowledge_retriever

class AgentTools:
    """
    为 LangGraph Agent 定义的工具集。
    """

    @tool
    def document_search(query: str) -> str:
        """
        使用知识库检索相关文档片段。
        输入是一个字符串，表示要搜索的查询。
        返回是找到的相关文档片段的字符串。
        """
        docs = knowledge_retriever.get_retriever().invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

# 将工具实例化，以便在 Agent 中使用
agent_tools = [AgentTools().document_search]
