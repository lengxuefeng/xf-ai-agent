# # /app/services/tools.py
#
# from langchain_core.tools import tool
# from app.services.retriever import knowledge_retriever  # 假设你已定义好 retriever 实例
#
#
# # --- 定义工具函数 ---
# @tool
# def document_search(query: str) -> str:
#     """
#     使用知识库检索相关文档片段。
#
#     Args:
#         query: 要搜索的查询字符串。
#
#     Returns:
#         包含找到的相关文档片段的字符串（以换行符分隔）。
#     """
#     docs = knowledge_retriever.get_retriever().invoke(query)
#     return "\n\n".join([doc.page_content for doc in docs])
#
#
# # --- 工具列表 ---
# # 所有定义的工具需要放入 agent_tools 列表中以供绑定使用
# agent_tools = [document_search]
