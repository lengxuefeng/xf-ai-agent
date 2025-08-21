# # /app/agent.py
#
# import operator
# from typing import Annotated, List, TypedDict
#
# from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
# from langchain_google_genai import ChatGoogleGenerativeAI
#
# from langgraph.graph import END, StateGraph
# from langgraph.prebuilt import ToolNode, tools_condition
#
# from app.services.tools import agent_tools
# from app.utils.config import settings
#
#
# # --- 1. 定义代理的状态 ---
# # 使用 TypedDict 结构定义状态，存储所有消息历史（人类消息、AI 消息、工具消息等）
# # Annotated[List[BaseMessage], operator.add] 表示 LangGraph 会自动累加消息列表
# class AgentState(TypedDict):
#     """
#     LangGraph 代理的状态。
#
#     Attributes:
#         messages: 消息列表，代表了对话的完整历史，包括人类、AI和工具消息。
#     """
#     messages: Annotated[List[BaseMessage], operator.add]
#
#
# # --- 2. 定义 LangGraph Agent 类 ---
# class LangGraphAgent:
#     """
#     基于 LangGraph 的智能代理，能够使用工具进行交互。
#     """
#
#     def __init__(self):
#         """
#         初始化代理：设置 LLM、工具节点和编译图。
#         """
#         self.llm = None
#         if not settings.GOOGLE_API_KEY:
#             print("⚠️ 警告: 未设置 GOOGLE_API_KEY，Gemini 模型将不会被初始化。")
#         else:
#             try:
#                 # 初始化 Gemini-Pro 模型，并绑定工具
#                 self.llm = ChatGoogleGenerativeAI(
#                     model="gemini-pro", temperature=0, google_api_key=settings.GOOGLE_API_KEY
#                 ).bind_tools(agent_tools)
#             except Exception as e:
#                 print(f"⚠️ 警告: 无法初始化 Gemini-Pro 模型: {e}")
#                 self.llm = None
#
#         self.agent_runnable = self._build_agent_graph()
#
#     def _build_agent_graph(self):
#         """
#         构建 LangGraph 代理流程图，包含模型调用和工具调用节点。
#         """
#         if not self.llm:
#             def fallback_agent(state: AgentState):
#                 return {"messages": [AIMessage(content="AI 模型未初始化，无法提供服务。")]}
#             return fallback_agent
#
#         # 创建流程图构建器
#         graph = StateGraph(AgentState)
#
#         # 添加 LLM 节点
#         graph.add_node("llm", self.call_model)
#
#         # 添加工具节点，需要将工具列表传递给 ToolNode
#         graph.add_node("tools", ToolNode(tools=agent_tools))
#
#         # 设置起始节点为 llm
#         graph.set_entry_point("llm")
#
#         # 条件边判断是否要调用工具（新版使用 tools_condition 自动判断）
#         graph.add_conditional_edges(
#             "llm",
#             tools_condition,
#             {
#                 "tools": "tools",  # 若需调用工具则跳转至 tools 节点
#                 END: END,          # 否则结束
#             }
#         )
#
#         # 工具执行完后，继续回到 llm
#         graph.add_edge("tools", "llm")
#
#         return graph.compile()
#
#     def call_model(self, state: AgentState) -> dict:
#         """
#         LLM 调用节点：调用模型生成回复。
#         """
#         messages = state["messages"]
#         response = self.llm.invoke(messages)
#         return {"messages": [response]}
#
#     def invoke_agent(self, message: str) -> str:
#         """
#         执行整个 LangGraph 流程，从用户输入到最终回复。
#         """
#         if not self.agent_runnable:
#             return "对不起，代理当前不可用。"
#
#         initial_state = {"messages": [HumanMessage(content=message)]}
#         final_state = self.agent_runnable.invoke(initial_state)
#         final_answer = final_state["messages"][-1]
#         if isinstance(final_answer, AIMessage):
#             return final_answer.content
#
#         return "对不起，我无法处理您的问题。"
#
#
# # --- 3. 创建单例代理实例 ---
# langgraph_agent = LangGraphAgent()
