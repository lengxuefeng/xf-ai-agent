from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.agents import AgentFinish
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph import StateGraph, END
from app.utils.config import settings
from app.services.tools import agent_tools

# 定义代理的状态
class AgentState(TypedDict):
    """
    LangGraph 代理的状态。

    Attributes:
        messages: 消息列表，包括人类消息和 AI 消息。
        tool_calls: 工具调用列表。
        tool_result: 工具执行结果。
    """
    messages: Annotated[List[BaseMessage], operator.add]
    tool_calls: List[ToolInvocation]
    tool_result: str

class LangGraphAgent:
    """
    基于 LangGraph 的智能代理。
    """
    def __init__(self):
        if not settings.GOOGLE_API_KEY:
            print("警告: 未设置 GOOGLE_API_KEY，ChatGoogleGenerativeAI 模型将不会被初始化。")
            self.llm = None
        else:
            try:
                self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, google_api_key=settings.GOOGLE_API_KEY)
            except Exception as e:
                print(f"警告: 无法初始化 ChatGoogleGenerativeAI 模型，AI 聊天功能可能受限: {e}")
                self.llm = None
        self.tools = agent_tools
        self.tool_executor = ToolExecutor(self.tools)
        self.agent_runnable = self._build_agent_graph()

    def _build_agent_graph(self):
        """
        构建 LangGraph 代理图。
        """
        if not self.llm:
            return "AI 模型未初始化，无法提供服务。"
        # 将工具绑定到 LLM
        llm_with_tools = self.llm.bind_tools(self.tools)

        # 定义图
        graph = StateGraph(AgentState)

        # 定义节点
        def call_model(state: AgentState):
            messages = state["messages"]
            if not self.llm:
                return {"messages": [AIMessage(content="AI 模型未初始化，无法生成响应。")]}
            response = self.llm.invoke(messages)
            return {"messages": [response]}

        def call_tool(state: AgentState):
            tool_calls = state["messages"][-1].tool_calls
            if not tool_calls:
                raise ValueError("没有工具调用被检测到。")
            
            # 假设只有一个工具调用
            tool_invocation = tool_calls[0]
            action = ToolInvocation(tool=tool_invocation.name, tool_input=tool_invocation.args)
            response = self.tool_executor.invoke(action)
            return {"tool_result": str(response), "messages": [AIMessage(content=f"工具执行结果: {response}")]}

        # 定义边
        graph.add_node("llm", call_model)
        graph.add_node("action", call_tool)

        graph.set_entry_point("llm")

        def should_continue(state: AgentState):
            last_message = state["messages"][-1]
            if last_message.tool_calls:
                return "action"
            else:
                return END

        graph.add_conditional_edges(
            "llm",
            should_continue,
            {
                "action": "action",
                END: END
            },
        )
        graph.add_edge("action", "llm")

        return graph.compile()

    def invoke_agent(self, message: str) -> str:
        """
        调用代理并获取响应。
        """
        # 初始状态，包含用户消息
        initial_state = {"messages": [HumanMessage(content=message)]}
        
        # 调用代理
        # LangGraph 的 invoke 方法返回的是一个生成器，需要迭代获取最终结果
        # 这里我们只取最后一个状态的消息作为最终响应
        final_state = None
        for s in self.agent_runnable.stream(initial_state):
            final_state = s

        if final_state and "messages" in final_state:
            # 查找最后一个 AIMessage
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, AIMessage):
                    return msg.content
            # 如果没有 AIMessage，返回最后一个 HumanMessage 的内容（作为回显）
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, HumanMessage):
                    return msg.content
        return "对不起，我无法理解您的问题。"

# 单例模式，确保 LangGraphAgent 只有一个实例
langgraph_agent = LangGraphAgent()
