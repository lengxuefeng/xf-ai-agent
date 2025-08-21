import io
from tkinter import END
from typing import Annotated, TypedDict

from IPython.core.display_functions import display
from IPython.display import Image as IPyImage  # 重命名为 IPyImage
from PIL import Image as PILImage  # 重命名为 PILImage，避免冲突
from dotenv import load_dotenv
from langchain.agents import Agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import add_messages, StateGraph

from agent.agents.code_agent import CodeAgentState
from agent.graph_state import AgentRequest
from agent.llm.loader_llm_multi import load_open_router
from agent.llm.ollama_model import load_ollama_model
from utils import concurrent_executor
from utils import redis_manager

load_dotenv()

"""
医疗问答子图（Medical Agent）。

特典：
1.  不使用任何外部工具，完全依赖大语言模型（LLM）的内部知识。
2.  在回答用户关于健康或医疗的问题后，总会自动附加一条免责声明。
    这是为了强调 AI 的回答不能替代专业的医疗建议，确保使用的安全性。
"""


class MedicalAgentState(TypedDict):
    """
    医疗问答子图状态
    """
    messages: Annotated[list, add_messages]


class MedicalAgent:
    """
    医疗问答智能体
    """

    def __init__(self, req: AgentRequest, max_threads: int = 2):
        if not req.model:
            raise ValueError("医疗模型初始化失败，请检查配置。")
        self.llm = req.model
        self.redis_manager = redis_manager.RedisManager()
        self.session_id = req.session_id
        self.subgraph_id = req.subgraph_id
        self.concurrent_executor = concurrent_executor.ConcurrentExecutor(max_threads=max_threads)
        # 定义问答提示词模版
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    你是一个医疗健康问答助手。请根据用户的提问，利用你的知识库提供相关信息。
                    你的回答应该清晰、准确，并使用通俗易懂的语言。
                    请注意，你提供的所有信息都不能替代专业的医疗诊断和建议。
                    """
                ),
                # 消息占位符，将在这里插入整理对话历史，在对话链中传递历史消息，动态注入上下文，实现多轮对话记忆功能
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        # 定义问答链
        self.chain = prompt | self.llm
        self.graph = self._build_graph()

    # 定义图的节点
    def _agent_node(self, state: MedicalAgentState):
        """
        定义智能体节点（Agent Node）。

        Args:
            state (MedicalAgentState): 当前图的状态。

        Returns:
            dict: 包含模型生成的新消息（已附带免责声明）的字典。
        """
        response = self.chain.invoke({"messages": state["messages"]})
        # 附加免责声明
        disclaimer = (
            "\n\n---\n"
            "**免责声明：** 我是一个 AI 助手，以上信息仅供参考，不能作为专业的医疗建议、诊断或治疗方案。"
            "如有任何健康问题，请务必咨询医生或其他有资质的医疗专业人士。"
        )
        response.content += disclaimer
        return {"messages": [response]}

    def _build_graph(self):
        """
        构建医疗问答子图
        """
        graph = StateGraph(MedicalAgentState)
        graph.add_node("agent", self._agent_node)
        graph.set_entry_point("agent")
        graph.add_edge("agent", "__end__")
        return graph.compile()

    def run(self, req: AgentRequest) -> CodeAgentState:
        """
        运行医疗问答子图
        """
        state = self.redis_manager.load_graph_state(req.session_id, subgraph_id=self.subgraph_id)
        if not state:
            state = {"messages": []}

        last_msg = state["messages"][-1] if state["messages"] else None
        if not last_msg or getattr(last_msg, "content", None) != req.user_input:
            state["messages"].append(HumanMessage(content=req.user_input))

        result = self.graph.invoke(state)

        self.redis_manager.save_graph_state(result, req.session_id, req.subgraph_id)
        return result


if __name__ == '__main__':
    llm = load_open_router("deepseek/deepseek-chat-v3-0324:free")
    req = AgentRequest(
        user_input="肚子疼怎么办",
        model=llm,
        session_id="123232",
        subgraph_id="medical_agent",
    )
    medical_agent = MedicalAgent(req)
    result = medical_agent.run(req)
    # print(json.dumps(result, indent=2))
    print(result["messages"][-1].content)
