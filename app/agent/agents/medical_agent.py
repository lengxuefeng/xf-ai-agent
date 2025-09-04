# -*- coding: utf-8 -*-
from typing import Annotated, TypedDict, Generator

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import add_messages

from agent.agent_builder import create_simple_agent_executor
from agent.graph_state import AgentRequest
from agent.llm.loader_llm_multi import load_open_router
from agent.llm.model_config import ModelConfig
from utils import redis_manager

load_dotenv()

"""
医疗问答子图（Medical Agent）。

特点：
1.  不使用任何外部工具，完全依赖大语言模型（LLM）的内部知识。
2.  在回答用户关于健康或医疗的问题后，总会自动附加一条免责声明。
3.  使用 `create_simple_agent_executor` 构建器创建，结构简洁。
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

    def __init__(self, req: AgentRequest):
        """
        初始化医疗 Agent
        """
        if not req.model:
            raise ValueError("医疗模型初始化失败，请检查配置。")

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
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # 定义一个后处理函数，用于在模型输出后附加免责声明
        def add_disclaimer(response: AIMessage) -> AIMessage:
            disclaimer = (
                "\n\n---\n"
                "**免责声明：** 我是一个 AI 助手，以上信息仅供参考，不能作为专业的医疗建议、诊断或治疗方案。"
                "如有任何健康问题，请务必咨询医生或其他有资质的医疗专业人士。"
            )
            response.content += disclaimer
            return response

        self.graph = create_simple_agent_executor(
            model=req.model,
            prompt=prompt,
            state_class=MedicalAgentState,
            post_process_func=add_disclaimer
        )
        self.redis_manager = redis_manager.RedisManager()
        self.session_id = req.session_id
        self.subgraph_id = "medical_agent"

    def run(self, req: AgentRequest) -> Generator:
        """
        以流式方式执行 Agent。
        """
        state = self.redis_manager.load_graph_state(req.session_id, self.subgraph_id)
        if not state:
            state = {"messages": []}

        last_msg = state["messages"][-1] if state["messages"] else None
        if not last_msg or not isinstance(last_msg, HumanMessage):
            state["messages"].append(HumanMessage(content=req.user_input))

        final_state = None
        for event in self.graph.stream(state):
            final_state = event
            yield event

        if final_state:
            self.redis_manager.save_graph_state(final_state, req.session_id, self.subgraph_id)


# if __name__ == '__main__':
#     config = ModelConfig(
#         model="deepseek/deepseek-chat-v3-0324:free",
#         model_key="sk-xxxx",
#         model_url="https://openrouter.ai/api/v1",
#     )
#     agent_llm = load_open_router(config)
#     agent_req = AgentRequest(
#         user_input="我最近总是头疼，是怎么回事？",
#         model=agent_llm,
#         session_id="session_medical_12345",
#         subgraph_id="medical_agent",
#     )
#     medical_agent = MedicalAgent(req=agent_req)
#
#     final_state = None
#     for chunk in medical_agent.run(agent_req):
#         final_state = chunk
#         print("---CHUNK START---")
#         print(final_state)
#         print("---CHUNK END---\\n")
#
#     print("\n\n===== FINAL RESPONSE =====")
#     print(final_state["messages"][-1].content)
