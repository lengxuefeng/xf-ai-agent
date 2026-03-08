from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, START, END, add_messages

from agent.base import BaseAgent
from agent.graph_state import AgentRequest
from utils.custom_logger import get_logger
from agent.prompts.medical_prompt import MedicalPrompt

log = get_logger(__name__)

class MedicalAgentState(TypedDict):
    """
    医疗问答子图状态
    """
    messages: Annotated[List[BaseMessage], add_messages]


class MedicalAgent(BaseAgent):
    """
    医疗问答智能体 (LangGraph 1.0 Refactored)
    """

    def __init__(self, req: AgentRequest):
        super().__init__(req)
        if not req.model:
            raise ValueError("医疗模型初始化失败，请检查配置。")
        self.llm = req.model
        self.subgraph_id = "medical_agent"
        
        # 提示词
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", MedicalPrompt.SYSTEM),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> Runnable:
        workflow = StateGraph(MedicalAgentState)

        def model_node(state: MedicalAgentState):
            chain = self.prompt | self.llm
            response = chain.invoke(state)
            
            # 附加免责声明
            response.content += MedicalPrompt.DISCLAIMER
            return {"messages": [response]}

        workflow.add_node("agent", model_node)
        workflow.add_edge(START, "agent")
        workflow.add_edge("agent", END)

        return workflow.compile(checkpointer=self.checkpointer)
