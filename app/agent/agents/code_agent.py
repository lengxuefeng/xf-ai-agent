from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig

from agent.base import BaseAgent
from agent.graph_state import AgentRequest
from agent.tools.code_tools import execute_python_code
from utils.custom_logger import get_logger
from agent.graphs.checkpointer import checkpointer # 使用全局 Checkpointer

log = get_logger(__name__)

class CodeAgentState(TypedDict):
    """
    代码编写子图状态
    Attributes:
        messages: 对话消息列表
        code_to_execute: 待执行代码
        execution_result: 执行结果
    """
    messages: Annotated[List[BaseMessage], add_messages]
    code_to_execute: Optional[str]
    execution_result: Optional[str]


class CodeAgent(BaseAgent):
    """
    Code Agent (LangGraph 1.0 Refactored)
    """

    def __init__(self, req: AgentRequest):
        super().__init__(req)
        if not req.model:
            raise ValueError("Code Agent 模型初始化失败，请检查配置。")
        self.llm = req.model
        self.checkpointer = checkpointer  # 使用全局单例
        self.subgraph_id = "code_agent"
        
        # 提示词
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个 Python 编程专家。你的任务是根据用户需求编写 Python 代码。"
                    "请只返回可执行的 Python 代码块，不要包含任何 markdown 格式（如 ```python ... ```）。"
                ),
                MessagesPlaceholder(variable_name="messages")
            ]
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(CodeAgentState)

        # 添加节点
        workflow.add_node("generate_code", self._generate_code_node)
        workflow.add_node("execute_code", self._execute_code_node)
        workflow.add_node("analyze_result", self._analyze_result_node)

        # 定义边
        workflow.add_edge(START, "generate_code")
        workflow.add_edge("generate_code", "execute_code")
        workflow.add_edge("execute_code", "analyze_result")
        workflow.add_edge("analyze_result", END)

        # 编译图 (使用全局 checkpointer)
        return workflow.compile(checkpointer=self.checkpointer)

    def _generate_code_node(self, state: CodeAgentState, config: RunnableConfig):
        chain = self.prompt | self.llm
        response = chain.invoke({"messages": state["messages"]})
        code = response.content.strip().replace("```python", "").replace("```", "")
        return {"code_to_execute": code}

    def _execute_code_node(self, state: CodeAgentState, config: RunnableConfig):
        code = state["code_to_execute"]
        
        # --- Native Interrupt Logic ---
        decision = interrupt({
            "action_requests": [{
                "type": "code_approval",
                "name": "execute_code",
                "args": {"code": code},
                "description": f"即将执行 Python 代码:\n{code[:100]}..." # 截断显示
            }],
            "message": "需要审批代码执行"
        })
        
        if decision.get("action") == "reject":
            return {"execution_result": "用户拒绝执行代码。"}
            
        # 执行代码
        try:
            result = execute_python_code(code)
            return {"execution_result": str(result)}
        except Exception as e:
            return {"execution_result": f"代码执行错误: {str(e)}"}

    def _analyze_result_node(self, state: CodeAgentState):
        result = state["execution_result"]
        # 简单包装结果
        return {"messages": [AIMessage(content=f"执行结果: {result}")]}

