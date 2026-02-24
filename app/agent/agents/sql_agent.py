from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.base import BaseAgent
from agent.graph_state import AgentRequest
from agent.tools.sql_tools import get_schema, execute_sql
from utils.custom_logger import get_logger, LogTarget
from agent.graphs.checkpointer import checkpointer # 使用全局 Checkpointer

log = get_logger(__name__)

class SqlAgentState(TypedDict):
    """
    定义 SQL 子图的状态。
    """
    messages: Annotated[List[BaseMessage], add_messages]
    sql_to_execute: str


class SqlAgent(BaseAgent):
    """
    SQL Agent (LangGraph 1.0 Refactored)
    
    工作流程:
    1. 获取 Schema
    2. 生成 SQL
    3. 人工审核 (使用 native interrupt)
    4. 执行 SQL
    5. 生成自然语言回复
    """

    def __init__(self, req: AgentRequest):
        super().__init__(req)
        if not req.model:
            raise ValueError("SQL 模型初始化失败，请检查配置。")
        self.llm = req.model
        self.checkpointer = checkpointer  # 使用全局单例
        self.subgraph_id = "sql_agent"
        
        # 提示词模板
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个 SQL 专家。你的任务是根据用户的自然语言问题和数据库的表结构，生成一句准确的 SQL 查询语句。"
                    "请只返回 SQL 语句，不要包含任何其他解释或格式（如 ```sql ... ```）。"
                    "数据库表结构如下：\n{schema}"
                ),
                MessagesPlaceholder(variable_name="messages")
            ]
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(SqlAgentState)

        # 添加节点
        workflow.add_node("get_schema", self._get_schema_node)
        workflow.add_node("generate_sql", self._generate_sql_node)
        workflow.add_node("execute_sql", self._execute_sql_node)
        workflow.add_node("generate_response", self._generate_response_node)

        # 定义边
        workflow.add_edge(START, "get_schema")
        workflow.add_edge("get_schema", "generate_sql")
        workflow.add_edge("generate_sql", "execute_sql")
        workflow.add_edge("execute_sql", "generate_response")
        workflow.add_edge("generate_response", END)

        # 编译图 (使用全局 checkpointer)
        return workflow.compile(checkpointer=self.checkpointer)

    def _get_schema_node(self, state: SqlAgentState):
        schema_info = get_schema()
        # 将 schema 信息作为 ToolMessage 添加到历史记录（模拟工具调用结果）
        # 或者直接作为 SystemMessage 的一部分（这里为了简单，作为上下文消息）
        return {"messages": [AIMessage(content=f"数据库表结构上下文已加载。")]}

    def _generate_sql_node(self, state: SqlAgentState):
        schema_info = get_schema() # 重新获取或从 state 获取
        # 构建 prompt
        chain = self.prompt.partial(schema=schema_info) | self.llm
        response = chain.invoke({"messages": state["messages"]})
        sql = response.content.strip().replace("```sql", "").replace("```", "")
        return {"sql_to_execute": sql}

    def _execute_sql_node(self, state: SqlAgentState):
        sql = state["sql_to_execute"]
        
        # --- Native Interrupt Logic ---
        # 触发中断，等待用户批准
        # interrupt() 会暂停执行，并返回 resume 时传入的值
        decision = interrupt({
            "action_requests": [{
                "type": "sql_approval",
                "name": "execute_sql",
                "args": {"sql": sql},
                "description": f"即将执行 SQL: {sql}，请确认是否安全。"
            }],
            "message": "需要审批 SQL 执行"
        })
        
        # Resume 后继续执行
        if decision.get("action") == "reject":
            return {"messages": [AIMessage(content="用户拒绝了 SQL 执行。")]}
            
        # 执行 SQL
        try:
            result = execute_sql(sql)
            result_message = ToolMessage(content=f"SQL 执行结果：\n{result}", tool_call_id="execute_sql", name="execute_sql")
            return {"messages": [result_message]}
        except Exception as e:
            return {"messages": [AIMessage(content=f"SQL 执行出错: {str(e)}")]}

    def _generate_response_node(self, state: SqlAgentState):
        # 检查上一条消息是否是执行结果
        last_msg = state["messages"][-1]
        if isinstance(last_msg, ToolMessage):
            # 生成最终回答
            response_prompt = (
                "你是一个数据分析师。请根据用户的原始问题和以下的 SQL 执行结果，生成一句通顺的、人类可读的回答。\n"
                "SQL 执行结果:\n{result}"
            )
            # 这里简单处理，实际应该用 LLM
            # 为演示，直接让 LLM 总结
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", response_prompt),
                ("human", "请总结以上结果。")
            ])
            chain = summary_prompt | self.llm
            response = chain.invoke({"result": last_msg.content})
            return {"messages": [response]}
        else:
            # 如果是拒绝执行，直接返回
            return {"messages": []}

    # 移除 run 方法，使用 BaseAgent 的通用 run (稍后重构)
