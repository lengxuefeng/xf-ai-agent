import hashlib
import time

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

from config.constants.approval_constants import ApprovalDecision, DEFAULT_ALLOWED_DECISIONS, SQL_APPROVAL_ACTION_NAME
from config.constants.sql_agent_constants import (
    SQL_AGENT_APPROVAL_DESC_TEMPLATE,
    SQL_AGENT_APPROVAL_MESSAGE,
    SQL_AGENT_REJECTED_MESSAGE,
)
from supervisor.base import BaseAgent
from supervisor.graph_state import AgentRequest, AgentState
from tools.agent_tools.sql_tools import format_sql_result_for_user, get_schema
from tools.gateway.federated_query_gateway import federated_query_gateway

SQL_SCHEMA_TOOL_NAME = "get_schema"


@tool(SQL_SCHEMA_TOOL_NAME)
def get_schema_tool() -> str:
    """读取数据库 schema；当表名、字段名或 join 关系不明确时先调用。"""
    return get_schema()


@tool(SQL_APPROVAL_ACTION_NAME)
def execute_sql_tool(sql: str) -> str:
    """执行只读 SQL。只有在需要真实查库时才调用，且必须传入完整的只读 SQL。"""
    normalized_sql = str(sql or "").strip()
    if not normalized_sql:
        return "未生成可执行 SQL，请重试。"

    decision = interrupt(
        {
            "action_requests": [
                {
                    "type": "sql_approval",
                    "name": SQL_APPROVAL_ACTION_NAME,
                    "args": {"sql": normalized_sql},
                    "description": SQL_AGENT_APPROVAL_DESC_TEMPLATE.format(sql=normalized_sql),
                    "id": (
                        f"{SQL_APPROVAL_ACTION_NAME}_"
                        f"{hashlib.md5((normalized_sql + str(int(time.time() * 1000))).encode('utf-8')).hexdigest()[:10]}"
                    ),
                }
            ],
            "allowed_decisions": list(DEFAULT_ALLOWED_DECISIONS),
            "message": SQL_AGENT_APPROVAL_MESSAGE,
        }
    )
    action = decision.get("action") if isinstance(decision, dict) else decision
    if action == ApprovalDecision.REJECT.value:
        return SQL_AGENT_REJECTED_MESSAGE

    raw_result = federated_query_gateway.execute_local_sql(normalized_sql)
    return format_sql_result_for_user(normalized_sql, raw_result)


class SqlAgent(BaseAgent):
    REACT_SYSTEM_PROMPT = (
        "你是一个 SQL 专家，必须通过真实工具完成数据库查询。"
        f"可用工具只有 `{SQL_SCHEMA_TOOL_NAME}` 和 `{SQL_APPROVAL_ACTION_NAME}`。"
        "拿不准 schema 时先查 schema；需要真实结果时再执行 SQL；最终只输出中文结论。"
    )

    def __init__(self, req: AgentRequest):
        super().__init__(req)
        if not req.model:
            raise ValueError("SQL 模型初始化失败，请检查配置。")
        self.llm = req.model
        self.subgraph_id = "sql_agent"
        self.system_prompt = self.REACT_SYSTEM_PROMPT
        self.tools = [get_schema_tool, execute_sql_tool]
        self.graph = self._build_graph()

    async def _model_node(self, state: AgentState, config: RunnableConfig):
        messages = list(state.get("messages", []) or [])
        llm_with_tools = self.llm.bind_tools(self.tools)
        response = await llm_with_tools.ainvoke(messages, config=config)
        return {"messages": [response]}

    def _build_graph(self) -> Runnable:
        workflow = StateGraph(AgentState)
        workflow.add_node("model_node", self._model_node, retry_policy=self.RETRY_POLICY)
        workflow.add_node("tools", ToolNode(self.tools), retry_policy=self.RETRY_POLICY)
        workflow.add_edge(START, "model_node")
        workflow.add_conditional_edges("model_node", tools_condition)
        workflow.add_edge("tools", "model_node")
        return workflow.compile(checkpointer=self.checkpointer)
