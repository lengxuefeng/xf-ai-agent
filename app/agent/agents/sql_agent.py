import re
from typing import TypedDict, Annotated, List, Dict, Set
import hashlib
import time
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.base import BaseAgent
from agent.gateway.federated_query_gateway import federated_query_gateway
from agent.graph_state import AgentRequest
from agent.tools.sql_tools import get_schema, format_sql_result_for_user
from utils.custom_logger import get_logger, LogTarget
from agent.graphs.checkpointer import checkpointer # 使用全局 Checkpointer
from agent.prompts.sql_prompt import SqlPrompt

log = get_logger(__name__)

class SqlAgentState(TypedDict):
    """
    定义 SQL 子图的状态。
    """
    messages: Annotated[List[BaseMessage], add_messages]
    sql_to_execute: str
    sql_result: str


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
                ("system", SqlPrompt.GENERATE_SQL),
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

    @staticmethod
    def _latest_human_text(messages: List[BaseMessage]) -> str:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return (msg.content or "").strip()
        return ""

    @staticmethod
    def _human_texts(messages: List[BaseMessage]) -> List[str]:
        return [(m.content or "").strip() for m in messages if isinstance(m, HumanMessage)]

    @staticmethod
    def _extract_limit(text: str, default: int = 10) -> int:
        t = (text or "").lower()
        m = re.search(r"前\s*(\d+)\s*条", t)
        if m:
            return int(m.group(1))
        m = re.search(r"limit\s+(\d+)", t)
        if m:
            return int(m.group(1))
        return default

    @staticmethod
    def _parse_schema_tables(schema_info: str) -> Dict[str, Set[str]]:
        """
        从 get_schema 的文本输出中解析:
        {
          table_name: {col1, col2, ...}
        }
        """
        tables: Dict[str, Set[str]] = {}
        current_table = ""
        for raw in (schema_info or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("表名:"):
                current_table = line.split(":", 1)[1].strip()
                if current_table:
                    tables.setdefault(current_table, set())
                continue
            if current_table and "|" in line and not line.startswith("---") and not line.startswith("字段名"):
                col = line.split("|", 1)[0].strip()
                if col:
                    tables[current_table].add(col)
        return tables

    @staticmethod
    def _extract_sql_tables(sql: str) -> Set[str]:
        t = (sql or "").strip().lower()
        pattern = r"\b(?:from|join|update|into)\s+([a-zA-Z_][a-zA-Z0-9_\.]*)"
        found = set()
        for table in re.findall(pattern, t):
            found.add(table.split(".")[-1])
        return found

    def _infer_holter_intent(self, messages: List[BaseMessage]) -> bool:
        texts = self._human_texts(messages)
        if not texts:
            return False
        latest = texts[-1].lower()
        if "holter" in latest or "动态心电" in latest or "云柚" in latest:
            return True
        # 补充追问场景：本轮没写 holter，但上一轮明确了 holter 且本轮是排序/limit补充条件
        if any(k in latest for k in ["倒序", "降序", "id", "前", "limit", "order by"]):
            history = " ".join(texts[:-1]).lower()
            if "holter" in history or "动态心电" in history:
                return True
        return False

    @staticmethod
    def _choose_holter_table(schema_tables: Dict[str, Set[str]]) -> str:
        holter_tables = [t for t in schema_tables.keys() if "holter" in t.lower()]
        if not holter_tables:
            return "holter"

        best_table = holter_tables[0]
        best_score = -1
        for t in holter_tables:
            cols = schema_tables.get(t, set())
            score = 0
            if "user_id" in cols:
                score += 4
            if "id" in cols:
                score += 3
            if any(c in cols for c in ["report_time", "usage_date", "create_time", "start_time", "update_time"]):
                score += 2
            if any(k in t.lower() for k in ["usage", "record", "report"]):
                score += 1
            if score > best_score:
                best_score = score
                best_table = t
        return best_table

    @staticmethod
    def _pick_order_column(cols: Set[str]) -> str:
        for c in ["id", "report_time", "usage_date", "create_time", "start_time", "update_time"]:
            if c in cols:
                return c
        return "id"

    def _build_holter_fallback_sql(self, user_text: str, schema_tables: Dict[str, Set[str]]) -> str:
        table = self._choose_holter_table(schema_tables)
        cols = schema_tables.get(table, set())
        limit_n = self._extract_limit(user_text, default=5)
        order_col = self._pick_order_column(cols)

        selected_cols: List[str] = []
        for c in ["id", "user_id", "user_name", "nick_name", order_col]:
            if c in cols and c not in selected_cols:
                selected_cols.append(c)
        select_expr = ", ".join(selected_cols) if selected_cols else "*"
        return f"SELECT {select_expr} FROM {table} ORDER BY {order_col} DESC LIMIT {limit_n};"

    def _get_schema_node(self, state: SqlAgentState):
        schema_info = get_schema()
        # 将 schema 信息作为 ToolMessage 添加到历史记录（模拟工具调用结果）
        # 或者直接作为 SystemMessage 的一部分（这里为了简单，作为上下文消息）
        return {"messages": [AIMessage(content=f"数据库表结构上下文已加载。")]}

    def _generate_sql_node(self, state: SqlAgentState):
        schema_info = get_schema() # 重新获取或从 state 获取
        messages = state.get("messages", [])
        latest_user_text = self._latest_human_text(messages)
        holter_intent = self._infer_holter_intent(messages)
        schema_tables = self._parse_schema_tables(schema_info)

        prompt = self.prompt
        if holter_intent:
            holter_tables = [t for t in schema_tables.keys() if "holter" in t.lower()]
            allowed_tables = holter_tables or ["holter"]
            extra_rule = (
                "【强约束】用户明确要查 holter 业务数据。"
                f"你只能使用这些表: {', '.join(allowed_tables)}。"
                "严禁使用 t_chat_message、t_chat_session 等聊天记录表。"
                "若用户要求按 id 倒序前 N 条，优先输出 ORDER BY id DESC LIMIT N。"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", SqlPrompt.GENERATE_SQL + "\n" + extra_rule),
                MessagesPlaceholder(variable_name="messages")
            ])

        chain = prompt.partial(schema=schema_info) | self.llm
        response = chain.invoke({"messages": messages})
        sql = response.content.strip().replace("```sql", "").replace("```", "")

        if holter_intent:
            used_tables = self._extract_sql_tables(sql)
            bad_tables = {"t_chat_message", "t_chat_session", "chat_message", "chat_session"}
            has_holter_table = any("holter" in t.lower() for t in used_tables)
            if (used_tables & bad_tables) or not has_holter_table:
                fallback_sql = self._build_holter_fallback_sql(latest_user_text, schema_tables)
                log.warning(f"SQL Guard: 检测到非 holter 表 SQL，已改写为: {fallback_sql}")
                sql = fallback_sql

        return {"sql_to_execute": sql}

    def _execute_sql_node(self, state: SqlAgentState):
        sql = state["sql_to_execute"]

        # 纯原生中断：该节点在 interrupt 时由 LangGraph 挂起，resume 后从此节点继续执行。
        decision_payload = {
            "action_requests": [{
                "type": "sql_approval",
                "name": "execute_sql",
                "args": {"sql": sql},
                "description": f"即将执行 SQL: {sql}，请确认是否安全。",
                "id": f"execute_sql_{hashlib.md5((sql + str(int(time.time() * 1000))).encode('utf-8')).hexdigest()[:10]}"
            }],
            "allowed_decisions": ["approve", "reject"],
            "message": "需要审批 SQL 执行"
        }
        decision = interrupt(decision_payload)

        action = decision.get("action") if isinstance(decision, dict) else decision
        if action == "reject":
            return {"messages": [AIMessage(content="用户拒绝了 SQL 执行。")]}

        try:
            result = federated_query_gateway.execute_local_sql(sql)
            result_message = ToolMessage(
                content=f"SQL 执行结果：\n{result}",
                tool_call_id="execute_sql",
                name="execute_sql"
            )
            return {"messages": [result_message], "sql_result": result}
        except Exception as e:
            error_text = f"执行 SQL 时发生错误: {str(e)}"
            return {
                "messages": [ToolMessage(content=error_text, tool_call_id="execute_sql", name="execute_sql")],
                "sql_result": error_text
            }

    def _generate_response_node(self, state: SqlAgentState):
        messages = state.get("messages", [])
        if not messages:
            return {"messages": [AIMessage(content="未获取到可展示的查询结果。")]}

        last_msg = messages[-1]
        if isinstance(last_msg, ToolMessage):
            sql = state.get("sql_to_execute", "")
            raw_result = state.get("sql_result", "") or last_msg.content.replace("SQL 执行结果：\n", "", 1)
            formatted = format_sql_result_for_user(sql, raw_result)
            return {"messages": [AIMessage(content=formatted, name="sql_agent", response_metadata={"synthetic": True})]}

        # 如果是拒绝执行等非 ToolMessage 分支，直接透传最近结果
        if isinstance(last_msg, AIMessage):
            return {"messages": [last_msg]}
        return {"messages": []}

    # 移除 run 方法，使用 BaseAgent 的通用 run (稍后重构)
