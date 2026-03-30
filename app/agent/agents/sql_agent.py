"""
SQL Agent：处理数据库查询和 SQL 生成的智能体

核心职责：
- 理解用户的自然语言查询需求，生成对应的 SQL 语句
- 执行 SQL 查询并返回结果
- Holter（动态心电）业务查询的专门优化
- SQL 执行前的安全审批机制，防止误操作
- 智能表选择和字段选择，提升查询准确率

业务场景示例：
1. 用户问"查询最近 5 条 Holter 数据" → 生成 SQL: SELECT * FROM holter ORDER BY id DESC LIMIT 5
2. 用户问"查询用户 ID 为 123 的所有记录" → 生成 SQL: SELECT * FROM holter WHERE user_id = 123
3. 用户问"查询所有 Holter 数据" → 插入 ORDER BY 和 LIMIT，避免返回过多数据
4. 用户问"Holter 数据按时间排序，取前 10 条" → 生成 SQL: SELECT * FROM holter ORDER BY time_col DESC LIMIT 10

设计要点：
- 基于 LangGraph 构建多节点子图：获取 schema → 生成 SQL → 审批 → 执行 → 格式化结果
- Holter 业务优化：智能识别 Holter 查询意图，选择合适的表和字段
- SQL 安全防护：通过审批中断机制，所有 SQL 执行前必须用户确认
- Schema 解析：将数据库表结构转化为文本，供 LLM 理解
- SQL 守卫：检测生成 SQL 是否使用了错误表（如聊天表），自动纠正
- LIMIT 约束：默认添加 LIMIT 防止查询大量数据，可根据用户指令调整

与其他 Agent 的区别：
- weather_agent：使用天气 API
- search_agent：使用搜索引擎
- sql_agent：使用数据库查询，涉及 Schema 理解和 SQL 生成
- code_agent：处理代码相关问题

特殊设计：
- Holter 业务优化：针对医疗数据的查询特点，提供表选择和字段选择逻辑
- 审批中断：使用 LangGraph 的 interrupt 机制，实现审批挂起和恢复
- 守卫机制：检测生成的 SQL 是否包含错误表，自动生成兜底 SQL
- 多轮查询：支持基于上一轮结果进行补充查询（如排序、调整 LIMIT）

输出结果：
- SQL 执行结果的表格展示
- 友好的错误提示
- 审批确认流程
"""

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
from constants.approval_constants import ApprovalDecision, DEFAULT_ALLOWED_DECISIONS, SQL_APPROVAL_ACTION_NAME
from constants.sql_agent_constants import (
    SQL_AGENT_APPROVAL_DESC_TEMPLATE,
    SQL_AGENT_APPROVAL_MESSAGE,
    SQL_AGENT_ORDER_RULE_HINT,
    SQL_AGENT_REJECTED_MESSAGE,
    SQL_AGENT_RESULT_PREFIX,
    SQL_AGENT_SCHEMA_LOADED_MESSAGE,
)
from constants.sql_agent_keywords import (
    SQL_AGENT_KEYWORDS,
    SQL_AGENT_LIMIT_PATTERNS,
    SQL_AGENT_ORDER_COLUMN_CANDIDATES,
    SqlAgentKeywordGroup,
)
from constants.sql_tool_constants import SQL_MSG_EXEC_ERROR_PREFIX
from utils.custom_logger import get_logger, LogTarget
from agent.prompts.sql_prompt import SqlPrompt

log = get_logger(__name__)


class SqlAgentState(TypedDict):
    """SQL 子图状态

    状态字段说明：
    - messages: 消息列表，包括用户消息、AI 消息、工具消息
    - sql_to_execute: 待执行的 SQL 语句
    - sql_result: SQL 执行结果（原始数据）
    - schema_info: 数据库 schema 信息缓存，避免同一轮重复查询

    使用场景：
    - 在子图执行过程中维护状态
    - 跨节点传递消息和 SQL 信息
    - 缓存 schema 信息，避免重复查询数据库元数据
    """
    messages: Annotated[List[BaseMessage], add_messages]
    sql_to_execute: str  # 待执行的 SQL
    sql_result: str  # SQL 执行结果
    schema_info: str  # schema 缓存，避免同一轮重复查询


class SqlAgent(BaseAgent):
    """SQL Agent：处理数据库查询和 SQL 生成

    主要功能：
    - 理解自然语言查询，生成对应 SQL
    - 执行 SQL 查询并返回结果
    - Holter 业务查询优化（智能表选择、字段选择）
    - SQL 执行前的安全审批
    - 格式化查询结果，便于用户理解

    工作流程：
    1. 获取数据库 schema 信息
    2. 分析用户查询，识别 Holter 查询意图
    3. 生成 SQL 语句（LLM 生成 + 守卫修正）
    4. 审批 SQL 执行（用户确认）
    5. 执行 SQL 并获取结果
    6. 格式化结果并返回

    典型使用场景：
    - "查询最近 5 条 Holter 数据"
    - "查询用户 ID 为 123 的记录"
    - "按时间排序，取前 10 条记录"
    - "查询 Holter 表中所有字段"

    不适合场景：
    - 互联网信息搜索（应使用 search_agent）
    - 代码相关问题（应使用 code_agent）
    - 天气查询（应使用 weather_agent）

    特殊功能：
    - Holter 业务优化：智能识别医疗数据查询，选择合适表和字段
    - 审批机制：所有 SQL 执行前必须用户确认
    - SQL 守卫：检测错误表，自动纠正
    - LIMIT 约束：默认限制查询条数，可按需调整
    """

    def __init__(self, req: AgentRequest):
        """
        初始化 SQL Agent

        参数说明：
        - req: Agent 请求对象，包含模型配置、会话信息等

        初始化步骤：
        1. 调用父类 BaseAgent 初始化
        2. 验证模型配置，确保 SQL 模型可用
        3. 创建 LLM 实例
        4. 构建生成 SQL 的提示词模板
        5. 构建 SQL 子图

        异常处理：
        - 如果模型未配置，抛出 ValueError 提示检查配置

        示例：
        >>> req = AgentRequest(model=llm, session_id="xxx")
        >>> sql_agent = SqlAgent(req)
        >>> result = sql_agent.invoke({"messages": [HumanMessage("查询最近 5 条记录")]})
        """
        super().__init__(req)
        if not req.model:
            raise ValueError("SQL 模型初始化失败，请检查配置。")
        self.llm = req.model
        self.subgraph_id = "sql_agent"

        # 提示词模板：生成 SQL 用的提示词
        # GENERATE_SQL 包含 SQL 生成规则、schema 信息占位符等
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SqlPrompt.GENERATE_SQL),
                MessagesPlaceholder(variable_name="messages")
            ]
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        构建 SQL 子图

        子图结构：
        1. get_schema：获取数据库 schema 信息
        2. generate_sql：根据用户问题生成 SQL
        3. execute_sql：执行 SQL（含审批中断）
        4. generate_response：格式化结果并返回

        流程：START → get_schema → generate_sql → execute_sql → generate_response → END

        返回值：
        - StateGraph: 编译后的可执行子图，使用全局 checkpointer
        """
        workflow = StateGraph(SqlAgentState)

        # 添加节点
        workflow.add_node("get_schema", self._get_schema_node)
        workflow.add_node("generate_sql", self._generate_sql_node)
        workflow.add_node("execute_sql", self._execute_sql_node)
        workflow.add_node("generate_response", self._generate_response_node)

        # 定义边：线性的执行流程
        workflow.add_edge(START, "get_schema")
        workflow.add_edge("get_schema", "generate_sql")
        workflow.add_edge("generate_sql", "execute_sql")
        workflow.add_edge("execute_sql", "generate_response")
        workflow.add_edge("generate_response", END)

        # 编译图 (使用全局 checkpointer，支持中断恢复)
        return workflow.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _latest_human_text(messages: List[BaseMessage]) -> str:
        """
        提取最近一条用户消息

        设计目的：
        - 在多轮对话中找到用户的最新提问
        - 用于生成 SQL 时获取用户原始问题
        - 用于提取 LIMIT 条数等参数

        实现逻辑：
        - 从消息列表末尾反向遍历
        - 找到第一个 HumanMessage 类型消息
        - 提取其文本内容并去除首尾空白

        参数说明：
        - messages: 消息列表

        返回值：
        - str: 最新的人类消息文本，如果没有返回空字符串

        示例：
        >>> messages = [
        ...     HumanMessage("查询所有记录"),
        ...     AIMessage("我需要知道更多细节"),
        ...     HumanMessage("查询最近 5 条")
        ... ]
        >>> SqlAgent._latest_human_text(messages)
        "查询最近 5 条"
        """
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return (msg.content or "").strip()
        return ""

    @staticmethod
    def _human_texts(messages: List[BaseMessage]) -> List[str]:
        """
        提取所有用户消息

        设计目的：
        - 获取对话历史中所有用户输入
        - 用于判断查询上下文（如是否是 Holter 业务）
        - 用于检测多轮查询的上下文关系

        实现逻辑：
        - 遍历所有消息
        - 筛选出 HumanMessage 类型
        - 提取文本内容并去除空白

        参数说明：
        - messages: 消息列表

        返回值：
        - List[str]: 所有用户消息文本列表

        示例：
        >>> messages = [
        ...     HumanMessage("查询 Holter 数据"),
        ...     AIMessage("好的"),
        ...     HumanMessage("按时间排序"),
        ...     AIMessage("好的")
        ... ]
        >>> SqlAgent._human_texts(messages)
        ["查询 Holter 数据", "按时间排序"]
        """
        return [(m.content or "").strip() for m in messages if isinstance(m, HumanMessage)]

    @staticmethod
    def _extract_limit(text: str, default: int = 10) -> int:
        """
        提取 LIMIT 条数

        设计目的：
        - 从用户文本中提取查询的条数限制
        - 用于生成 SQL 时的 LIMIT 子句
        - 避免默认返回过多数据

        检测模式：
        - "前 N 条" → N
        - "查询 N 条" → N
        - "N 个" → N
        - 等等（定义在 SQL_AGENT_LIMIT_PATTERNS 中）

        参数说明：
        - text: 用户文本
        - default: 默认条数，当未检测到时使用

        返回值：
        - int: 提取到的条数

        示例：
        >>> SqlAgent._extract_limit("查询前 5 条记录")
        5
        >>> SqlAgent._extract_limit("查询所有记录", default=10)
        10
        """
        t = (text or "").lower()
        for pattern in SQL_AGENT_LIMIT_PATTERNS:
            m = re.search(pattern, t)
            if m:
                return int(m.group(1))
        return default

    @staticmethod
    def _parse_schema_tables(schema_info: str) -> Dict[str, Set[str]]:
        """
        解析 schema 为 {表名: {字段名集合}}

        设计目的：
        - 将 schema 文本解析为结构化数据
        - 便于后续查询表名和字段名
        - 用于 SQL 守卫（检测是否使用了正确表）

        Schema 格式：
        ```
        表名: user
        字段名 | 类型 | 说明
        id | int | 主键
        name | varchar | 用户名

        表名: holter
        字段名 | 类型 | 说明
        user_id | int | 用户 ID
        time | timestamp | 记录时间
        ```

        参数说明：
        - schema_info: schema 文本信息

        返回值：
        - Dict[str, Set[str]]: {表名: {字段名集合}}

        示例：
        >>> schema = "表名: user\\nid | int | 主键\\nname | varchar | 用户名"
        >>> SqlAgent._parse_schema_tables(schema)
        {"user": {"id", "name"}}
        """
        tables: Dict[str, Set[str]] = {}
        current_table = ""

        for raw in (schema_info or "").splitlines():
            line = raw.strip()
            if not line:
                continue

            # 检测表名行：表名: xxx
            if line.startswith("表名:"):
                current_table = line.split(":", 1)[1].strip()
                if current_table:
                    tables.setdefault(current_table, set())
                continue

            # 检测字段行：id | int | 主键
            if current_table and "|" in line and not line.startswith("---") and not line.startswith("字段名"):
                col = line.split("|", 1)[0].strip()
                if col:
                    tables[current_table].add(col)

        return tables

    @staticmethod
    def _extract_sql_tables(sql: str) -> Set[str]:
        """
        从 SQL 中提取涉及的表名

        设计目的：
        - 检测生成的 SQL 使用了哪些表
        - 用于 SQL 守卫（是否使用了错误表）
        - 用于 Holter 业务守卫（是否使用了 Holter 表）

        检测逻辑：
        - 正则匹配 FROM、JOIN、UPDATE、INTO 后的表名
        - 提取最后一部分（支持 schema.table 格式）

        参数说明：
        - sql: SQL 语句

        返回值：
        - Set[str]: 涉及的表名集合

        示例：
        >>> SqlAgent._extract_sql_tables("SELECT * FROM user JOIN holter ON user.id = holter.user_id")
        {"user", "holter"}
        """
        t = (sql or "").strip().lower()
        # 匹配 FROM、JOIN、UPDATE、INTO 后的表名
        pattern = r"\b(?:from|join|update|into)\s+([a-zA-Z_][a-zA-Z0-9_\.]*)"
        found = set()
        for table in re.findall(pattern, t):
            # 提取最后一部分（支持 schema.table 格式）
            found.add(table.split(".")[-1])
        return found

    def _infer_holter_intent(self, messages: List[BaseMessage]) -> bool:
        """
        识别是否查询 Holter 数据

        设计目的：
        - 检测用户是否要查询 Holter（动态心电）业务数据
        - Holter 查询需要特殊处理（限制表、强制排序等）
        - 支持多轮查询的上下文判断

        检测逻辑：
        1. 直接检测：最新用户消息中是否包含 Holter 关键词
        2. 上下文检测：
           - 最新消息包含排序或 LIMIT 指令
           - 历史消息中包含 Holter 关键词
           - 判定为 Holter 查询的补充指令

        Holter 关键词：holter、动态心电、心电图等
        排序关键词：排序、按...排序、order by 等
        LIMIT 关键词：前 N 条、N 条、limit 等

        参数说明：
        - messages: 消息列表

        返回值：
        - bool: True 表示识别为 Holter 查询，False 表示非 Holter 查询

        示例：
        >>> messages = [HumanMessage("查询最近 5 条 Holter 数据")]
        >>> SqlAgent()._infer_holter_intent(messages)
        True
        >>> messages = [
        ...     HumanMessage("查询 Holter 数据"),
        ...     AIMessage("好的"),
        ...     HumanMessage("按时间排序")
        ... ]
        >>> SqlAgent()._infer_holter_intent(messages)
        True
        """
        texts = self._human_texts(messages)
        if not texts:
            return False

        latest = texts[-1].lower()

        # 直接检测：最新消息是否包含 Holter 关键词
        if any(k in latest for k in SQL_AGENT_KEYWORDS[SqlAgentKeywordGroup.HOLTER_DOMAIN]):
            return True

        # 上下文检测：本轮没写 holter，但上一轮明确了 holter 且本轮是排序/limit 补充条件
        if any(k in latest for k in SQL_AGENT_KEYWORDS[SqlAgentKeywordGroup.ORDER_HINT]):
            history = " ".join(texts[:-1]).lower()
            if any(k in history for k in SQL_AGENT_KEYWORDS[SqlAgentKeywordGroup.HOLTER_DOMAIN]):
                return True

        return False

    @staticmethod
    def _choose_holter_table(schema_tables: Dict[str, Set[str]]) -> str:
        """
        选择最像 Holter 业务表的表

        设计目的：
        - 在多个可能的 Holter 表中选择最合适的一个
        - 基于表名和字段进行评分
        - 提升查询准确率

        评分规则：
        - 包含 user_id 字段：+4 分
        - 包含 id 字段：+3 分
        - 包含时间字段（time、timestamp 等）：+2 分
        - 表名包含 holter 关键词：+1 分

        参数说明：
        - schema_tables: {表名: {字段名集合}}

        返回值：
        - str: 最合适的表名，如果没有则返回 "holter"

        示例：
        >>> schema_tables = {
        ...     "holter_data": {"id", "user_id", "time"},
        ...     "user": {"id", "name"}
        ... }
        >>> SqlAgent._choose_holter_table(schema_tables)
        "holter_data"
        """
        # 筛选出可能的 Holter 表（表名包含 holter、data、record 等关键词）
        holter_tables = [t for t in schema_tables.keys() if any(k in t.lower() for k in SQL_AGENT_KEYWORDS[SqlAgentKeywordGroup.HOLTER_TABLE_HINT])]
        if not holter_tables:
            return "holter"

        # 评分选择最佳表
        best_table = holter_tables[0]
        best_score = -1
        for t in holter_tables:
            cols = schema_tables.get(t, set())
            score = 0

            if "user_id" in cols:
                score += 4
            if "id" in cols:
                score += 3
            if any(c in cols for c in SQL_AGENT_KEYWORDS[SqlAgentKeywordGroup.HOLTER_TIME_COLUMNS]):
                score += 2
            if any(k in t.lower() for k in SQL_AGENT_KEYWORDS[SqlAgentKeywordGroup.HOLTER_TABLE_NAME_HINT]):
                score += 1

            if score > best_score:
                best_score = score
                best_table = t

        return best_table

    @staticmethod
    def _pick_order_column(cols: Set[str]) -> str:
        """
        选择排序字段

        设计目的：
        - 为查询选择合适的排序字段
        - 优先选择 id、时间等字段
        - 确保查询结果有序，便于用户理解

        候选字段（按优先级）：
        1. id
        2. time、created_at、updated_at 等时间字段

        参数说明：
        - cols: 可用字段集合

        返回值：
        - str: 排序字段名，默认 "id"

        示例：
        >>> cols = {"id", "name", "time", "user_id"}
        >>> SqlAgent._pick_order_column(cols)
        "id"
        """
        for c in SQL_AGENT_ORDER_COLUMN_CANDIDATES:
            if c in cols:
                return c
        return "id"

    def _build_holter_fallback_sql(self, user_text: str, schema_tables: Dict[str, Set[str]]) -> str:
        """
        构建 Holter 查询的兜底 SQL

        设计目的：
        - 当 LLM 生成的 SQL 不符合要求时，自动生成兜底 SQL
        - 用于 SQL 守卫检测到错误表时的修正
        - 确保返回有效的查询结果

        构建逻辑：
        1. 选择最合适的 Holter 表
        2. 从用户文本中提取 LIMIT 条数
        3. 选择排序字段
        4. 选择要展示的字段（优先时间、user_id 等关键字段）
        5. 生成 SQL: SELECT fields FROM table ORDER BY col DESC LIMIT n

        参数说明：
        - user_text: 用户原始文本
        - schema_tables: schema 信息

        返回值：
        - str: 兜底 SQL 语句

        示例：
        >>> user_text = "查询最近 5 条记录"
        >>> schema_tables = {"holter": {"id", "user_id", "time"}}
        >>> SqlAgent()._build_holter_fallback_sql(user_text, schema_tables)
        "SELECT id, user_id, time FROM holter ORDER BY id DESC LIMIT 5;"
        """
        table = self._choose_holter_table(schema_tables)
        cols = schema_tables.get(table, set())
        limit_n = self._extract_limit(user_text, default=5)
        order_col = self._pick_order_column(cols)

        # 选择要展示的字段：优先选择 Holter 业务关键字段 + 排序字段
        selected_cols: List[str] = []
        for c in (*SQL_AGENT_KEYWORDS[SqlAgentKeywordGroup.HOLTER_SELECT_COLUMNS], order_col):
            if c in cols and c not in selected_cols:
                selected_cols.append(c)

        # 如果没有选中任何字段，使用 *
        select_expr = ", ".join(selected_cols) if selected_cols else "*"
        return f"SELECT {select_expr} FROM {table} ORDER BY {order_col} DESC LIMIT {limit_n};"

    def _get_schema_node(self, state: SqlAgentState):
        """
        获取数据库 schema 信息

        功能说明：
        - 调用 get_schema() 获取数据库表结构
        - 将 schema 信息缓存到状态中
        - 传递给后续节点使用

        返回值：
        - dict: 更新后的状态，包含 schema_info 和消息

        使用场景：
        - 子图第一个节点
        - 为 SQL 生成提供表结构信息
        """
        schema_info = get_schema()
        return {
            "schema_info": schema_info,
            "messages": [AIMessage(content=SQL_AGENT_SCHEMA_LOADED_MESSAGE)],
        }

    def _generate_sql_node(self, state: SqlAgentState):
        """
        生成 SQL 语句

        功能说明：
        - 根据用户问题和 schema 信息生成 SQL
        - 如果是 Holter 查询，添加约束规则
        - SQL 守卫：检测是否使用了错误表，如有则修正

        处理流程：
        1. 获取 schema 信息和用户问题
        2. 检测是否是 Holter 查询
        3. 如果是 Holter 查询，添加约束规则（只能用 Holter 表，必须排序）
        4. 调用 LLM 生成 SQL
        5. 如果是 Holter 查询，检测生成的 SQL
           - 检查是否使用了错误表（如聊天表）
           - 检查是否使用了 Holter 表
           - 如有问题，生成兜底 SQL
        6. 返回生成的 SQL

        参数说明：
        - state: 子图状态

        返回值：
        - dict: 更新后的状态，包含 sql_to_execute

        Holter 约束规则：
        - 只能使用 Holter 相关表（不能使用 t_chat_message、t_chat_session 等）
        - 必须包含 ORDER BY 和 LIMIT
        - 推荐按时间或 ID 降序排序
        """
        schema_info = state.get("schema_info") or get_schema()
        messages = state.get("messages", [])
        latest_user_text = self._latest_human_text(messages)
        holter_intent = self._infer_holter_intent(messages)
        schema_tables = self._parse_schema_tables(schema_info)

        prompt = self.prompt

        # 如果是 Holter 查询，添加约束规则
        if holter_intent:
            holter_tables = [t for t in schema_tables.keys() if any(k in t.lower() for k in SQL_AGENT_KEYWORDS[SqlAgentKeywordGroup.HOLTER_TABLE_HINT])]
            allowed_tables = holter_tables or ["holter"]

            extra_rule = (
                "【强约束】用户明确要查 holter 业务数据。"
                f"你只能使用这些表: {', '.join(allowed_tables)}。"
                "严禁使用 t_chat_message、t_chat_session 等聊天记录表。"
                f"{SQL_AGENT_ORDER_RULE_HINT}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", SqlPrompt.GENERATE_SQL + "\n" + extra_rule),
                MessagesPlaceholder(variable_name="messages")
            ])

        # 调用 LLM 生成 SQL
        chain = prompt.partial(schema=schema_info) | self.llm
        response = chain.invoke({"messages": messages})
        sql = response.content.strip().replace("```sql", "").replace("```", "")

        # 如果是 Holter 查询，SQL 守卫检测
        if holter_intent:
            used_tables = self._extract_sql_tables(sql)
            bad_tables = set(SQL_AGENT_KEYWORDS[SqlAgentKeywordGroup.HOLTER_BAD_TABLES])
            has_holter_table = any(any(k in t.lower() for k in SQL_AGENT_KEYWORDS[SqlAgentKeywordGroup.HOLTER_TABLE_HINT]) for t in used_tables)

            # 检测：使用了错误表 或 没有使用 Holter 表
            if (used_tables & bad_tables) or not has_holter_table:
                fallback_sql = self._build_holter_fallback_sql(latest_user_text, schema_tables)
                log.warning(f"SQL Guard: 检测到非 holter 表 SQL，已改写为: {fallback_sql}")
                sql = fallback_sql

        return {"sql_to_execute": sql}

    def _execute_sql_node(self, state: SqlAgentState):
        """
        执行 SQL（含审批中断）

        功能说明：
        - 使用 LangGraph 的 interrupt 机制实现审批中断
        - 挂起子图执行，等待用户审批决策
        - 根据审批结果执行或拒绝 SQL

        审批流程：
        1. 构建审批请求数据（包含 SQL 和描述）
        2. 调用 interrupt 挂起执行
        3. 等待用户审批（在 Resume 时恢复）
        4. 根据审批决策执行或拒绝

        决策类型：
        - APPROVE: 批准执行 SQL
        - REJECT: 拒绝执行 SQL
        - MODIFY: 修改 SQL 后执行（预留）

        参数说明：
        - state: 子图状态

        返回值：
        - dict: 更新后的状态，包含消息和结果
        - 如拒绝：返回拒绝消息
        - 如执行：返回执行结果（ToolMessage）

        异常处理：
        - SQL 执行错误时，返回错误消息

        使用场景：
        - 所有 SQL 执行前必须审批
        - 防止误操作和恶意查询
        """
        sql = state["sql_to_execute"]

        # 构建审批请求数据
        # action_requests: 审批动作列表
        # allowed_decisions: 允许的决策选项
        # message: 审批提示消息
        decision_payload = {
            "action_requests": [{
                "type": "sql_approval",
                "name": SQL_APPROVAL_ACTION_NAME,
                "args": {"sql": sql},
                "description": SQL_AGENT_APPROVAL_DESC_TEMPLATE.format(sql=sql),
                "id": f"{SQL_APPROVAL_ACTION_NAME}_{hashlib.md5((sql + str(int(time.time() * 1000))).encode('utf-8')).hexdigest()[:10]}"
            }],
            "allowed_decisions": list(DEFAULT_ALLOWED_DECISIONS),
            "message": SQL_AGENT_APPROVAL_MESSAGE,
        }

        # 纯原生中断：该节点在 interrupt 时由 LangGraph 挂起，resume 后从此节点继续执行
        decision = interrupt(decision_payload)

        # 处理审批决策
        action = decision.get("action") if isinstance(decision, dict) else decision
        if action == ApprovalDecision.REJECT.value:
            return {"messages": [AIMessage(content=SQL_AGENT_REJECTED_MESSAGE)]}

        # 执行 SQL
        try:
            result = federated_query_gateway.execute_local_sql(sql)
            result_message = ToolMessage(
                content=f"{SQL_AGENT_RESULT_PREFIX}{result}",
                tool_call_id=SQL_APPROVAL_ACTION_NAME,
                name=SQL_APPROVAL_ACTION_NAME,
            )
            return {"messages": [result_message], "sql_result": result}
        except Exception as e:
            error_text = f"{SQL_MSG_EXEC_ERROR_PREFIX}{str(e)}"
            return {
                "messages": [ToolMessage(content=error_text, tool_call_id=SQL_APPROVAL_ACTION_NAME, name=SQL_APPROVAL_ACTION_NAME)],
                "sql_result": error_text
            }

    def _generate_response_node(self, state: SqlAgentState):
        """
        格式化查询结果并返回

        功能说明：
        - 将 SQL 执行结果格式化为用户友好的表格
        - 处理拒绝执行等情况
        - 返回最终回复

        处理逻辑：
        1. 获取最新消息
        2. 如果是 ToolMessage（执行成功），格式化结果
        3. 如果是 AIMessage（如拒绝执行），直接透传
        4. 返回格式化后的消息

        参数说明：
        - state: 子图状态

        返回值：
        - dict: 更新后的状态，包含格式化后的消息

        格式化说明：
        - format_sql_result_for_user 将原始 SQL 结果转换为 Markdown 表格
        - 添加 synthetic 标记，表示这是 AI 生成的回复
        """
        messages = state.get("messages", [])
        if not messages:
            return {"messages": [AIMessage(content="未获取到可展示的查询结果。")]}

        last_msg = messages[-1]

        # 如果是 ToolMessage（SQL 执行成功），格式化结果
        if isinstance(last_msg, ToolMessage):
            sql = state.get("sql_to_execute", "")
            raw_result = state.get("sql_result", "") or last_msg.content.replace(SQL_AGENT_RESULT_PREFIX, "", 1)
            formatted = format_sql_result_for_user(sql, raw_result)
            return {"messages": [AIMessage(content=formatted, name="sql_agent", response_metadata={"synthetic": True})]}

        # 如果是 AIMessage（如拒绝执行），直接透传
        if isinstance(last_msg, AIMessage):
            return {"messages": [last_msg]}

        return {"messages": []}

    # 移除 run 方法，使用 BaseAgent 的通用 run (稍后重构)
