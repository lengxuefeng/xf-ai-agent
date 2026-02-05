# -*- coding: utf-8 -*-
from typing import Annotated, TypedDict, Generator

from dotenv import load_dotenv
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import add_messages, StateGraph

from agent.graph_state import AgentRequest
from agent.tools.sql_tools import get_schema, execute_sql
from utils import redis_manager

load_dotenv(verbose=True)

"""
自然语言转SQL子图（SQL Agent）。

该子图的流程设计为：
1.  接收用户的自然语言查询（例如，“研发部谁的工资最高？”）。
2.  首先自动获取数据库的表结构（schema）。
3.  将用户问题和表结构信息结合，利用 LLM 生成 SQL 查询语句。
4.  **人工断点**：将生成的 SQL 语句展示给用户，并暂停执行，等待用户确认。
5.  如果用户批准，则执行该 SQL 语句。
6.  将执行结果转换为自然语言，返回给主图。
"""


class SqlAgentState(TypedDict):
    """
    定义 SQL 子图的状态。
    """
    messages: Annotated[list, add_messages]
    interrupt: str
    sql_to_execute: str


class SqlAgent:
    """
    SQL Agent

    由于其工作流程（获取 schema -> 生成 SQL -> 中断 -> 执行 -> 生成响应）高度定制化，
    因此它保留了自己独立的图构建方法 `_build_graph`，而没有使用通用构建器。
    """

    def __init__(self, req: AgentRequest):
        if not req.model:
            raise ValueError("SQL 模型初始化失败，请检查配置。")
        self.llm = req.model
        self.redis_manager = redis_manager.RedisManager()
        self.session_id = req.session_id
        self.subgraph_id = "sql_agent"
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

    def _get_schema_node(self, state: SqlAgentState):
        schema_info = get_schema()
        schema_message = ToolMessage(content=f"数据库表结构：\n{schema_info}", tool_call_id="get_schema")
        return {"messages": [schema_message]}

    def _generate_sql_node(self, state: SqlAgentState):
        schema = state["messages"][-1].content
        chain = self.prompt.partial(schema=schema) | self.llm
        response = chain.invoke({"messages": [state["messages"][-1]]})
        return {"sql_to_execute": response.content}

    def _interrupt_node(self, state: SqlAgentState):
        sql = state["sql_to_execute"]
        interrupt_message = (
            f"将执行以下 SQL 语句，请审核：\n\n"
            f"```sql\n{sql}\n```\n\n"
            f"请输入 'ok' 确认执行，或输入您的修改意见后按回车键。"
        )
        return {"interrupt": interrupt_message}

    def _execute_sql_node(self, state: SqlAgentState):
        sql = state["sql_to_execute"]
        result = execute_sql(sql)
        result_message = ToolMessage(content=str(f"SQL 执行结果：\n{result}"), tool_call_id="execute_sql")
        return {"messages": [result_message]}

    def _generate_response_node(self, state: SqlAgentState):
        response_prompt = (
            "你是一个数据分析师。请根据用户的原始问题和以下的 SQL 执行结果，生成一句通顺的、人类可读的回答。\n"
            "原始问题: {question}\n"
            "SQL 执行结果:\n{result}"
        )
        question = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                question = msg.content
                break
        chain = ChatPromptTemplate.from_template(response_prompt) | self.llm
        response = chain.invoke({
            "question": question,
            "result": state["messages"][-1].content
        })
        return {"messages": [response]}

    def _router(self, state: SqlAgentState):
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            if last_message.content.strip().lower() == "ok":
                return "execute_sql"
            else:
                # 用户提供了修改意见，重新生成 SQL
                return "generate_sql"
        # 默认或AI消息后，结束
        return "__end__"

    def _build_graph(self):
        graph = StateGraph(SqlAgentState)
        graph.add_node("get_schema", self._get_schema_node)
        graph.add_node("generate_sql", self._generate_sql_node)
        graph.add_node("interrupt", self._interrupt_node)
        graph.add_node("execute_sql", self._execute_sql_node)
        graph.add_node("generate_response", self._generate_response_node)

        graph.set_entry_point("get_schema")
        graph.add_edge("get_schema", "generate_sql")
        graph.add_edge("generate_sql", "interrupt")
        graph.add_conditional_edges("interrupt", self._router)
        graph.add_edge("execute_sql", "generate_response")
        graph.add_edge("generate_response", "__end__")

        return graph.compile(interrupt_after=["interrupt"])

    def run(self, req: AgentRequest) -> Generator:
        """
        以流式方式执行 Agent。
        """
        state = self.redis_manager.load_graph_state(req.session_id, self.subgraph_id)
        if not state:
            state = {"messages": [], "interrupt": "", "sql_to_execute": ""}
        else:
            state["interrupt"] = ""

        last_msg = state["messages"][-1] if state["messages"] else None
        if not last_msg or getattr(last_msg, "content", "") != req.user_input:
            state["messages"].append(HumanMessage(content=req.user_input))

        final_state = None
        for event in self.graph.stream(state):
            final_state = event
            yield event

        if final_state:
            self.redis_manager.save_graph_state(final_state, req.session_id, self.subgraph_id)



if __name__ == '__main__':
    create_sql_agent()