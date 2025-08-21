import io
from typing import Annotated, TypedDict

from IPython.core.display_functions import display
from dotenv import load_dotenv
from langchain.agents import Agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode

from agent.graph_state import AgentRequest
from agent.llm.loader_llm_multi import load_open_router
from agent.llm.ollama_model import load_ollama_model
from agent.tools.sql_tools import get_schema, execute_sql
from IPython.display import Image as IPyImage  # 重命名为 IPyImage
from PIL import Image as PILImage  # 重命名为 PILImage，避免冲突

from utils import redis_manager

"""
自然语言转SQL子图（SQL Agent）。

该子图的流程设计为：
    1.接收用户的自然语言查询（例如，“研发部谁的工资最高？”）。
    2.首先自动获取数据库的表结构（schema）。
    3.将用户问题和表结构信息结合，利用 LLM 生成 SQL 查询语句。
    4.**人工断点**：将生成的 SQL 语句展示给用户，并暂停执行，等待用户确认。
        这是为了防止 AI 生成破坏性或错误的 SQL，是生产环境中的重要安全措施。
    5.如果用户批准，则执行该 SQL 语句。
    6.将执行结果转换为自然语言，返回给主图。
"""
load_dotenv(verbose=True)


class SqlAgentState(TypedDict):
    """
    定义 SQL 子图的状态。

    Attributes:
        messages (Annotated[list, add_messages]): 对话消息列表。
        interrupt (str): 中断信息字段，用于需要人工审核时暂停图的执行。
        sql_to_execute (str): 存储由模型生成、待用户确认执行的 SQL 语句。
    """
    messages: Annotated[list, add_messages]
    interrupt: str
    sql_to_execute: str


class SqlAgent:
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
        """
        获取数据库表结构节点。

        该节点负责获取数据库的表结构信息，并将其添加到状态中。
        """
        # 获取表结构
        schema_info = get_schema()
        # 将工具返回的结构包装成TooMessage
        schema_message = ToolMessage(
            content=f"数据库表结构：\n{schema_info}",
            tool_call_id="get_schema"
        )
        return {"messages": [schema_message]}

    def _generate_sql_node(self, state: SqlAgentState):
        """
        生成 SQL 节点。

        该节点负责根据用户问题和表结构信息，利用 LLM 生成 SQL 查询语句。
        """
        # 再次获取 schema 以确保信息最新
        # schema = get_schema()
        schema = state["messages"][-1].content
        # LangChain 调用：使用 .partial 方法预填充提示中的 schema，然后与模型链接
        chain = self.prompt.partial(schema=schema) | self.llm
        # LangChain 调用：执行链。只传入最新的用户消息以避免混淆。
        response = chain.invoke({"messages": [state["messages"][-1]]})
        # 将生成的 SQL 存储在状态的 `sql_to_execute` 字段中，待用户确认
        return {"sql_to_execute": response.content}

    def _interrupt_node(self, state: SqlAgentState):
        """
        中断节点。

        该节点负责在生成 SQL 后，等待用户确认。
        """
        # 等待用户确认
        sql = state["sql_to_execute"]
        interrupt_message = (
            f"将执行以下 SQL 语句，请审核：\n\n"
            f"```sql\n{sql}\n```\n\n"
            f"请输入 'ok' 确认执行，或输入您的修改意见后按回车键。"
        )
        return {"interrupt": interrupt_message}

    def _execute_sql_node(self, state: SqlAgentState):
        """
        执行 SQL 节点。

        该节点负责执行用户确认后的 SQL 语句。
        """
        # 执行 SQL 语句
        sql = state["sql_to_execute"]
        # 执行 SQL 语句
        result = execute_sql(sql)
        result_message = ToolMessage(content=str(f"SQL 执行结果：\n{result}"), tool_call_id="execute_sql")
        return {"messages": [result_message]}

    def _generate_response_node(self, state: SqlAgentState):
        """
        定义生成最终自然语言回答的节点。

        这个节点将 SQL 执行结果转换为用户易于理解的自然语言。
        """
        # 定义一个新的提示，指导模型扮演数据分析师
        response_prompt = (
            "你是一个数据分析师。请根据用户的原始问题和以下的 SQL 执行结果，生成一句通顺的、人类可读的回答。\n"
            "原始问题: {question}\n"
            "SQL 执行结果:\n{result}"
        )
        # 从历史问题找出用户的原始wenti
        question = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                question = msg.content
                break
        # LangChain 调用：创建一个新的提示链
        chain = ChatPromptTemplate.from_template(response_prompt) | self.llm
        # LangChain 调用：执行链，传入原始问题和 SQL 结果
        response = chain.invoke({
            "question": question,  # 对应response_prompt里面的{question}
            "result": state["messages"][-1].content  # 对应response_prompt里面的{result}
        })
        return {"messages": [response]}

    def _router(self, state: SqlAgentState):
        """
        定义路由函数，根据状态确定下一个节点。
        """
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            if last_message.content.strip().lower() == "ok":
                # 如果用户同意，则走向 'execute_sql' 节点
                return "execute_sql"
            else:
                # 如果用户提供了修改意见，将其添加到状态中
                state["sql_to_execute"] = last_message.content
                return "generate_sql"
        else:
            return "__end__"

    def _build_graph(self):
        """
        构建 SQL 智能体的状态图。
        """
        graph = StateGraph(SqlAgentState)
        graph.add_node("get_schema", self._get_schema_node)
        graph.add_node("generate_sql", self._generate_sql_node)
        graph.add_node("interrupt", self._interrupt_node)
        graph.add_node("execute_sql", self._execute_sql_node)
        graph.add_node("generate_response", self._generate_response_node)

        # LangChain 调用：设置图的入口点
        graph.set_entry_point("get_schema")
        # LangChain 调用：添加入口点到生成SQL节点的边
        graph.add_edge("get_schema", "generate_sql")
        # LangChain 调用：添加生成SQL节点到中断节点的边
        graph.add_edge("generate_sql", "interrupt")

        # LangChain 调用：添加从中断节点出发的条件边
        graph.add_conditional_edges("interrupt", self._router)
        # LangChain 调用：添加执行SQL节点到生成回答节点的边
        graph.add_edge("execute_sql", "generate_response")
        # LangChain 调用：添加生成回答节点到结束节点的边
        graph.add_edge("generate_response", "__end__")

        return graph.compile(interrupt_after=["interrupt"])

    def run(self, req: AgentRequest):
        """
        执行子图一次，支持暂停/中断功能
        自动加载历史状态，避免重复追加 user_input
        """
        # 1. 拉取历史
        state = self.redis_manager.load_graph_state(req.session_id, req.subgraph_id)
        if not state:
            state = {"messages": [], "interrupt": ""}
        else:
            # 清理上一次中断，保证新一轮中断独立
            state["interrupt"] = ""

        # 2. 避免重复追加用户输入
        last_msg = state["messages"][-1] if state["messages"] else None
        if not last_msg or getattr(last_msg, "content", "") != req.user_input:
            state["messages"].append(HumanMessage(content=req.user_input))

        # 3. 执行子图
        result = self.graph.invoke(state)

        # 4. 保存执行结果
        self.redis_manager.save_graph_state(result, req.session_id, req.subgraph_id)

        return result


if __name__ == '__main__':
    llm = load_open_router("deepseek/deepseek-chat-v3-0324:free")
    # llm = load_ollama_model("qwen3:8b")
    # llm = load_ollama_model("gemma3:270m")
    req = AgentRequest(
        user_input="ok",
        # user_input="目前用户设置的MCP有哪些？",
        # user_input="不对，要重写",
        # user_input="写一个python的hello world并且需要格式化输出",
        # user_input="重写",
        model=llm,
        session_id="1231111",
        subgraph_id="code_agent",
    )
    sql_agent = SqlAgent(req=req)
    result = sql_agent.run(req)
    print(result["messages"][-1].content)

    # 中断提示（如果触发暂停）
    if result.get("interrupt"):
        print("中断提示:", result["interrupt"])
