import logging
import os
import re
from typing import TypedDict, Annotated, Dict, Generator

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import add_messages, StateGraph

from agent.graph_state import AgentRequest
from agent.llm.loader_llm_multi import load_open_router
from agent.llm.model_config import ModelConfig
from utils import concurrent_executor
from utils import redis_manager

load_dotenv()
logger = logging.getLogger(__name__)

"""
代码编写子图（Code Agent）

核心特性是引入了人工断点（Human-in-the-loop）：
    1.接收用户的编程需求（例如，“用 Python 写一个快速排序函数”）。
    2.调用 LLM 生成代码。
    3.**暂停执行**：将生成的代码展示给用户，并请求审核。
    4.根据用户的反馈（例如，“接受”、“修改”或“重写”），决定是结束流程还是重新生成代码。
"""


class CodeAgentState(TypedDict):
    """
    代码编写子图状态

    Attributes:
        messages: 对话消息列表，使用 add_messages 追加新消息
        interrupt: 中断信息，包含用户输入的提示信息
        context: 外部工具的输出上下文
        out: 工作流输出，存储生成的代码或工具结果
    """
    messages: Annotated[list, add_messages]
    interrupt: str
    context: Dict
    out: str


from agent.agent_builder import create_interruptable_agent_executor


class CodeAgent:
    """
    代码编写 Agent。

    本 Agent 的核心是实现一个带有人工审核环节（Human-in-the-loop）的图。
    它首先生成代码，然后暂停等待用户审核，并根据用户的反馈决定下一步操作。

    图的构建过程被抽象到了 `create_interruptable_agent_executor` 中，
    而本类则专注于实现各个节点（agent, interrupt, router）的具体业务逻辑。
    """

    def __init__(self, req: AgentRequest, max_threads: int = 2):
        """
        初始化代码 Agent。

        Args:
            req (AgentRequest): 包含模型、会话 ID 等信息的请求对象。
            max_threads (int): 并发执行器线程数。
        """
        if not req.model:
            raise ValueError("代码编写智能体模型加载失败，请检查配置。")

        self.model = req.model
        self.state_manager = redis_manager.RedisManager()
        self.session_id = req.session_id
        self.subgraph_id = req.subgraph_id
        self.concurrent_executor = concurrent_executor.ConcurrentExecutor(max_threads=max_threads)

        # 定义 Agent 的核心提示和执行链
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个专业的软件工程师。你的任务是根据用户的需求，编写高质量、可读性强且功能正确的代码。"
                    "请在代码块中提供完整的代码实现。如果用户提供了反馈，请根据反馈修改你的代码。"
                ),
                MessagesPlaceholder(variable_name="messages")
            ]
        )
        self.chain = prompt | self.model

        # 使用通用构建器创建可中断的图执行器
        # 将节点逻辑作为参数传入，实现了逻辑与结构的解耦
        self.graph = create_interruptable_agent_executor(
            state_class=CodeAgentState,
            agent_node=self._agent_node,
            interrupt_node=self._interrupt_node,
            router_node=self._router
        )

    def _is_user_confirm(self, message: str, llm=None) -> bool:
        """
        判断用户输入是否表示确认。
        优先用本地关键词匹配，如果没命中，再用 LLM 判断。
        """
        if not message:
            return False
        normalized = message.strip().lower()
        # 关键词匹配
        if normalized in os.getenv("CONFIRM_KEYWORDS").split(","):
            return True

        # 关键词匹配失败，用 LLM 判断
        if llm:
            prompt = f"用户输入: {message}\n这是否表示确认继续？只回答 yes 或 no"
            result = llm.invoke(prompt)
            return result.content.strip().lower().startswith("y")

        return False

    def _agent_node(self, state: CodeAgentState):
        """
        定义 Agent 节点：调用 LLM 生成代码。
        """
        response = self.chain.invoke({"messages": state["messages"]})
        return {"messages": [response]}

    def _interrupt_node(self, state: CodeAgentState):
        """
        定义中断节点：准备中断信息，等待用户审核。
        """
        last_human_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        if last_human_msg and self._is_user_confirm(message=last_human_msg.content):
            return None  # 用户输入了确认指令，不中断

        last_msg = state["messages"][-1]
        if getattr(last_msg, "content", None):
            content = last_msg.content
            # 清理 <think> 等特殊标记
            clean_content_text = content.replace("<think>", "").replace("</think>", "").strip()
            # 提取代码块内容
            code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", clean_content_text, re.DOTALL)
            clean_content_text = "\n".join(code_blocks) if code_blocks else clean_content_text

            interrupt_msg = (
                f"已生成以下代码，请审核：\n\n"
                f"{clean_content_text}\n\n"
                "请输入 'ok' 接受代码，或输入您的修改意见后按回车键。"
            )
            return {"interrupt": interrupt_msg}
        return None

    def _router(self, state: CodeAgentState):
        """
        定义路由函数：根据用户反馈决定下一步是结束还是重试。
        """
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                if self._is_user_confirm(message=msg.content):
                    return "__end__"  # 用户确认，结束
                break  # 只检查最新一条用户消息
        return "agent"  # 用户提出修改意见，返回 agent 节点

    def run(self, req: AgentRequest) -> Generator:
        """
        执行子图一次，支持暂停/中断功能。
        自动加载历史状态，避免重复追加 user_input。
        """
        state = self.state_manager.load_graph_state(req.session_id, req.subgraph_id)
        if not state:
            state = {"messages": [], "interrupt": ""}
        else:
            state["interrupt"] = ""  # 清理上一次中断

        last_msg = state["messages"][-1] if state["messages"] else None
        if not last_msg or getattr(last_msg, "content", "") != req.user_input:
            state["messages"].append(HumanMessage(content=req.user_input))

        result = self.graph.invoke(state)
        self.state_manager.save_graph_state(result, req.session_id, req.subgraph_id)
        yield result



# if __name__ == '__main__':
#     config = ModelConfig(
#         model="deepseek/deepseek-chat-v3-0324:free",
#         model_key="sk-xxxx",
#         model_url="https://openrouter.ai/api/v1",
#     )
#     llm = load_open_router(config)
#     # llm = load_ollama_model("qwen3:8b")
#     # llm = load_ollama_model("gemma3:270m")
#     req = AgentRequest(
#         # user_input="ok",
#         user_input="可以",
#         # user_input="不对，要重写",
#         # user_input="写一个python的hello world并且需要格式化输出",
#         # user_input="重写",
#         model=llm,
#         session_id="123",
#         subgraph_id="code_agent",
#     )
#     code_agent = CodeAgent(req=req)
#     result = code_agent.run(req)
#     print(result["messages"][-1].content)
#
#     # 中断提示（如果触发暂停）
#     if result.get("interrupt"):
#         print("中断提示:", result["interrupt"])

