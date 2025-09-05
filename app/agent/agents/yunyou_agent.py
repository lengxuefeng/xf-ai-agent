import logging
from typing import TypedDict, Annotated, Generator

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import add_messages

from agent.agent_builder import create_yunyou_agent_executor
from agent.graph_state import AgentRequest
from agent.llm.ollama_model import load_ollama_model
from agent.tools.yunyou_tools import holter_list, holter_type_count, holter_report_count
from utils import redis_manager

load_dotenv()
logger = logging.getLogger(__name__)

"""
定义和创建云柚相关子图（Yunyou Agent）。

功能：
    1. 接收用户输入。
    2. 利用云柚相关工具获取信息。
    3. 整合信息，生成自然语言回答。
    4. 返回回答给主图。
"""


class YunyouState(TypedDict):
    messages: Annotated[list, add_messages]


class YunyouAgent:
    def __init__(self, req: AgentRequest):
        """
        初始化云柚 Agent。

        Args:
            req (AgentRequest): 包含模型、会话 ID 等信息的请求对象。
        """
        if not req.model:
            logger.error("云柚模型初始化失败，模型对象为空")
            raise ValueError("云柚模型初始化失败，请检查配置。")
        self.model = req.model

        # 初始化状态管理器
        try:
            self.redis_manager = redis_manager.RedisManager()
            logger.info("云柚 Agent Redis 连接初始化成功")
        except Exception as e:
            logger.error(f"云柚 Agent Redis 连接初始化失败: {e}")
            raise

        self.session_id = req.session_id
        self.subgraph_id = "yunyou_agent"

        # 定义 Agent 的核心提示和执行链
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    你是一个专业的 Holter 数据分析助手，负责根据用户的查询需求调用 YunYou 子图工具。

                    【工具说明】
                    1. holter_list  
                       - 用途：查询 Holter 列表信息  
                       - 必须参数：
                            startUseDay (str) : 开始日期，格式 yyyy-MM-dd
                            endUseDay   (str) : 结束日期，格式 yyyy-MM-dd
                       - 可选参数：
                            isUploaded (int)  : 0=否, 1=是, -1=无数据
                            reportStatus (int): -1=无数据, 0=待审核, 1=审核中, 2=人工审核完成, 3=自动审核完成
                            holterType (int)  : 0=24小时, 1=2小时, 2=24小时(夜间), 3=48小时

                    2. holter_type_count  
                       - 用途：查询 Holter 类型统计  
                       - 必须参数：
                            startUseDay (str)
                            endUseDay   (str)

                    3. holter_report_count  
                       - 用途：查询 Holter 报告状态统计  
                       - 必须参数：
                            startUseDay (str)
                            endUseDay   (str)

                    【执行规范】
                    - 首先，分析用户的请求，严格映射到对应的工具。
                    - 如果用户输入的内容无法唯一确定工具，则结合上下文推理最合适的工具。
                    - 在调用工具时，必须提供完整参数：  
                      - 如果用户提供了起止日期，直接使用。  
                      - 如果用户未提供日期，默认使用最近一天（startUseDay = endUseDay = 今天）。  
                    - 工具调用后，必须根据返回结果生成自然语言总结，使用通俗易懂的表述，不要直接照搬 JSON。

                    【回答示例】
                    用户问：“帮我查一下 2020-07-30 的 holter 报告审核情况”  
                    → 使用 holter_report_count 工具，参数 {{startUseDay:"2020-07-30", endUseDay:"2020-07-30"}}  
                    → 假设返回:
                    \"\"\"
                    [{{"count":12,"reportStatus":2}}, {{"count":5,"reportStatus":3}}]
                    \"\"\"
                    → 回答：“2020-07-30 总共有 17 份 Holter 报告，其中 12 份已人工审核完成，5 份已自动审核完成。”

                    请严格遵循以上规范。
                    """
                ),
                MessagesPlaceholder(variable_name="messages")
            ]
        )
        self.chain = prompt | self.model

        tools = [holter_list, holter_type_count, holter_report_count]
        try:
            self.graph = create_yunyou_agent_executor(
                chain=self.chain,
                tools=tools,
                state_class=YunyouState
            )
            logger.info("云柚 Agent 执行器创建成功")
        except Exception as e:
            logger.error(f"云柚 Agent 执行器创建失败: {e}")
            raise

    def run(self, req: AgentRequest) -> Generator:
        """
        执行 Agent 的主流程。

        该方法负责加载历史状态、追加新消息、调用图执行器，并保存最新状态。

        Args:
            req (AgentRequest): 包含用户输入的请求对象。

        Yields:
            dict: 执行过程中的事件流。
        """
        # 从 Redis 加载当前会话的历史状态
        state = self.redis_manager.load_graph_state(req.session_id, subgraph_id=self.subgraph_id)
        if not state:
            state = {"messages": []}

        # 避免重复追加相同的用户输入
        last_msg = state["messages"][-1] if state["messages"] else None
        if not last_msg or getattr(last_msg, "content", None) != req.user_input:
            state["messages"].append(HumanMessage(content=req.user_input))

        # 以流式方式调用图执行器
        final_state = None
        for event in self.graph.stream(state):
            final_state = event
            yield event

        # 将执行后的新状态保存回 Redis
        if final_state:
            self.redis_manager.save_graph_state(final_state, req.session_id, req.subgraph_id)

#
# if __name__ == '__main__':
#     # llm = load_open_router("deepseek/deepseek-chat-v3-0324:free")
#     llm = load_ollama_model("qwen3:8b")
#     agent_req = AgentRequest(
#         user_input="今天holter的类型有哪些，每个类型的数量是多少",
#         model=llm,
#         session_id="session_1756001239211_uqdut8idsdsv2",
#         subgraph_id="yunyou_agent",
#     )
#     yunyou_agent = YunyouAgent(agent_req)
#     main_result = yunyou_agent.run(agent_req)
#     for msg in main_result:
#         print(msg)
#     # print(main_result["messages"][-1].content)
