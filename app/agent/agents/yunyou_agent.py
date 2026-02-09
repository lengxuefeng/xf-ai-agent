import os

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from agent.base import BaseAgent
from agent.graph_state import AgentRequest
from agent.tools.yunyou_tools import holter_list, holter_type_count, holter_report_count
from schemas.base_skill import ConfigurableSkillMiddleware
from utils.custom_logger import get_logger, LogTarget
from utils.file_utils import FileUtils

log = get_logger(__name__)
# logger = logging.getLogger(__name__)

"""
定义和创建云柚相关子图（Yunyou Agent）。
该 Agent 继承自 BaseAgent，只专注于定义自己的核心逻辑。
"""


class YunyouAgent(BaseAgent):
    """
    云柚专家 Agent，继承自 BaseAgent。
    """

    def __init__(self, req: AgentRequest):
        # 首先调用父类的 __init__，它会处理通用逻辑并调用下面的 _create_graph
        super().__init__(req)

    def _create_graph(self):
        """
        实现父类的抽象方法，创建并返回该 Agent 专属的图执行器。
        """
        # 初始化持久化 Checkpointer (BaseAgent 会负责持久化到 Redis)
        self.checkpointer = InMemorySaver()

        system_prompt = FileUtils.read_project_file(os.getenv("PROMPT_TEMPLATE_YUNYOU"))
        if system_prompt is None:
            raise FileNotFoundError("无法读取云柚代理提示词文件")
        tools = [holter_list, holter_type_count, holter_report_count]
        log.info(f"✅ 云柚代理工具加载成功: {len(tools)} 个工具", target=LogTarget.ALL)
        graph = create_agent(
            model=self.model,
            tools=tools,
            system_prompt=system_prompt,
            checkpointer=self.checkpointer,
            middleware=[ConfigurableSkillMiddleware(os.getenv("SKILL_YUNYOU")),
                        # 开启摘要功能
                        # SummarizationMiddleware(
                        #     model=self.model,
                        #     trigger=[
                        #         ("token", 3000)
                        #     ],
                        #     keep=("messages", 10)
                        # ),
                        # 开启人类交互中间件
                        HumanInTheLoopMiddleware(
                            interrupt_on={
                                "holter_report_count": {
                                    "allowed_decisions": ["approve", "reject"],
                                },
                                "holter_list": False,
                                "holter_type_count": False
                            },
                        )
                        ],

        )
        return graph
