import logging
import re
import json
import uuid
from typing import TypedDict, Annotated

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import add_messages

from agent.agent_builder import create_tool_agent_executor
from agent.base import BaseAgent
from agent.graph_state import AgentRequest
from agent.tools.yunyou_tools import holter_list, holter_type_count, holter_report_count
from utils.file_utils import FileUtils

logger = logging.getLogger(__name__)

"""
定义和创建云柚相关子图（Yunyou Agent）。
该 Agent 继承自 BaseAgent，只专注于定义自己的核心逻辑。
"""


class YunyouState(TypedDict):
    messages: Annotated[list, add_messages]


def _parse_yunyou_model_response(message: AIMessage) -> AIMessage:
    """针对 gpt-oss 等特定模型非标准工具调用输出的定制化解析器。"""
    content = message.content
    match = re.search(r"to=(?P<tool_name>\w+).*?<\|message\|>(?P<args_json>.*?)<\|call\|>", content)

    if not match:
        return message

    tool_name = match.group("tool_name")
    args_json_str = match.group("args_json")

    try:
        args = json.loads(args_json_str)
        tool_call = {
            "name": tool_name,
            "args": args,
            "id": f"call_{uuid.uuid4()}"
        }
        return AIMessage(
            content="",
            tool_calls=[tool_call],
            id=message.id,
            usage_metadata=message.usage_metadata,
            response_metadata=message.response_metadata
        )
    except (json.JSONDecodeError, AttributeError):
        return message


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
        cue_word = FileUtils.read_project_file("app/agent/template/yunyou_agent_cue_word.txt")
        if cue_word is None:
            raise FileNotFoundError("无法读取云柚代理提示词文件")
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""{cue_word}"""),
            MessagesPlaceholder(variable_name="messages")
        ])

        chain = prompt | self.model
        tools = [holter_list, holter_type_count, holter_report_count]
        self.model.bind_tools(tools)

        # 使用统一的构建器，并传入定制化的响应解析器
        graph = create_tool_agent_executor(
            chain=chain,
            tools=tools,
            state_class=YunyouState,
            response_parser=_parse_yunyou_model_response
        )
        return graph
