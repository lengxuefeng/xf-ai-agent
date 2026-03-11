# -*- coding: utf-8 -*-
"""
LangChain Stream Events 集成，用于捕获所有中间事件
"""
import json
import re
from typing import Generator, List

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from agent.graphs.supervisor import create_graph
from constants.sse_constants import SseEventType


def to_sse(data: dict) -> str:
    """将字典格式化为 Server-Sent Event (SSE) 字符串。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def stream_text(text: str, delay: float = 0.0) -> Generator[str, None, None]:
    """
    将文本逐字符流式输出，实现打字机效果。
    自动识别 </think> 标签内容为思考过程，其他为正常回复。

    Args:
        text: 要输出的文本
        delay: 每个字符之间的延迟（秒），默认0.0以获得最快响应速度

    Yields:
        str: SSE格式的流式文本块
    """
    if not text:
        return

    # 解析文本，分离 thinking 和正常内容
    # 查找所有 </think> 块
    think_pattern = r'</think>(.*?)</think>'
    parts = re.split(think_pattern, text, flags=re.DOTALL)

    for i, part in enumerate(parts):
        if not part.strip():  # 跳过空白部分
            continue

        # 奇数索引是 </think> 标签内的内容（思考过程）
        is_thinking = (i % 2 == 1)
        content_type = SseEventType.THINKING.value if is_thinking else SseEventType.STREAM.value

        # 逐字符流式输出
        for j, char in enumerate(part):
            yield to_sse({
                "type": content_type,
                "content": char
            })


class GraphRunnerWithEvents:
    """
    图运行器，支持使用 astream_events 捕获所有中间事件。
    """

    def __init__(self, model_config: dict = None, enable_events: bool = True):
        """
        初始化图运行器

        Args:
            model_config: 模型配置字典，包含模型相关参数
            enable_events: 是否启用事件输出
        """
        self.model_config = model_config or {}
        self.graph = None
        self.enable_events = enable_events

    async def astream_run(
            self, user_input: str, session_id: str, model_config: dict = None, history_messages: list = []
    ) -> Generator[str, None, None]:
        """
        使用 astream_events 执行图，捕获所有中间事件。

        Args:
            user_input: 用户输入
            session_id: 会话 ID
            model_config: 模型配置字典
            history_messages: 历史消息列表
        """
        # 合并模型配置
        final_config = {**self.model_config, **(model_config or {})}

        # 创建图
        if self.graph is None or model_config:
            self.graph = create_graph(final_config)

        # 构建消息历史
        messages: List[BaseMessage] = []
        for msg in history_messages:
            if msg.get('user_content') is not None:
                messages.append(HumanMessage(content=msg['user_content']))
            if msg.get('model_content') is not None:
                messages.append(AIMessage(content=msg['model_content'], name=msg.get('name')))
        messages.append(HumanMessage(content=user_input))

        initial_state = {
            "messages": messages,
            "session_id": session_id,
            "llm_config": final_config
        }

        try:
            # 使用 astream_events 捕获所有事件
            async for event in self.graph.astream_events(initial_state, version="v1"):
                if not self.enable_events:
                    continue

                event_type = event["event"]

                # 1. LLM 相关事件
                if "on_chat_model_start" in event_type:
                    model_name = event.get("data", {}).get("input", {}).get("model", "unknown")
                    yield to_sse({
                        "type": SseEventType.LOG.value,
                        "log_type": "info",
                        "logger": "llm",
                        "message": f"🤖 调用模型: {model_name}"
                    })

                elif "on_chat_model_stream" in event_type:
                    content = event.get("data", {}).get("chunk", {}).get("content", "")
                    if content:
                        # 流式输出
                        yield to_sse({
                            "type": SseEventType.STREAM.value,
                            "content": content
                        })

                elif "on_chat_model_end" in event_type:
                    # 检查响应中的 reasoning_content（某些模型支持）
                    response = event.get("data", {}).get("output", {})
                    if hasattr(response, 'response_metadata'):
                        metadata = response.response_metadata
                        if metadata:
                            # 检查是否有 reasoning_tokens
                            usage = metadata.get('token_usage', {})
                            reasoning_tokens = usage.get('reasoning_tokens', 0)
                            if reasoning_tokens > 0:
                                yield to_sse({
                                    "type": SseEventType.THINKING.value,
                                    "content": f"💭 思考了 {reasoning_tokens} 个 token"
                                })

                # 2. Chain 相关事件
                elif "on_chain_start" in event_type:
                    chain_name = event.get("name", "unknown")
                    yield to_sse({
                        "type": SseEventType.LOG.value,
                        "log_type": "info",
                        "logger": "chain",
                        "message": f"🔗 执行 Chain: {chain_name}"
                    })

                elif "on_chain_end" in event_type:
                    chain_name = event.get("name", "unknown")
                    output = event.get("data", {}).get("output", {})
                    yield to_sse({
                        "type": SseEventType.LOG.value,
                        "log_type": "info",
                        "logger": "chain",
                        "message": f"✅ Chain 完成: {chain_name}"
                    })

                # 3. Tool 相关事件
                elif "on_tool_start" in event_type:
                    tool_name = event.get("name", "unknown")
                    yield to_sse({
                        "type": SseEventType.LOG.value,
                        "log_type": "info",
                        "logger": "tool",
                        "message": f"🛠️ 调用工具: {tool_name}"
                    })

                elif "on_tool_end" in event_type:
                    tool_name = event.get("name", "unknown")
                    output = event.get("data", {}).get("output", {})
                    yield to_sse({
                        "type": SseEventType.LOG.value,
                        "log_type": "info",
                        "logger": "tool",
                        "message": f"✅ 工具完成: {tool_name}"
                    })

                # 4. Agent 相关事件
                elif "on_agent_action" in event_type:
                    action = event.get("data", {}).get("action", {})
                    tool = action.get("tool", "unknown")
                    yield to_sse({
                        "type": SseEventType.THINKING.value,
                        "content": f"🤔 Agent 决定使用工具: {tool}"
                    })

        except Exception as e:
            yield to_sse({
                "type": SseEventType.ERROR.value,
                "content": f"执行错误: {str(e)}"
            })
