# -*- coding: utf-8 -*-
import json
import logging
import re
import time
from typing import Generator, List

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from agent.graphs.supervisor import create_graph

from utils.custom_logger import get_logger, LogTarget

log = get_logger(__name__)


def to_sse(data: dict) -> str:
    """将字典格式化为 Server-Sent Event (SSE) 字符串。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def stream_text(text: str, delay: float = 0.03) -> Generator[str, None, None]:
    """
    将文本逐字符流式输出，实现打字机效果。
    自动识别 </think> 标签内容为思考过程，其他为正常回复。
    
    Args:
        text: 要输出的文本
        delay: 每个字符之间的延迟（秒）
    
    Yields:
        str: SSE格式的流式文本块
    """
    if not text:
        return

    # 解析文本，分离 thinking 和正常内容
    import re
    
    # 查找所有 </think> 块
    think_pattern = r'</think>(.*?)</think>'
    parts = re.split(think_pattern, text, flags=re.DOTALL)

    for i, part in enumerate(parts):
        if not part.strip():  # 跳过空白部分
            continue

        # 奇数索引是 </think> 标签内的内容（思考过程）
        is_thinking = (i % 2 == 1)
        content_type = "thinking" if is_thinking else "stream"

        # 逐字符流式输出
        for j, char in enumerate(part):
            # 如果是中文字符、标点或空格，可以稍微快一点
            if ord(char) > 127 or char in '。！？，；：\\n ':
                current_delay = delay * 0.5
            else:
                current_delay = delay

            yield to_sse({
                "type": content_type,
                "content": char
            })

            if current_delay > 0:
                time.sleep(current_delay)


class GraphRunner:
    """
    图运行器，负责执行 Agent Supervisor 并处理流式输出。
    """

    def __init__(self, model_config: dict = None):
        """
        初始化图运行器

        Args:
            model_config: 模型配置字典，包含模型相关参数
        """
        self.model_config = model_config or {}
        self.graph = None

    def stream_run(self, user_input: str, session_id: str, model_config: dict = None, history_messages: list = []) -> Generator[str, None, None]:
        """
        以流式方式执行主图，并产出格式化的 SSE 事件。

        Args:
            user_input: 用户输入
            session_id: 会话 ID
            model_config: 模型配置字典
            history_messages: 历史消息列表
        """
        # 合并模型配置
        final_config = {**self.model_config, **(model_config or {})}

        # 根据配置创建或更新图
        if self.graph is None or model_config:
            try:
                log.info(f"开始创建图，配置: {final_config.get('model', 'unknown')}", target=LogTarget.LOG)
                self.graph = create_graph(final_config)
            except RuntimeError as e:
                log.error(f"模型加载失败: {e}", target=LogTarget.ALL)
                # 模型加载失败，直接抛出异常
                yield to_sse({"type": "error", "content": f"模型加载失败: {str(e)}"})
                return

        # 构建消息历史
        messages: List[BaseMessage] = []
        for msg in history_messages:
            if msg.get('user_content') is not None:
                messages.append(HumanMessage(content=msg['user_content']))
            if msg.get('model_content') is not None:
                # 修复：在重建 AIMessage 时，包含其 'name' 属性，以维持上下文
                messages.append(AIMessage(content=msg['model_content'], name=msg.get('name')))
        messages.append(HumanMessage(content=user_input))

        initial_state = {
            "messages": messages,
            "session_id": session_id,
            "llm_config": final_config  # 将模型配置传入状态
        }

        # 此列表用于从所有事件中累积消息，以解决状态在 END 节点丢失的问题
        accumulated_messages: List[BaseMessage] = []
        added_message_ids = set()

        try:
            # 使用 stream + subgraphs（LangGraph 的标准方式）
            for event in self.graph.stream(initial_state, subgraphs=True):
                # 打印每个事件用于调试（元组结构：(路径, 状态））
                print(f"--- EVENT ---\n{event}\n--- END EVENT ---")

                # 解析元组：提取路径和状态字典（关键修复点）
                path, state = event  # 解包 (路径元组, 状态字典)

                # 1. 处理思考过程（使用解析后的state字典）
                if "supervisor" in state:
                    next_agent = state["supervisor"].get("next")
                    log.info(f"Supervisor 决策: {next_agent or 'FINISH'}", target=LogTarget.ALL)
                    if next_agent and next_agent != "FINISH":
                        yield to_sse({"type": "thinking", "content": f"🔀 正在路由到: {next_agent}..."})
                    else:
                        if next_agent == "FINISH":
                            log.info("任务已完成", target=LogTarget.ALL)
                            yield to_sse({"type": "thinking", "content": "✅ 任务已完成"})

                # 2. 从每个事件中提取新消息并累积
                # state 的格式是 {"node_name": {"messages": [...]}}
                for node_name, node_output in state.items():  # 这里用state而不是event
                    if isinstance(node_output, dict) and "messages" in node_output:
                        unique_messages = [msg for msg in node_output["messages"] if msg.id not in added_message_ids]
                        if unique_messages:
                            accumulated_messages.extend(unique_messages)
                            for msg in unique_messages:
                                added_message_ids.add(msg.id)

                # 3. 检查是否有中断请求
                for key, value in state.items():  # 这里用state而不是event
                    if isinstance(value, dict) and "interrupt" in value and value["interrupt"]:
                        log.warning(f"收到中断请求: {value['interrupt']}", target=LogTarget.ALL)
                        yield to_sse({"type": "interrupt", "content": value["interrupt"]})
                        return

            # 4. 流结束，从累积的消息列表中提取最终结果
            if accumulated_messages:
                final_message = accumulated_messages[-1]
                if isinstance(final_message, AIMessage):
                    # 发送流开始信号
                    yield to_sse({"type": "response_start", "content": ""})

                    # 逐字符流式输出 AI 的回复
                    for chunk in stream_text(final_message.content):
                        yield chunk

                    # 发送流结束信号，并附上 agent name
                    yield to_sse({
                        "type": "response_end",
                        "content": "",
                        "name": getattr(final_message, 'name', None)
                    })
                else:
                    # 如果最后一条消息不是AI发出的，说明流程异常
                    yield to_sse({"type": "error", "content": "任务流程异常，AI未生成最终回复。"})
            else:
                # 如果整个过程都没有任何消息，返回错误
                yield to_sse({"type": "error", "content": "任务执行完毕，但未能获取到任何消息。"})

        except Exception as e:
            log.exception(f"Graph execution error")
            yield to_sse({"type": "error", "content": f"执行过程中发生错误: {str(e)}"})

