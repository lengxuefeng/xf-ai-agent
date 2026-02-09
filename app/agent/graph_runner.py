# -*- coding: utf-8 -*-
import json
import logging
import re
import time
from typing import Generator, List

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from agent.graphs.supervisor import create_graph

from utils.custom_logger import get_logger, LogTarget, CustomLogger
from services.interrupt_service import interrupt_service

log = get_logger(__name__)

# 创建一个全局的日志输出队列（用于在流式输出中显示日志）
from queue import Queue

log_output_queue = Queue()


# 设置日志回调函数
def log_callback(message: str):
    """日志输出回调函数"""
    print(f"[log_callback] 收到日志: {message.strip()}")
    log_output_queue.put(message)


# 初始化时设置全局回调（所有 logger 实例都会使用）
CustomLogger.add_global_sse_callback(log_callback)
print(f"[graph_runner] 全局日志回调已设置: {log_callback}")


def to_sse(data: dict) -> str:
    """将字典格式化为 Server-Sent Event (SSE) 字符串。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def stream_text(text: str, delay: float = 0.0) -> Generator[str, None, None]:
    """
    将文本逐字符流式输出，实现打字机效果。
    自动识别 <think> 标签内容为思考过程，其他为正常回复。

    Args:
        text: 要输出的文本
        delay: 每个字符之间的延迟（秒），默认0.0以获得最快响应速度

    Yields:
        str: SSE格式的流式文本块
    """
    if not text:
        return

    # 解析文本，分离 thinking 和正常内容
    import re

    # 查找所有 <think> 块
    think_pattern = r'<think>(.*?)</think>'
    parts = re.split(think_pattern, text, flags=re.DOTALL)

    for i, part in enumerate(parts):
        if not part.strip():  # 跳过空白部分
            continue

        # 奇数索引是 <think> 标签内的内容（思考过程）
        is_thinking = (i % 2 == 1)
        content_type = "thinking" if is_thinking else "stream"

        # 逐字符流式输出
        for j, char in enumerate(part):
            # 如果是中文字符、标点或空格，可以稍微快一点
            if ord(char) > 127 or char in '。！？，；：\n ':
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
        self.last_config_hash = None

    def stream_run(self, user_input: str, session_id: str, model_config: dict = None, history_messages: list = []) -> \
    Generator[str, None, None]:
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

        # 计算配置hash，只在配置改变时才重新创建图
        import hashlib
        import json
        config_str = json.dumps(final_config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()

        # 根据配置创建或更新图
        if self.graph is None or self.last_config_hash != config_hash:
            try:
                log.info(f"开始创建图，配置: {final_config.get('model', 'unknown')}", target=LogTarget.LOG)
                self.graph = create_graph(final_config)
                self.last_config_hash = config_hash
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
        
        # 仅当不是恢复请求时才添加用户消息
        if user_input and user_input != "[RESUME]":
            messages.append(HumanMessage(content=user_input))
        elif user_input == "[RESUME]":
            log.info("接收到恢复执行请求 [RESUME]，跳过添加消息，将触发 Supervisor 重新路由")

        initial_state = {
            "messages": messages,
            "session_id": session_id,
            "llm_config": final_config  # 将模型配置传入状态
        }

        # 此列表用于从所有事件中累积消息，以解决状态在 END 节点丢失的问题
        accumulated_messages: List[BaseMessage] = []
        added_message_ids = set()

        try:
            # 首先输出队列中已存在的日志
            while not log_output_queue.empty():
                try:
                    log_message = log_output_queue.get_nowait()
                    yield log_message
                except:
                    break

            # 使用 stream + subgraphs（LangGraph 的标准方式）
            for event in self.graph.stream(initial_state, subgraphs=True):
                # 检查并输出日志队列中的消息
                while not log_output_queue.empty():
                    try:
                        log_message = log_output_queue.get_nowait()
                        yield log_message
                    except:
                        break
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

                # 3. 检查是否有中断请求（支持 __interrupt__ 和 interrupt 键）
                interrupt_source = None

                # 情况 1: 直接在 state 中 (LangGraph 最新格式)
                if "__interrupt__" in state:
                    interrupt_source = state["__interrupt__"]

                # 情况 2: 在节点输出中 (Supervisor 包装或旧格式)
                if not interrupt_source:
                    for key, value in state.items():
                        if isinstance(value, dict):
                            if "__interrupt__" in value:
                                interrupt_source = value["__interrupt__"]
                                break
                            elif "interrupt" in value and value["interrupt"]:
                                interrupt_source = value["interrupt"]
                                break

                if interrupt_source:
                    interrupt_value = interrupt_source

                    # 处理元组包装 (Interrupt(...),) - 某些版本的 LangGraph 可能会这样返回
                    if isinstance(interrupt_source, tuple) and len(interrupt_source) > 0:
                        interrupt_value = interrupt_source[0]

                    # 如果是 Interrupt 对象，提取 value
                    if hasattr(interrupt_value, 'value'):
                        interrupt_value = interrupt_value.value

                    # 获取 action_requests (兼容对象属性和字典键)
                    action_requests = []
                    if hasattr(interrupt_value, 'action_requests'):
                        action_requests = interrupt_value.action_requests
                    elif isinstance(interrupt_value, dict) and 'action_requests' in interrupt_value:
                        action_requests = interrupt_value['action_requests']

                    # 记录详细的工具调用信息到日志
                    if action_requests:
                        log.warning(f"收到中断请求，需要人工审核以下操作:", target=LogTarget.ALL)

                        # 注册待审核请求
                        for action in action_requests:
                            # 兼容对象和字典
                            action_name = getattr(action, 'name', None) or action.get('name', 'unknown') if isinstance(action, dict) else getattr(action, 'name', 'unknown')
                            action_args = getattr(action, 'args', None) or action.get('args', {}) if isinstance(action, dict) else getattr(action, 'args', {})
                            action_desc = getattr(action, 'description', None) or action.get('description', '') if isinstance(action, dict) else getattr(action, 'description', '')

                            # 记录日志
                            import json
                            log.warning(f"  - 工具: {action_name}", target=LogTarget.ALL)
                            args_str = json.dumps(action_args, ensure_ascii=False)
                            log.warning(f"    参数: {args_str}", target=LogTarget.ALL)
                            if action_desc:
                                log.warning(f"    描述: {action_desc}", target=LogTarget.ALL)

                            # 注册待审核请求
                            interrupt_service.register_pending_approval(
                                session_id=session_id,
                                message_id=f"{session_id}_{action_name}",
                                action_name=action_name,
                                action_args=action_args,
                                description=action_desc
                            )

                    # 提取 action_requests 并格式化
                    if action_requests:
                        actions_info = []
                        for action in action_requests:
                            # 兼容对象和字典
                            action_name = getattr(action, 'name', None) or action.get('name', 'unknown') if isinstance(action, dict) else getattr(action, 'name', 'unknown')
                            action_args = getattr(action, 'args', None) or action.get('args', {}) if isinstance(action, dict) else getattr(action, 'args', {})
                            import json
                            args_str = json.dumps(action_args, ensure_ascii=False)
                            actions_info.append(f"工具: {action_name}, 参数: {args_str}")

                        interrupt_message = f"\n\n🛡️ **需要人工审核**\n\n{chr(10).join([f'  • {info}' for info in actions_info])}\n\n请审核是否允许执行这些操作。"
                    else:
                        interrupt_message = str(interrupt_value)

                    # 发送中断事件
                    yield to_sse({"type": "interrupt", "content": interrupt_message})

                    # 发送流结束信号，标记当前回复结束
                    yield to_sse({
                        "type": "response_end",
                        "content": "",
                        "name": "interrupted"
                    })
                    return

            # 4. 流结束，从累积的消息列表中提取最终结果
            if accumulated_messages:
                final_message = accumulated_messages[-1]
                if isinstance(final_message, AIMessage):
                    print(f"[graph_runner] 开始输出AI回复，长度: {len(final_message.content)}")
                    # 发送流开始信号
                    yield to_sse({"type": "response_start", "content": ""})

                    # 逐字符流式输出 AI 的回复（无延迟）
                    for i, char in enumerate(final_message.content):
                        yield to_sse({
                            "type": "stream",
                            "content": char
                        })
                        # 每100个字符打印一次调试信息
                        if (i + 1) % 100 == 0:
                            print(f"[graph_runner] 已输出 {i + 1}/{len(final_message.content)} 个字符")

                    print(f"[graph_runner] AI回复输出完成")

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
