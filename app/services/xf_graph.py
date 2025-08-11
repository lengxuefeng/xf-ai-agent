import json
import logging
from typing import Generator

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.constants import END
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.llm.loader_llm_multi import load_chat_model, load_open_router, load_zhipu_model, load_tongyi_model
from app.llm.ollama_model import load_ollama_model
from schemas.chat_schemas import ChatRequest
from schemas.graph_state import GraphState
from app.utils.config import Settings
from tools.search_tools import get_weather
from utils.agent_utils import AgentUtils

logger = logging.getLogger(__name__)


class XfGraph:
    """
    Xf智能体类，用于根据用户问题调用不同工具来获取信息。
    """

    def __init__(self, model_name: str = "qwen3:8b", model_type: str = "ollama"):
        """
        初始化智能体。

        Args:
            model_name: 模型名称，默认使用 "qwen3:8b"。
            model_type: 模型类型，默认使用 "ollama"。
        """
        self.model_name = model_name
        self.model_type = model_type
        self.llm = self._load_llm()
        self.tools = self._load_tools()
        self.graph = self._create_graph()

    def _load_tools(self):
        tools = [get_weather]
        return ToolNode(tools=tools)

    def _create_graph(self) -> CompiledStateGraph:
        """
        创建并配置状态图工作流

        Returns:
            CompiledStateGraph: 编译后的状态图实例。
        """

        # 创建流程图构建器
        graph = StateGraph(GraphState)

        # 添加 LLM 节点
        graph.add_node("llm", self._call_model)

        # 添加 工具 节点
        graph.add_node("tools", self.tools)

        # 设置起始节点为 llm
        graph.set_entry_point("llm")

        # 添加边判断是否需要工具调用(使用 tools_condition 自动判断)
        graph.add_conditional_edges(
            "llm",
            tools_condition,
            {
                "tools": "tools",  # 若需要调用工具则跳转至tools工具节点
                END: END,  # 否则结束
            }
        )

        # 添加边，工具执行完后，继续回到 llm
        graph.add_edge("tools", "llm")

        # 编译图
        return graph.compile()

    def _load_llm(self):
        """
        加载指定类型的LLM模型

        Returns:
            加载的LLM模型实例
        """
        logger.info(f"加载模型: {self.model_name}, 类型: {self.model_type}")
        # 检查模型类型是否支持
        if self.model_type not in Settings.SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"不支持的模型类型: {self.model_type}。支持的模型类型有: {', '.join(Settings.SUPPORTED_MODEL_TYPES)}")

        # 获取对应类型的加载函数
        if self.model_type == "ollama":
            return load_ollama_model(self.model_name)
        elif self.model_type == "openRouter":
            return load_open_router(self.model_name)
        elif self.model_type == "chat":
            return load_chat_model(self.model_name)
        elif self.model_type == "zhipu":
            return load_zhipu_model(self.model_name)
        elif self.model_type == "qwen":
            return load_tongyi_model(self.model_name)
        else:
            raise ValueError(
                f"不支持的模型类型: {self.model_type}。支持的模型类型有: {', '.join(Settings.SUPPORTED_MODEL_TYPES)}")

    def _call_model(self, state: GraphState) -> dict:
        """
        调用模型并返回结果

        Args:
            state: 图状态实例，包含用户问题和其他上下文信息

        Returns:
            模型生成的回复内容
        """
        messages = state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def invoke_agent(self, message: str) -> str:
        """
        执行整个 LangGraph 流程，从用户输入到最终回复。

        Args:
            message: 用户输入的消息内容

        Returns:
            模型生成的回复内容
        """
        if not self.llm:
            return "对不起，代理当前不可用。"
        initial_state = {"messages": [HumanMessage(content=message)]}
        final_state = self.graph.ainvoke(initial_state)
        final_answer = final_state["messages"][-1]
        if isinstance(final_answer, AIMessage):
            return final_answer.content

        return "对不起，我无法处理您的问题。"

    def stream_agent_handle(self, req: ChatRequest) -> Generator[str, None, None]:
        """
        处理聊天请求，执行智能体流程并返回流式响应。
        """
        self.model_name = req.settings.modelName
        self.model_type = req.settings.modelType
        self.llm = self._load_llm()
        return self._stream_agent(req.message)

    def _stream_agent(self, message: str) -> Generator[str, None, None]:
        """
        执行整个 LangGraph 流程，从用户输入到最终回复，支持流式输出。
        思考阶段使用thinking类型输出，最终回复使用response类型输出。
        """
        if not self.llm:
            yield AgentUtils.to_sse({"type": "error", "content": "对不起，当前代理不可用，请稍后重试。"})
            return

        has_valid_response = False
        initial_state = {"messages": [HumanMessage(content=message)]}

        try:
            # 使用LangGraph的stream方法获取流式响应
            # 注意：stream方法返回的是一个迭代器，不是协程
            current_content = ""
            thinking_content = ""

            # 迭代处理每个响应块
            # 显式声明stream返回的是可迭代对象，而不是协程
            # 使用await将协程转换为结果，避免协程类型错误
            # 由于self.graph.stream返回的是协程类型，我们需要使用异步迭代器
            # 但由于当前函数是同步生成器，我们需要将协程转换为结果
            # 这里使用list()将迭代器转换为列表，避免协程类型错误
            response_chunks = list(self.graph.stream(initial_state))
            for chunk in response_chunks:
                # 检查chunk是否包含llm键，以及llm是否包含messages键
                llm_data = chunk.get("llm", {})
                if not isinstance(llm_data, dict):
                    continue

                messages = llm_data.get("messages", [])
                if not isinstance(messages, list) or not messages:
                    continue

                for msg in messages:
                    if isinstance(msg, AIMessage):
                        has_valid_response = True
                        # 获取消息内容，确保类型安全
                        content = msg.content

                        # 处理内容格式，确保content_str是字符串
                        if isinstance(content, (dict, list)):
                            # 如果内容是字典或列表，转换为JSON字符串
                            content_str = json.dumps(content, ensure_ascii=False)
                        elif content is None:
                            # 如果内容为None，使用空字符串
                            content_str = ""
                        else:
                            # 其他情况，强制转换为字符串
                            content_str = str(content)

                        # 计算增量内容
                        if len(content_str) > len(current_content):
                            # 只发送新增的部分
                            delta = content_str[len(current_content):]
                            current_content = content_str

                            # 在思考阶段过滤掉<think>和</think>标签
                            clean_delta = delta
                            if "<think>" in clean_delta:
                                # 如果增量内容包含思考开始标记，则移除
                                clean_delta = clean_delta.replace("<think>", "")
                            if "</think>" in clean_delta:
                                # 如果增量内容包含思考结束标记，则移除
                                clean_delta = clean_delta.replace("</think>", "")

                            # 如果清理后的增量内容不为空，则输出
                            if clean_delta.strip():
                                # 实时流式输出增量内容
                                # ✅ 打字机效果：逐字符返回
                                for char in clean_delta:
                                    yield AgentUtils.to_sse({"type": "thinking", "content": char})
                                    # yield AgentUtils.to_sse({"type": "thinking", "content": clean_delta})
                            thinking_content = current_content

            # 思考完成后，处理最终响应
            if has_valid_response and thinking_content:
                # 从思考内容中提取最终回复，移除思考过程
                final_response = thinking_content

                # 检查是否包含思考过程标记
                if "<think>" in final_response and "</think>" in final_response:
                    # 提取</think>标签后的内容作为最终回复
                    try:
                        final_response = final_response.split("</think>", 1)[1].strip()
                    except IndexError:
                        # 如果分割失败，则使用原始内容
                        pass
                elif "<think>" in final_response:
                    # 如果只有开始标签，尝试移除整个思考部分
                    try:
                        final_response = final_response.split("<think>", 1)[0].strip()
                        if not final_response:  # 如果为空，可能思考内容在前面
                            final_response = thinking_content.replace("<think>", "").strip()
                    except IndexError:
                        # 如果分割失败，则移除标签
                        final_response = thinking_content.replace("<think>", "").strip()

                # yield AgentUtils.to_sse({"type": "response", "content": final_response})
                for char in final_response:
                    yield AgentUtils.to_sse({"type": "response", "content": char})
            elif not has_valid_response:
                yield AgentUtils.to_sse({"type": "error", "content": "对不起，我无法处理您的问题。"})

        except Exception as e:
            logger.error(f"流式处理出错: {str(e)}")
            yield AgentUtils.to_sse({"type": "error", "content": "对不起，处理过程中发生错误，请重试。"})


# 创建单例代理实例
xf_graph = XfGraph()
