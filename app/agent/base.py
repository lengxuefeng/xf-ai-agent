from abc import ABC, abstractmethod
from typing import Generator, Any, Dict, Optional
from langchain_core.runnables import Runnable, RunnableConfig
from agent.graph_state import AgentRequest
from agent.graphs.checkpointer import checkpointer
from utils.custom_logger import get_logger

log = get_logger(__name__)

"""
所有具体业务智能体（如 YunyouAgent）的抽象基类。

本模块提供了核心的 `BaseAgent` 类，负责：
1. 统一处理来自用户的 AgentRequest 请求封装。
2. 绑定和管理由于多轮对话产生的 Checkpointer 状态持久化。
3. 统一规范流式输出与图执行的生命周期隔离机制。
"""


class BaseAgent(ABC):
    def __init__(self, req: AgentRequest):
        """
        初始化 Agent 实例。

        Args:
            req (AgentRequest): 包含对话上下文和模型配置的请求对象。
        """
        self.req = req
        self.session_id = req.session_id
        # 如果未提供子图标识，默认使用类名
        self.subgraph_id = getattr(req, "subgraph_id", None) or self.__class__.__name__
        self.checkpointer = checkpointer

    @abstractmethod
    def _build_graph(self) -> Runnable:
        """子类必须实现此方法，返回编译后的 StateGraph"""
        pass

    def run(self, req: AgentRequest, config: Optional[RunnableConfig] = None) -> Generator[Dict[str, Any], None, None]:
        """
        统一的图执行入口。

        接收外部传入的 config 并进行安全拷贝与合并，以保证 LangSmith Trace 的上下文连续性。
        同时为 LangGraph 配置持久化 ID，确保不同的业务子图在相同 Session 下状态有效隔离。

        Args:
            req (AgentRequest): 包含对话上下文的请求对象。
            config (Optional[RunnableConfig]): 由上游传递的 LangGraph 配置项。

        Yields:
            Generator[Dict[str, Any], None, None]: 流程图中产生的状态更新事件流或错误信息。
        """
        # 1. 继承父级 config（包含 LangSmith 的 tags, callbacks 等）
        base_config = config or {}
        # 安全复制，避免修改上游的 config 指针产生副作用
        configurable = base_config.get("configurable", {}).copy()

        # 2. 注入当前子 Agent 的持久化标识
        # 采用联合 thread_id 以在同一对话会话中隔离不同子图的状态，避免直接使用 checkpoint_ns 报错
        configurable["thread_id"] = f"{self.session_id}_{self.subgraph_id}"

        final_config = {**base_config, "configurable": configurable}

        # 3. 构造初始状态
        input_message = {"messages": [("human", req.user_input)]}

        try:
            # 仅流式传输更新，中断状态(Interrupt)交由外部 GraphRunner 统一扫描
            for event in self.graph.stream(input_message, config=final_config, stream_mode="updates"):
                yield event
        except Exception as e:
            log.error(f"Agent {self.subgraph_id} 运行出错: {str(e)}")
            yield {"error": str(e)}

    def get_state(self) -> Any:
        """
        查询当前子图运行状态。

        提供给外部系统查询智能体当前的图状态（例如是否处于 Interrupt 中断挂起等待审核等）。

        Returns:
            Any: 当前图状态对象。
        """
        config = {
            "configurable": {
                "thread_id": f"{self.session_id}_{self.subgraph_id}"
            }
        }
        return self.graph.get_state(config)