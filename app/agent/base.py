from abc import ABC, abstractmethod
from typing import Generator, Any, Dict, Optional
from langchain_core.runnables import Runnable, RunnableConfig
from agent.graph_state import AgentRequest
from agent.graphs.checkpointer import checkpointer
from utils.custom_logger import get_logger

log = get_logger(__name__)

"""
【模块说明】
所有具体业务智能体（如 YunyouAgent）的抽象基类。
负责统一处理请求封装、Checkpointer 绑定以及流式输出逻辑。
"""


class BaseAgent(ABC):
    def __init__(self, req: AgentRequest):
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
        统一执行入口。
        这里接收外部传入的 config，并与其进行合并。
        这是保证 LangSmith Trace 连续性，以及 LangGraph 状态隔离的核心。
        """
        # 1. 继承父级 config（包含 LangSmith 的 tags, callbacks 等）
        base_config = config or {}
        configurable = base_config.get("configurable", {})

        # 2. 注入当前子 Agent 的持久化标识
        configurable["thread_id"] = self.session_id
        configurable["checkpoint_ns"] = self.subgraph_id  # 命名空间隔离，防止不同子图状态冲突

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

    def get_state(self):
        """提供给外部查询当前子图运行状态（如是否处于 Interrupt 挂起）的方法"""
        config = {
            "configurable": {
                "thread_id": self.session_id,
                "checkpoint_ns": self.subgraph_id
            }
        }
        return self.graph.get_state(config)