# -*- coding: utf-8 -*-
import logging
from abc import ABC, abstractmethod
from typing import Generator

from langchain_core.messages import HumanMessage

from agent.graph_state import AgentRequest
from utils import redis_manager
from utils.custom_logger import LogTarget, get_logger

log = get_logger(__name__)

"""
定义 Agent 的抽象基类
"""


class BaseAgent(ABC):
    """
    所有 Agent 的抽象基类，封装了通用的运行和状态管理逻辑。

    子类需要实现 `_create_graph` 方法来定义自己的核心图逻辑。
    通过继承该基类，可以避免在每个具体 Agent 中重复编写 Redis 交互等样板代码。
    """

    def __init__(self, req: AgentRequest):
        """
        初始化基类，设置通用属性。
        """
        if not req.model:
            log.error(f"{self.__class__.__name__} 模型初始化失败，模型对象为空")
            raise ValueError(f"{self.__class__.__name__} 模型初始化失败，请检查配置。")

        self.model = req.model
        self.session_id = req.session_id
        self.subgraph_id = req.subgraph_id
        self.redis_manager = redis_manager.RedisManager()

        # 子类必须在自己的 __init__ 方法中调用 _create_graph 来实例化 self.graph
        self.graph = self._create_graph()
        if not self.graph:
            raise NotImplementedError("子类必须实现 _create_graph 并实例化 self.graph")

    @abstractmethod
    def _create_graph(self):
        """
        抽象方法，子类必须实现该方法来创建并返回其具体的 LangGraph 执行器。
        """
        pass

    def run(self, req: AgentRequest) -> Generator:
        """
        执行 Agent 的主流程，包含状态加载、运行图、状态保存。
        这是一个通用方法，所有子类共享此逻辑。
        """
        # 1. 从 Redis 加载当前会话的历史状态（使用try-except避免Redis故障阻塞）
        try:
            state = self.redis_manager.load_graph_state(self.session_id, subgraph_id=self.subgraph_id)
        except Exception as e:
            log.warning(f"加载状态失败，使用空状态: {e}")
            state = None

        if not state:
            state = {"messages": []}

        # 2. 避免重复追加相同的用户输入
        # 获取最后一条人类消息
        last_human_msg = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_human_msg = msg
                break

        # 如果最后一条人类消息不存在或内容不同，则添加新的用户输入
        # 注意：如果是 resume 模式，不需要添加新消息，下面的逻辑会处理
        
        # 3. 检查恢复指令 & Checkpoint 加载
        resume_command = None
        
        # 加载 Checkpoint
        checkpoint_loaded = False
        if hasattr(self, 'checkpointer') and self.checkpointer:
            self.redis_manager.load_checkpoint(self.checkpointer, self.session_id)
            if self.session_id in self.checkpointer.storage:
                checkpoint_loaded = True
            
            # 检查是否有待恢复的审批
            from services.interrupt_service import interrupt_service
            try:
                from langgraph.types import Command
                
                if self.session_id in interrupt_service.pending_approvals:
                    approvals = interrupt_service.pending_approvals[self.session_id]
                    # 只有当 Checkpoint 成功加载时才尝试 Resume
                    if checkpoint_loaded:
                        for msg_id, approval in list(approvals.items()):
                            if approval['status'] in ['approve', 'reject']:
                                resume_command = Command(resume=approval['status'])
                                log.info(f"检测到审批结果，准备恢复执行: {approval['status']}", target=LogTarget.ALL)
                                del approvals[msg_id] # 消费掉
                                break
                    else:
                        # 如果有 Approval 但没有 Checkpoint，清除 Approval 以避免死循环（或者保留让用户重新触发？）
                        # 最好是打印警告，并不使用 resume_command，这样会退化为重新执行。
                        log.warning("检测到 Approval 但 Checkpoint 丢失（可能是上次保存失败），无法 Resume，将重新开始执行。", target=LogTarget.ALL)
                        
            except ImportError:
                log.warning("当前环境不支持 langgraph.types.Command，无法自动恢复中断")

        # 只有在非 resume 模式下才添加用户消息
        if not resume_command:
            if not last_human_msg or last_human_msg.content != req.user_input:
                state["messages"].append(HumanMessage(content=req.user_input))

        # 4. 以流式方式调用图执行器，并收集模型回答
        final_state = state.copy()
        config = {"configurable": {"thread_id": self.session_id}}
        
        try:
            # 确保graph不为None
            if self.graph and hasattr(self.graph, 'stream'):
                # 构造参数
                stream_args = {}
                if resume_command:
                    stream_args["input"] = None
                    stream_args["command"] = resume_command
                else:
                    stream_args["input"] = final_state
                
                stream_args["config"] = config

                # 调用 stream
                # 注意：如果 stream 不支持 command 参数，这里会报错。
                # 但既然用了新版 LangGraph 特性，我们假设它支持。
                # 如果是旧版，可能需要 stream_args["input"] = resume_command
                
                try:
                    iterator = self.graph.stream(**stream_args)
                except TypeError as e:
                    if "unexpected keyword argument 'command'" in str(e) and resume_command:
                        # 降级尝试：将 Command 作为 input
                        log.info("降级尝试：将 Command 作为 input 传入")
                        stream_args.pop("command")
                        stream_args["input"] = resume_command
                        iterator = self.graph.stream(**stream_args)
                    else:
                        raise e

                for event in iterator:
                    log.debug(f"处理事件: {event}")
                    # 处理不同格式的事件
                    if isinstance(event, dict):
                        # 直接包含messages的事件
                        if "messages" in event:
                            final_state["messages"].extend(event["messages"])
                        # 节点格式的事件
                        else:
                            for node_name, node_output in event.items():
                                if isinstance(node_output, dict) and "messages" in node_output:
                                    final_state["messages"].extend(node_output["messages"])
                    yield event
            else:
                log.error(f"图执行器不存在或没有stream方法: {self.__class__.__name__}")
                yield {}
        finally:
            # 保存 Checkpoint
            if hasattr(self, 'checkpointer') and self.checkpointer:
                self.redis_manager.save_checkpoint(self.checkpointer, self.session_id)

        # 5. 实现短记忆：限制 messages 列表长度
        max_messages = 10
        if len(final_state["messages"]) > max_messages:
            final_state["messages"] = final_state["messages"][-max_messages:]

        # 5. 将执行后的新状态保存回 Redis（使用try-except避免Redis故障阻塞）
        if final_state:
            try:
                self.redis_manager.save_graph_state(final_state, self.session_id, self.subgraph_id)
            except Exception as e:
                log.warning(f"保存状态失败: {e}")
