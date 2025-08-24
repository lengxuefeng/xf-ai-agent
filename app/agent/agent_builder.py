# -*- coding: utf-8 -*-
"""
通用 Agent 构建器

该模块提供了一系列工厂函数，用于封装和简化 LangGraph 中不同类型 Agent（子图）的创建过程。
通过将通用的图构建逻辑（如节点定义、边的连接、编译）抽象出来，使得每个 Agent 的定义可以更专注于其核心业务逻辑。

核心功能：
- create_tool_agent_executor: 创建一个具备工具使用能力的 Agent 执行器。
- create_interruptable_agent_executor: 创建一个支持人工中断和反馈的 Agent 执行器。
- create_simple_agent_executor: 创建一个简单的、无工具的 Agent 执行器。
"""
from typing import List, Type, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


def create_tool_agent_executor(model: BaseChatModel, tools: List, state_class: Type[dict]):
    """
    工厂函数：创建一个具备工具使用能力的 Agent 执行器（编译后的图）。

    该函数封装了标准的 "Agent -> Tool -> Agent" 循环。

    工作流程：
    1. Agent 节点 (agent_node): 调用 LLM 决定是直接回答还是使用工具。
    2. 条件路由 (router): 检查 LLM 的输出，如果包含工具调用，则路由到工具节点；否则结束。
    3. 工具节点 (ToolNode): 执行具体的工具调用。
    4. 工具节点执行完毕后，将结果返回给 Agent 节点，由其整合信息并生成最终回复。

    Args:
        model (BaseChatModel): 用于驱动 Agent 的语言模型。
        tools (List): Agent 可用的工具列表。
        state_class (Type[dict]): 图的状态类 (TypedDict)，必须包含 'messages' 字段。

    Returns:
        A compiled LangGraph executor.
    """

    # 1. 定义 Agent 节点
    def agent_node(state):
        """
        定义智能体节点（Agent Node）。

        这个节点是决策点。它调用模型，模型会根据对话历史决定：
        1.  直接生成一个回答（如果信息足够）。
        2.  生成一个工具调用请求（如果需要搜索）。
        3.  在工具执行后，根据工具返回的结果生成最终回答。
        """
        # 检查最后一条消息是否是工具消息
        if isinstance(state["messages"][-1], ToolMessage):
            # 如果是，直接调用模型生成回复
            response = model.invoke(state["messages"])
        else:
            # 否则，将工具绑定到模型，让模型决定是否调用工具
            model_with_tools = model.bind_tools(tools)
            response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # 2. 定义路由逻辑
    def router(state):
        """
        定义路由函数，在 agent 节点执行后决定下一步走向。
        """
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            # 如果有工具调用，则走向 "tools" 节点
            return "tools"
        else:
            # 否则，流程结束
            return END

    # 3. 创建图
    graph = StateGraph(state_class)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    # 4. 设置入口点和边
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", router)
    graph.add_edge("tools", "agent")

    # 5. 编译图并返回
    return graph.compile()


def create_interruptable_agent_executor(
        state_class: Type[dict],
        agent_node: Callable,
        interrupt_node: Callable,
        router_node: Callable
):
    """
    工厂函数：创建一个支持人工中断的 Agent 执行器（编译后的图）。

    该函数封装了 "Agent -> Interrupt -> (User Feedback) -> Agent/End" 的循环。

    工作流程：
    1. Agent 节点: 执行核心任务，如生成代码或 SQL。
    2. Interrupt 节点: 准备中断信息，暂停图的执行，等待用户输入。
    3. 条件路由 (router_node): 在中断后，根据用户的最新反馈决定是回到 Agent 节点重试，还是结束流程。

    Args:
        state_class (Type[dict]): 图的状态类 (TypedDict)。
        agent_node (Callable): Agent 节点的执行函数。
        interrupt_node (Callable): 中断节点的执行函数。
        router_node (Callable): 中断后的路由逻辑函数。

    Returns:
        A compiled LangGraph executor with interruption.
    """
    graph = StateGraph(state_class)
    graph.add_node("agent", agent_node)
    graph.add_node("interrupt", interrupt_node)

    graph.set_entry_point("agent")
    graph.add_edge("agent", "interrupt")
    graph.add_conditional_edges("interrupt", router_node)

    # 编译图，并指定在 "interrupt" 节点后暂停
    return graph.compile(interrupt_after=["interrupt"])


def create_simple_agent_executor(
        model: BaseChatModel,
        prompt: ChatPromptTemplate,
        state_class: Type[dict],
        post_process_func: Callable = None
):
    """
    工厂函数：创建一个简单的、无工具的 Agent 执行器。

    该函数封装了 "Prompt -> LLM -> (Post-processing)" 的线性流程。

    Args:
        model (BaseChatModel): 用于驱动 Agent 的语言模型。
        prompt (ChatPromptTemplate): Agent 的提示模板。
        state_class (Type[dict]): 图的状态类。
        post_process_func (Callable, optional): 在 LLM 调用后对响应进行后处理的函数。

    Returns:
        A compiled LangGraph executor.
    """
    chain = prompt | model

    def agent_node(state):
        response = chain.invoke({"messages": state["messages"]})
        if post_process_func:
            response = post_process_func(response)
        return {"messages": [response]}

    graph = StateGraph(state_class)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    return graph.compile()
