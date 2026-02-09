import functools
import logging

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph

from agent.graph_state import AgentRequest
from agent.graphs.state import GraphState
from agent.registry import agent_classes, MEMBERS
from agent.llm.unified_loader import create_model_from_config
log = logging.getLogger(__name__)

"""
定义和构建多智能体协作的核心 - 主管图（Supervisor Graph）。
"""

# Supervisor 提示模板
supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个多智能体协作系统的主管。你的任务是分析用户的请求，并将其分派给最合适的专家智能体。"
            "你有以下专家智能体可供选择：{members}。"
            "请根据用户的最新消息，选择一个最合适的智能体来处理请求。"
            "如果用户的请求是简单的问候、闲聊或者询问你是谁，或者你无法判断需要哪个专家时，请回答 'CHAT'。"
            "如果你认为当前任务已完成或用户在表示感谢，请回答 'FINISH' 来结束对话。"
            "你的回答必须只能是下一个要调用的智能体的名称，例如 'search_agent', 'CHAT' 或 'FINISH'，不能包含任何其他文本或解释。"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# 解析 LLM 回复
def parse_next_agent(llm_response: str) -> str:
    if not llm_response:
        return "CHAT"
    llm_response = llm_response.strip()
    if "CHAT" in llm_response:
        return "CHAT"
    if "FINISH" in llm_response:
        return "FINISH"
    for member in MEMBERS:
        if member in llm_response:
            return member
    return "CHAT"


# Supervisor 节点
def _supervisor_node(state: GraphState, model):
    user_input = ""
    try:
        # 安全地获取消息列表
        if isinstance(state, dict):
            messages = list(state.get("messages", [])) # 复制列表以防修改影响原状态
        else:
            messages = list(getattr(state, "messages", []))

        # 预处理：如果最后一条是空的 AIMessage (通常是中断产生的副作用，或者 GraphRunner 未能正确添加恢复标记)，
        # 则暂时忽略它，以便 Supervisor 能看到上一条 HumanMessage 并正确路由
        if messages and isinstance(messages[-1], AIMessage) and not messages[-1].content and not messages[-1].tool_calls:
            log.info("检测到末尾存在空 AIMessage，可能是中断残留，暂时忽略以进行意图识别")
            messages.pop()

        log.info(f"消息列表长度: {len(messages)}")
        for i, msg in enumerate(messages):
            log.info(
                f"消息 {i}: 类型={type(msg)}, 内容={getattr(msg, 'content', 'N/A')[:100] if hasattr(msg, 'content') else 'N/A'}, name={getattr(msg, 'name', 'N/A')}")

        # 检查是否是与特定Agent的连续对话
        log.info(
            f"检查连续对话条件: len(messages)={len(messages)}, messages[-1]类型={type(messages[-1]) if len(messages) > 0 else 'N/A'}, messages[-2]类型={type(messages[-2]) if len(messages) > 1 else 'N/A'}")
        if len(messages) > 1 and isinstance(messages[-1], HumanMessage) and isinstance(messages[-2], AIMessage):
            last_agent_name = getattr(messages[-2], 'name', None)
            log.info(f"检查连续对话: last_agent_name={last_agent_name}")
            # 如果上一条消息来自yunyou_agent，则继续保持在yunyou_agent
            if last_agent_name and last_agent_name == "yunyou_agent":
                log.info(f"Continuing conversation with agent: {last_agent_name}")
                return {"next": last_agent_name}

            # 其他Agent的连续对话逻辑
            if last_agent_name and last_agent_name != "ChatAgent" and last_agent_name in MEMBERS:
                # 如果用户正在回复特定Agent，保持在同一Agent，除非用户明确切换
                if hasattr(messages[-1], 'content'):
                    user_input_lower = messages[-1].content.lower()
                else:
                    user_input_lower = ""
                log.info(f"检查其他Agent的连续对话: user_input_lower={user_input_lower}")
                switch_agent = False
                for name, info in agent_classes.items():
                    if name != last_agent_name and hasattr(info, 'keywords') and any(
                            kw.lower() in user_input_lower for kw in info.keywords):
                        switch_agent = True
                        break
                if not switch_agent:
                    log.info(f"Continuing conversation with agent: {last_agent_name}")
                    return {"next": last_agent_name}
        else:
            log.info("未满足连续对话条件")
            if len(messages) > 1:
                log.info(f"messages[-1]是HumanMessage: {isinstance(messages[-1], HumanMessage)}")
                log.info(f"messages[-2]是AIMessage: {isinstance(messages[-2], AIMessage)}")
                if len(messages) > 2:
                    log.info(f"messages[-3]类型: {type(messages[-3])}")
                    log.info(
                        f"messages[-3]内容: {getattr(messages[-3], 'content', 'N/A')[:100] if hasattr(messages[-3], 'content') else 'N/A'}")
                    log.info(f"messages[-3]名称: {getattr(messages[-3], 'name', 'N/A')}")

        if messages and isinstance(messages[-1], HumanMessage):
            user_input = messages[-1].content.lower() if hasattr(messages[-1], 'content') else ""
            log.info(f"用户输入: {user_input}")

        # 检查是否是简短回复（如"确定"、"是的"等），如果是，则保持当前智能体
        if user_input and len(user_input) <= 5 and any(
                kw in user_input for kw in ["确定", "是的", "好的", "ok", "yes"]):
            # 检查历史对话中最近的智能体调用
            for i in range(len(messages) - 2, -1, -1):
                msg = messages[i]
                if isinstance(msg, AIMessage):
                    # 检查工具调用
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        # 通过工具调用判断智能体类型
                        for tool_call in msg.tool_calls:
                            if hasattr(tool_call, "name"):
                                tool_name = getattr(tool_call, 'name', '').lower()
                                for agent_name in agent_classes.keys():
                                    if agent_name.lower() in tool_name:
                                        log.info(f"检测到简短确认回复，继续使用智能体: {agent_name}")
                                        return {"next": agent_name}
                            elif isinstance(tool_call, dict) and 'name' in tool_call:
                                tool_name = tool_call['name'].lower() if hasattr(tool_call['name'], 'lower') else str(
                                    tool_call['name'])
                                for agent_name in agent_classes.keys():
                                    if agent_name.lower() in tool_name:
                                        log.info(f"检测到简短确认回复，继续使用智能体: {agent_name}")
                                        return {"next": agent_name}
                    # 检查消息名称
                    elif hasattr(msg, "name") and msg.name and msg.name != "ChatAgent":
                        for agent_name in agent_classes.keys():
                            if agent_name in str(msg.name).lower():
                                log.info(f"检测到简短确认回复，继续使用智能体: {agent_name}")
                                return {"next": agent_name}

        # 关键词匹配
        for name, info in agent_classes.items():
            if hasattr(info, 'keywords') and any(kw.lower() in user_input for kw in info.keywords):
                log.info(f"关键词匹配成功: {name}")
                return {"next": name}

        # 如果关键词匹配失败，则直接进入聊天模式
        log.info("未匹配到任何Agent关键词，进入CHAT模式")
        return {"next": "CHAT"}
    except Exception as e:
        log.error(f"主管节点执行失败: {str(e)}")
        return {"next": "CHAT"}


# 通用 Agent 节点
def _agent_node(state: GraphState, agent_name: str, model, description: str = ""):
    user_input = ""
    if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
        user_input = state["messages"][-1].content if isinstance(state["messages"][-1].content, str) else ""

    llm_config = state.get("llm_config", {})

    req = AgentRequest(
        user_input=user_input if isinstance(user_input, str) else "",
        model=model,
        session_id=state.get("session_id", "") or "",
        subgraph_id=agent_name,
        llm_config=llm_config
    )

    agent_info = agent_classes[agent_name]
    agent_instance = agent_info.cls(req)

    # 运行子图并从事件流中寻找最终的回复
    final_answer = None
    for event in agent_instance.run(req):
        # 检查是否是新的中断格式
        if isinstance(event, dict) and "__interrupt__" in event:
            log.warning(f"Supervisor 检测到子图中断: {event['__interrupt__']}")
            # 返回包含特殊中断标记的状态，这将传递给 graph_runner
            return {"interrupt": event["__interrupt__"]}

        # event 的结构是 {'node_name': {'messages': [...]}}
        for node_name, node_output in event.items():
            # 检查旧的中断格式
            if isinstance(node_output, dict) and "interrupt" in node_output and node_output["interrupt"]:
                log.warning(f"Supervisor 检测到子图中断(旧格式): {node_output['interrupt']}")
                return {"interrupt": node_output["interrupt"]}

            if isinstance(node_output, dict) and "messages" in node_output:
                messages = node_output.get('messages', [])
                if messages:
                    # 获取当前节点产生的最新消息
                    last_message = messages[-1]
                    # 一个“最终答案”是一个没有工具调用的 AIMessage
                    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                        final_answer = last_message

    # 在遍历完所有子图事件后，使用找到的最终答案
    if final_answer:
        try:
            # 确保返回的 AIMessage 包含 agent_name，以便 supervisor 进行上下文判断
            if not getattr(final_answer, 'name', None):
                final_answer = AIMessage(
                    content=final_answer.content,
                    name=agent_name,
                    id=getattr(final_answer, 'id', None),
                    additional_kwargs=getattr(final_answer, 'additional_kwargs', {}),
                    response_metadata=getattr(final_answer, 'response_metadata', {}),
                    tool_calls=getattr(final_answer, 'tool_calls', []),
                    usage_metadata=getattr(final_answer, 'usage_metadata', None),
                    invalid_tool_calls=getattr(final_answer, 'invalid_tool_calls', [])
                )
        except Exception as e:
            log.error(f"创建带名称的AIMessage失败: {e}, 回退到简单版本")
            final_answer = AIMessage(content=final_answer.content, name=agent_name,
                                     id=getattr(final_answer, 'id', None))
        return {"messages": [final_answer]}

    # 如果遍历完整个子图的运行过程都没有找到一个明确的最终答案
    return {"messages": [AIMessage(content=f"抱歉，{agent_name} 在处理过程中未能生成明确的回复。", name=agent_name)]}


# 通用 Chat 节点
def _chat_node(state: GraphState, model):
    user_message = state["messages"][-1].content
    prompt = "你是一个名为 “XF AI Agent” 的人工智能助手。请直接、友好地回答用户的问题。"
    
    # 构造消息列表
    messages = [
        ("system", prompt),
        ("human", user_message)
    ]
    
    # 直接调用模型生成回复
    response = model.invoke(messages)
    response_content = response.content
    
    ai_message = AIMessage(content=response_content, name="ChatAgent")
    return {"messages": [ai_message]}


#  创建 Graph
def create_graph(model_config: dict | None = None):
    """
    创建主图，支持动态模型配置
    
    Args:
        model_config: 模型配置字典，包含模型相关参数
    """
    # 默认模型配置
    default_config = {
        'model': 'google/gemini-1.5-pro',
        'model_service': 'netlify-gemini',
        'deep_thinking_mode': 'auto',
        'rag_enabled': False,
        'similarity_threshold': 0.7,
        'embedding_model': 'bge-m3:latest'
    }

    # 合并配置
    final_config = {**default_config, **(model_config or {})}

    try:
        # 根据配置加载模型
        model, embedding_model = create_model_from_config(**final_config)
        log.info(f"✅ 模型加载成功: {final_config['model_service']} - {final_config['model']}")

    except Exception as e:
        error_msg = f"加载模型失败 - 服务: {final_config.get('model_service', 'unknown')}, 模型: {final_config.get('model', 'unknown')}, 错误: {str(e)}"
        log.error(f"❌ {error_msg}")
        # 不使用默认模型，直接抛出异常让上层处理
        raise RuntimeError(error_msg) from e

    workflow = StateGraph(GraphState)

    # 注册节点
    workflow.add_node("supervisor", functools.partial(_supervisor_node, model=model))
    workflow.add_node("chat_node", functools.partial(_chat_node, model=model))

    for name, info in agent_classes.items():
        node_fn = functools.partial(_agent_node, agent_name=name, model=model, description=info.description)
        workflow.add_node(name, node_fn)

    workflow.set_entry_point("supervisor")

    # 条件分支
    conditional_map: dict[str | None, str] = {name: name for name in MEMBERS}
    conditional_map["CHAT"] = "chat_node"
    conditional_map["FINISH"] = END

    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next", "CHAT"),
        conditional_map,
    )

    # 专家节点执行完直接结束
    for name in MEMBERS:
        workflow.add_edge(name, END)
    workflow.add_edge("chat_node", END)

    return workflow.compile()
