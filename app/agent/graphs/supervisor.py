import functools
from typing import Dict, List

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

from agent.agents.code_agent import CodeAgent
from agent.agents.medical_agent import MedicalAgent
from agent.agents.search_agent import SearchAgent
from agent.agents.sql_agent import SqlAgent
from agent.agents.weather_agent import WeatherAgent
from agent.graph_state import AgentRequest
from agent.graphs.state import GraphState
from agent.llm.loader_llm_multi import load_silicon_flow
from agent.llm.ollama_model import load_ollama_model

"""
定义和构建多智能体协作的核心 - 主管图（Supervisor Graph）。
"""


# 假设你已经有这些类
# from your_agents import MedicalAgent, CodeAgent, SqlAgent, WeatherAgent, SearchAgent
# from your_llm_loader import load_ollama_model
# from your_request_model import AgentRequest
# END 是 workflow 的结束节点常量

# -------------------------
# 1️⃣ 定义 AgentInfo
class AgentInfo:
    def __init__(self, cls, description: str, keywords: List[str]):
        self.cls = cls
        self.description = description
        self.keywords = keywords


# 2️⃣ 定义系统中所有 Agent
agent_classes: Dict[str, AgentInfo] = {
    "medical_agent": AgentInfo(
        cls=MedicalAgent,
        description="回答医疗健康问题，例如症状、药物咨询，输出附带免责声明",
        keywords=["病", "症状", "药", "健康", "血压", "心率"]
    ),
    "code_agent": AgentInfo(
        cls=CodeAgent,
        description="提供编程代码生成、分析、优化服务",
        keywords=["代码", "函数", "bug", "报错", "SQL", "Python", "Java"]
    ),
    "sql_agent": AgentInfo(
        cls=SqlAgent,
        description="处理数据库查询和 SQL 优化问题",
        keywords=["SQL", "数据库", "查询", "表", "索引"]
    ),
    "weather_agent": AgentInfo(
        cls=WeatherAgent,
        description="提供天气查询和预报服务",
        keywords=["天气", "气温", "预报", "降雨"]
    ),
    "search_agent": AgentInfo(
        cls=SearchAgent,
        description="处理通用搜索和信息检索",
        keywords=["搜索", "查找", "信息", "资料"]
    ),
}

MEMBERS = list(agent_classes.keys())

# -------------------------
# 3️⃣ Supervisor 提示模板
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


# -------------------------
# 4️⃣ 解析 LLM 回复
def parse_next_agent(llm_response: str) -> str:
    llm_response = llm_response.strip()

    if "CHAT" in llm_response:
        return "CHAT"
    if "FINISH" in llm_response:
        return "FINISH"

    for member in MEMBERS:
        if member in llm_response:
            return member

    return "CHAT"


# -------------------------
# 5️⃣ Supervisor 节点
def _supervisor_node(state: GraphState, model):
    user_input = ""
    if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
        user_input = state["messages"][-1].content.lower()

    # 关键词匹配
    for name, info in agent_classes.items():
        if any(kw.lower() in user_input for kw in info.keywords):
            return {"next": name}

    # LLM fallback
    member_descriptions = ", ".join(f"{name}: {info.description}" for name, info in agent_classes.items())
    supervisor_chain = (
            supervisor_prompt.partial(members=member_descriptions)
            | model
            | (lambda x: x.content)
    )
    next_agent_response = supervisor_chain.invoke({"messages": state["messages"]})
    next_agent = parse_next_agent(next_agent_response)

    if next_agent not in MEMBERS + ["CHAT", "FINISH"]:
        next_agent = "CHAT"

    return {"next": next_agent}


# -------------------------
# 6️⃣ 通用 Agent 节点
def _agent_node(state: GraphState, agent_name: str, model, description: str = ""):
    user_input = ""
    if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
        user_input = state["messages"][-1].content

    # 获取模型配置，如果没有则使用默认配置
    llm_config = state.get("llm_config", {})
    
    req = AgentRequest(
        user_input=user_input,
        model=model,
        session_id=state.get("session_id", ""),
        subgraph_id=agent_name,
        llm_config=llm_config
    )

    agent_info = agent_classes[agent_name]
    agent_instance = agent_info.cls(req)

    # 收集所有消息
    all_messages = []
    final_subgraph_state = None
    
    for event in agent_instance.run(req):
        final_subgraph_state = event
        # 从事件中提取消息
        for node_name, node_output in event.items():
            if isinstance(node_output, dict) and "messages" in node_output:
                all_messages.extend(node_output["messages"])

    # 如果有累积的消息，使用它们；否则检查最终状态
    if all_messages:
        return {"messages": all_messages}
    
    # 备用方案：从最终状态中获取消息
    if final_subgraph_state:
        # 尝试从不同可能的结构中获取消息
        for key, value in final_subgraph_state.items():
            if isinstance(value, dict) and "messages" in value:
                return {"messages": value["messages"]}
    
    return {"messages": []}


# -------------------------
# 7️⃣ 通用 Chat 节点
def _chat_node(state: GraphState, model):
    user_message = state["messages"][-1].content
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个名为 “XF AI Agent” 的人工智能助手。请直接、友好地回答用户的问题。"),
        ("human", "{question}")
    ])
    chain = prompt | model | (lambda x: x.content)
    response_content = chain.invoke({"question": user_message})
    ai_message = AIMessage(content=response_content, name="ChatAgent")
    return {"messages": [ai_message]}


# -------------------------
# 8️⃣ 创建 Graph
def create_graph(model_config: dict = None):
    """
    创建主图，支持动态模型配置
    
    Args:
        model_config: 模型配置字典，包含模型相关参数
    """
    # 导入统一模型加载器
    from agent.llm.unified_loader import create_model_from_config
    
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
        print(f"✅ 模型加载成功: {final_config['model_service']} - {final_config['model']}")
    except Exception as e:
        print(f"⚠️ 模型加载失败，使用默认模型: {e}")
        # 回退到默认模型
        model = load_silicon_flow("Qwen/QwQ-32B")
        embedding_model = None
    
    workflow = StateGraph(GraphState)

    # 注册节点
    workflow.add_node("supervisor", functools.partial(_supervisor_node, model=model))
    workflow.add_node("chat_node", functools.partial(_chat_node, model=model))

    for name, info in agent_classes.items():
        node_fn = functools.partial(_agent_node, agent_name=name, model=model, description=info.description)
        workflow.add_node(name, node_fn)

    workflow.set_entry_point("supervisor")

    # 条件分支
    conditional_map = {name: name for name in MEMBERS}
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



