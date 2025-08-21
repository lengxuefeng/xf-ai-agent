import io

from IPython.core.display_functions import display
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.constants import END
from langgraph.graph import StateGraph

from agent.agents.code_agent import create_code_agent
from agent.agents.medical_agent import create_medical_agent
from agent.agents.search_agent import create_search_agent
from agent.agents.sql_agent import create_sql_agent
from agent.agents.weather_agent import create_weather_agent
from agent.graphs.state import GraphState
from agent.llm.ollama_model import load_ollama_model
from IPython.display import Image as IPyImage  # 重命名为 IPyImage
from PIL import Image as PILImage  # 重命名为 PILImage，避免冲突
"""
定义和构建多智能体协作的核心 - 主管图（Supervisor Graph）。

该文件包含以下关键部分：
    1.模型和工具的初始化：加载所需的 LLM 和外部工具（如天气查询、网络搜索）。
    2.子图（Agents）的创建：为每个特定任务（如医疗、编码、SQL）创建独立的 LangGraph 实例。
    3.主管（Supervisor）的构建：
        -   定义一个特殊的主管链（Supervisor Chain），它能根据用户输入和对话历史，
            决定将任务路由到哪个子图，或者直接结束对话。
        -   将所有子图和主管本身注册为图中的节点。
        -   定义图的边（Edges），即节点之间的转换逻辑。
    4.人工断点（Human-in-the-loop）的实现：在特定节点（如代码生成、SQL 执行）后设置中断，
        等待用户确认。
    5.图的编译：将定义好的节点和边编译成一个可执行的 `StateGraph`。
"""

# 我们系统中所有可用的专家智能体（子图）
# "FINISH" 是一个特殊的成员，表示对话可以结束
MEMBERS = ["medical_agent", "code_agent", "sql_agent", "weather_agent", "search_agent", "FINISH"]

agent_runnables = {
    "medical_agent": create_medical_agent,
    "code_agent": create_code_agent,
    "sql_agent": create_sql_agent,
    "weather_agent": create_weather_agent,
    "search_agent": create_search_agent,
}

supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个多智能体协作系统的主管。你的任务是分析用户的请求，并将其分派给最合适的专家智能体。"
            "你有以下专家智能体可供选择：{members}。"
            "请根据用户的最新消息，选择一个最合适的智能体来处理请求。"
            "如果你认为当前任务已完成或用户在表示感谢，请选择 'FINISH' 来结束对话。"
            "你的回答必须只能是下一个要调用的智能体的名称，不能包含任何其他文本或解释。"
        ),
        # MessagesPlaceholder 是一个占位符，后续会用实际的对话历史替换它
        MessagesPlaceholder(variable_name="messages"),
    ]
)

model = load_ollama_model("qwen3:8b")

# 创建主图链
supervisor_chain = (
        supervisor_prompt.partial(members=", ".join(MEMBERS))
        | model
        # 从模型的输出中提取文本内容
        | (lambda x: x.content)
)


# 定义图的节点和边
def _supervisor_node(state: GraphState):
    """
    定义主管节点（Supervisor Node）。

    主管节点负责接收当前的对话状态，调用主管链（supervisor_chain）
    来决定下一个应该由哪个智能体接手，或者是否应该结束对话。

    Args:
        state (GraphState): 当前图的状态，包含对话历史等信息。

    Returns:
        dict: 一个字典，包含下一个节点的名称，用于更新图的状态。
    """
    next_agent_response = supervisor_chain.invoke({"messages": state["messages"]})
    # 从模型可能返回的复杂文本中，解析出最后一个词作为下一个智能体的名称
    # 这样做是为了增加系统的鲁棒性，防止模型输出额外的思考过程，
    next_agent = next_agent_response.strip().split('\n')[-1].replace("'", "").replace('"', "")

    # 如果解析出的名称无效或不在预设的成员列表中，则默认为结束

    if not next_agent or next_agent not in MEMBERS:
        next_agent = "FINISH"

    # 返回一个包含下一个节点名称的字典，LangGraph 会用它来更新 'next' 字段的状态
    return {"next": next_agent}


def _agent_node(state: GraphState, agent_name: str):
    """
    定义一个通用的智能体节点（Agent Node）。

    这个节点负责执行由主管指定的具体子图（智能体）。它会根据传入的 agent_name，
    获取对应的子图执行器 (runnable)，并调用它来处理任务。

    Args:
        state (GraphState): 当前图的状态。
        agent_name (str): 要执行的智能体的名称。

    Returns:
        dict: 一个字典，包含子图生成的新消息或中断信息，用于更新图的状态。
    """

    agent_runnable = agent_runnables[agent_name](model)
    result = agent_runnable.invoke({"messages": state["messages"]})

    # 如果子图返回了中断信息，直接将完整的中断状态向上传递
    if "interrupt" in result and result["interrupt"]:
        return result

    # 否则，只提取由该子图新生成的消息，以避免状态污染
    new_messages = result["messages"][len(result["messages"])]
    return {"messages": new_messages}


def _router_node(state: GraphState):
    """
    定义路由器（Router）。

    这是一个条件边的逻辑函数，用于在图的每次迭代后决定下一步的走向。
    它检查图的状态，判断是应该回到主管进行下一轮决策，还是结束流程，或者暂停以待人工干预。

    Args:
        state (GraphState): 当前图的状态。

    Returns:
        str: 下一个节点的名称（"supervisor", "FINISH", 或 "__interrupt__"）。
    """

    # 检查状态中是否有中断信号
    if state.get("interrupt"):
        return "__interrupt__"
    # 检查是否需要结束对话
    if state["next"] == "FINISH":
        return "FINISH"
    else:
        return "supervisor"


def create_graph():
    """
    创建并编译整个多智能体图。

    这个函数负责组装之前定义的所有节点和边，构建一个完整的、可执行的状态图。

    Returns:
        CompiledGraph: 一个编译好的、可执行的 LangGraph 实例。
    """

    workflow = StateGraph(GraphState)

    workflow.add_node("supervisor", _supervisor_node)

    # 遍历所有成员，为每个智能体添加一个节点
    for member in MEMBERS:
        if member != "FINISH":  # FINISH 不是一个可执行的节点
            workflow.add_node(member, lambda state, agent_name=member: _agent_node(state, agent_name))

    workflow.set_entry_point("supervisor")

    # 这个字典告诉图，当主管节点的输出（state["next"]）是某个智能体的名字时，应该跳转到对应的节点
    conditional_map = {member: member for member in MEMBERS if member != "FINISH"}
    # 如果输出是 "FINISH"，则跳转到图的终点 (END)
    conditional_map["FINISH"] = END

    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        conditional_map,
    )
    # 为每个智能体节点添加一条回到 "supervisor" 节点的边
    # 这就形成了一个工作循环：主管 -> 子图 -> 主管 -> ...
    for member in MEMBERS:
        if member != "FINISH":
            workflow.add_edge(member, "supervisor")

    # 编译图
    return workflow.compile()


if __name__ == '__main__':
    graph = create_graph()
    # 处理图片
    img_bytes = graph.get_graph(xray=True).draw_mermaid_png()
    pil_img = PILImage.open(io.BytesIO(img_bytes))
    img_name = "graph.png"
    pil_img.save(img_name)
    # 显示图片
    display(IPyImage(img_name))
    result = graph.invoke({"messages": [HumanMessage(content="杞县今天的天气")]})

    print(result["messages"][-1].content)

