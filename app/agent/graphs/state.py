import operator
from typing import TypedDict, Annotated, List, Optional, Dict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

"""
【模块说明】
定义主图 (Supervisor) 的状态流转机 (StateGraph)。
State 是 LangGraph 的灵魂，所有的节点函数都只是在读取和修改这个字典。

【三层决策架构】
Tier-0: Pre-Graph Rule Engine (图外部拦截)
Tier-1: Intent Router (小模型路由)
Tier-2: Parent Planner DAG (复杂任务拆解 → Dispatcher ⟲ Worker → Reducer → Aggregator)
"""


class SubTask(TypedDict):
    """单个子任务描述（用于 Parent_Planner 的 DAG 拆解）"""
    id: str                         # 子任务唯一标识 (如 "t1", "t2")
    agent: str                      # 分配的 Agent 名称 (如 "weather_agent", "CHAT")
    input: str                      # 子任务的完整输入描述 (自包含，无代词)
    depends_on: List[str]           # 依赖的其他子任务 ID 列表 (空 = 可并行)
    status: str                     # pending / dispatched / done / error
    result: Optional[str]           # 执行结果

class WorkerResult(TypedDict):
    """单个 Worker 执行返回的结果封装（支持 operator.add 并发收集）"""
    task_id: str
    result: str
    error: Optional[str]

class GraphState(TypedDict):
    """
    定义图的全局状态。

    Attributes:
        messages: 核心对话历史
        session_id: 当前对话的唯一标识
        llm_config: 大模型的配置

        ---- 数据域路由 (Domain Router) 输出 ----
        data_domain: 数据域 (YUNYOU_DB / LOCAL_DB / WEB_SEARCH / GENERAL)
        domain_confidence: 数据域识别置信度 0~1
        domain_route_source: 识别来源 (rule / llm / history)

        ---- 意图识别 (Intent Router) 输出 ----
        intent: 识别到的意图/Agent 名称
        intent_confidence: 置信度 0~1
        is_complex: 是否为复杂请求
        direct_answer: LLM 直接给出的回答

        ---- 任务规划 (Parent Planner) 输出 ----
        task_list: 子任务列表（DAG 描述）
        task_results: 子任务执行结果映射 {task_id: result}
        current_wave: 当前 Dispatcher 调度波次
        max_waves: 最大调度波次（安全阀）

        ---- 并发执行 (Dispatcher & Worker) ----
        active_tasks: 当前准备 Send 派发执行的 SubTask 列表
        worker_results: 并发 Worker 执行后收集的增量结果
        
        ---- 路由元数据 ----
        current_node: 当前执行到的节点名称
        next: 路由指示器
    """
    messages: Annotated[List[BaseMessage], add_messages]
    session_id: Optional[str]
    llm_config: Optional[dict]

    # 数据域路由输出
    data_domain: Optional[str]
    domain_confidence: Optional[float]
    domain_route_source: Optional[str]

    # 意图识别输出
    intent: Optional[str]
    intent_confidence: Optional[float]
    is_complex: Optional[bool]
    direct_answer: Optional[str]

    # 任务规划输出 (Tier-2 DAG)
    task_list: Optional[List[SubTask]]
    task_results: Optional[Dict[str, str]]
    current_wave: Optional[int]
    max_waves: Optional[int]

    # 并发执行输出 (Map-Reduce)
    active_tasks: Optional[List[SubTask]]
    worker_results: Annotated[List[WorkerResult], operator.add]

    # 路由元数据
    current_node: Optional[str]
    next: Optional[str]
    interrupt_payload: Optional[Dict]
