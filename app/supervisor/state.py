import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

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
    status: str                     # pending / dispatched / pending_approval / done / error / cancelled
    result: Optional[str]           # 执行结果

class WorkerResult(TypedDict):
    """单个 Worker 执行返回的结果封装（支持 operator.add 并发收集）"""
    task_id: str
    task: Optional[str]
    result: str
    error: Optional[str]
    # 执行该子任务的 Agent 名称
    agent: Optional[str]
    # 子任务总耗时（毫秒）
    elapsed_ms: Optional[int]

class AgentState(TypedDict, total=False):
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
        domain_candidates: 候选数据域列表（多域场景）
        intent_candidates: 候选意图/Agent 列表（多意图场景）
        route_strategy: 路由策略（single_domain / multi_domain_split / complex_single_domain）
        route_reason: 路由策略原因（便于前端和日志追踪）
        domain_elapsed_ms: Domain Router 耗时

        ---- 意图识别 (Intent Router) 输出 ----
        intent: 识别到的意图/Agent 名称
        intent_confidence: 置信度 0~1
        is_complex: 是否为复杂请求
        direct_answer: LLM 直接给出的回答
        intent_elapsed_ms: Intent Router 耗时

        ---- 任务规划 (Parent Planner) 输出 ----
        plan: 原子任务列表（Send 并发扇出输入）
        current_task: 当前 Worker 正在处理的单个任务文本
        task_list: 兼容旧链路保留的 DAG 描述
        task_results: 兼容旧链路保留的执行结果映射 {task_id: result}
        current_wave: 兼容旧链路保留的调度波次
        max_waves: 兼容旧链路保留的安全阀
        planner_source: 任务规划来源（rule_split / llm / fallback）
        planner_elapsed_ms: Parent Planner 耗时
        reflection_round: 已执行的自动反思轮次
        max_reflection_rounds: 允许的最大自动反思轮次
        next_task_sequence: 新增任务编号游标
        reflection_source: 本轮反思来源（llm / disabled / skipped）
        reflection_summary: 本轮反思结论摘要

        ---- 并发执行 (Dispatcher & Worker) ----
        active_tasks: 当前准备 Send 派发执行的子任务列表
        worker_results: 并发 Worker 执行后收集的增量结果
        
        ---- 路由元数据 ----
        current_node: 当前执行到的节点名称
        next: 路由指示器
        current_step_input: 当前正在执行的原子步骤
        current_step_agent: 当前步骤分配到的 Agent
        executor_active: 是否处于 Plan-and-Execute 执行闭环中
    """
    messages: Annotated[List[BaseMessage], add_messages]
    session_id: str
    llm_config: dict
    # 会话结构化上下文（城市/画像等）
    context_slots: Dict[str, Any]
    # 会话上下文摘要（用于系统提示注入）
    context_summary: str

    # 数据域路由输出
    data_domain: str
    domain_confidence: float
    domain_route_source: str
    domain_candidates: List[str]
    intent_candidates: List[str]
    route_strategy: str
    route_reason: str
    domain_elapsed_ms: int

    # 意图识别输出
    intent: str
    intent_confidence: float
    is_complex: bool
    direct_answer: str
    intent_elapsed_ms: int

    # Planner 长期记忆与规划
    plan: List[str]
    current_task: str
    current_task_id: str
    memory: Dict[str, Any]

    # 任务规划输出 (Tier-2 DAG)
    task_list: List[SubTask]
    task_results: Dict[str, str]
    current_wave: int
    max_waves: int
    planner_source: str
    planner_elapsed_ms: int
    reflection_round: int
    max_reflection_rounds: int
    next_task_sequence: int
    reflection_source: str
    reflection_summary: str

    # 并发执行输出 (Map-Reduce)
    active_tasks: List[Dict[str, Any]]
    worker_results: Annotated[List[WorkerResult], operator.add]

    # 路由元数据
    current_node: str
    next: str
    current_step_input: str
    current_step_agent: str
    executor_active: bool
    interrupt_payload: Dict[str, Any]
    error_message: str
    error_detail: str
    last_step_result: str
    last_step_status: str
    last_step_error: str
    replan_count: int
    max_replans: int
    replan_reason: str


GraphState = AgentState
