import functools
import re
import json
from typing import Optional, List, Dict, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START
from langgraph.constants import Send
from pydantic import BaseModel, Field

from agent.graph_state import AgentRequest
from agent.graphs.checkpointer import checkpointer
from agent.graphs.state import GraphState, SubTask, WorkerResult
from agent.llm.unified_loader import create_model_from_config
from agent.registry import agent_classes, MEMBERS
from agent.prompts.supervisor_prompt import IntentRouterPrompt, ChatFallbackPrompt, PlannerPrompt, AggregatorPrompt
from services.route_metrics_service import route_metrics_service
from utils.custom_logger import get_logger
from utils.date_utils import get_agent_date_context, get_current_time_context

log = get_logger(__name__)
INTERRUPT_RESULT_TYPE = "__interrupt__"

# --- Models ---
class IntentDecision(BaseModel):
    intent: str
    confidence: float
    is_complex: bool
    direct_answer: str


class DomainDecision(BaseModel):
    data_domain: str
    confidence: float
    source: str


class WorkerState(TypedDict):
    task: SubTask


def _looks_like_sql_request(text: str) -> bool:
    """快速识别可直接路由到 SQL Agent 的请求模式。"""
    t = (text or "").strip().lower()
    if not t:
        return False

    patterns = [
        r"\border\s+by\b",
        r"\blimit\b",
        r"\bselect\b",
        r"\bwhere\b",
        r"\bgroup\s+by\b",
        r"\bjoin\b",
        r"数据库",
        r"数据表",
        r"表里",
        r"按id",
        r"按\s*id",
        r"根据\s*id",
        r"倒序",
        r"倒叙",
        r"降序",
        r"前\d+条",
        r"最新\d+条",
        r"sql",
    ]
    return any(re.search(p, t) for p in patterns)


def _looks_like_holter_request(text: str) -> bool:
    """识别 Holter/云柚业务域请求，用于优先路由到 yunyou_agent。"""
    t = (text or "").strip().lower()
    if not t:
        return False

    keywords = [
        "holter",
        "云柚",
        "动态心电",
        "心电报告",
        "报告状态",
        "贴片",
    ]
    return any(k in t for k in keywords)


def _looks_like_search_request(text: str) -> bool:
    """识别互联网检索类问题。"""
    t = (text or "").strip().lower()
    if not t:
        return False
    keywords = [
        "活动",
        "新闻",
        "今天",
        "明天",
        "后天",
        "最新",
        "搜一下",
        "查一下",
        "郑州",
        "天气",
        "什么活动",
    ]
    return any(k in t for k in keywords)

# --- Helpers ---
def _latest_human_message(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        text = item.get("text") or item.get("content") or ""
                        if isinstance(text, str):
                            parts.append(text)
                return " ".join(parts).strip()
            return str(content or "").strip()
    return ""


def _content_to_text(content: Any) -> str:
    """将 LLM content（str / block list / other）统一转为可展示文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str) and text:
                    parts.append(text)
        if parts:
            return "\n".join(parts)
        return str(content)
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        if isinstance(text, str):
            return text
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _is_followup_supplement(text: str) -> bool:
    """识别用户是否在补充上轮追问信息（日期、确认词、简短参数）。"""
    t = (text or "").strip().lower()
    if not t:
        return False

    if t in {"是", "否", "好的", "继续", "确认", "yes", "no", "ok"}:
        return True

    # 仅包含日期/分隔符/空白的短文本，通常是对“请补充时间范围”的回答
    if len(t) <= 60 and re.fullmatch(r"[0-9\-\s:~至到/,，.]+", t):
        return True

    if len(t) <= 80 and re.search(r"\b20\d{2}-\d{2}-\d{2}\b", t):
        return True

    # 排序/条数/字段补充，常见于“你先查最近数据，再补一句按 id 倒序前 5 条”
    if len(t) <= 120 and any(k in t for k in ["按id", "按 id", "根据id", "根据 id", "倒序", "倒叙", "降序", "前", "limit", "order by"]):
        return True

    return False


def _history_hint_intent(messages: List[BaseMessage]) -> Optional[str]:
    """从最近上下文中推断业务域，避免补充信息被错误路由到 CHAT。"""
    recent = " ".join(
        (msg.content or "") for msg in messages[-8:]
        if isinstance(msg, (HumanMessage, AIMessage))
    ).lower()

    if any(k in recent for k in ["holter", "云柚", "心电", "报告状态", "审核"]):
        return "yunyou_agent"
    if any(k in recent for k in ["sql", "数据库", "查询语句", "数据表"]):
        return "sql_agent"
    return None

def _parse_json_from_text(text: str) -> dict:
    """从 LLM 返回文本中提取 JSON 对象（支持嵌套花括号）。"""
    # 找到第一个 '{' 后进行括号配对，避免非贪婪匹配截断嵌套结构
    start = text.find('{')
    if start == -1:
        log.error(f"Failed to find JSON object in text. Raw text was:\n{text}")
        raise ValueError("No JSON object found in text")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                json_str = text[start:i+1]
                # 清除由于大模型吐字不规范带来的尾部逗号 (Trailing Commas)
                json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
                return json.loads(json_str)
    raise ValueError("Unbalanced braces in JSON")


def _normalize_interrupt_payload(val: Any) -> dict:
    """将不同形态的 Interrupt 值转成统一 payload。"""
    if hasattr(val, "value"):
        return _normalize_interrupt_payload(getattr(val, "value"))

    if isinstance(val, dict):
        payload = dict(val)
    elif hasattr(val, "__dict__"):
        payload = dict(getattr(val, "__dict__", {}))
    else:
        payload = {"message": str(val)}

    payload.setdefault("message", "需要人工审核")
    payload.setdefault("allowed_decisions", ["approve", "reject"])
    payload.setdefault("action_requests", [])
    return payload


def _extract_interrupt_from_snapshot(snapshot: Any) -> Optional[dict]:
    tasks = getattr(snapshot, "tasks", None) or []
    for task in tasks:
        interrupts = getattr(task, "interrupts", None) or []
        if not interrupts:
            continue
        first_interrupt = interrupts[0]
        payload = getattr(first_interrupt, "value", first_interrupt)
        return _normalize_interrupt_payload(payload)
    return None


def _run_agent_to_completion(
    agent_name: str,
    user_input: str,
    model: BaseChatModel,
    config: RunnableConfig,
    session_id: str = ""
) -> Any:
    """共享的 Agent 执行逻辑，供 worker_node 和 single_agent_node 复用。"""
    if agent_name not in MEMBERS:
        # 降级为通用 CHAT
        response = model.invoke(
            [
                ("system", ChatFallbackPrompt.get_system_prompt()),
                ("system", get_agent_date_context()),
                HumanMessage(content=user_input),
            ],
            config=config,
        )
        return response.content

    req = AgentRequest(
        user_input=user_input, model=model,
        session_id=session_id or config.get("configurable", {}).get("thread_id", ""),
        subgraph_id=agent_name, llm_config={}
    )
    agent_instance = agent_classes[agent_name].cls(req)

    final_response = None
    agent_error = None
    for event in agent_instance.run(req, config=config):
        if not isinstance(event, dict):
            continue
        if "error" in event:
            agent_error = event["error"]
        if "interrupt" in event:
            payload = _normalize_interrupt_payload(event.get("interrupt"))
            payload["agent_name"] = payload.get("agent_name") or agent_name
            return {"type": INTERRUPT_RESULT_TYPE, "payload": payload}
        for node_val in event.values():
            if not isinstance(node_val, dict) or "messages" not in node_val:
                continue
            for msg in node_val.get("messages", []):
                # 实时代理: 将内部 Agent 的 Tool Calls 推送到日志系统 (这会被 graph_runner 的 interceptor 捕获并推至前端)
                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        log.info(f"被动调度: 正在调用工具 {tc.get('name', '...')} ...")
                # 寻找最终的纯文本响应
                if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                    final_response = msg

    if final_response:
        normalized = _content_to_text(final_response.content).strip()
        return normalized or "已完成处理，但未生成可展示文本。请重试或更换提问方式。"
        
    # 如果没有最终 response，检查是否因为中断挂起
    if agent_error is None:
        root_thread_id = session_id or config.get("configurable", {}).get("thread_id", "")
        if root_thread_id:
            subgraph_config = {"configurable": {"thread_id": f"{root_thread_id}_{agent_name}"}}
            snapshot = agent_instance.graph.get_state(subgraph_config)
            snapshot_interrupt = _extract_interrupt_from_snapshot(snapshot)
            if snapshot_interrupt:
                snapshot_interrupt["agent_name"] = snapshot_interrupt.get("agent_name") or agent_name
                return {"type": INTERRUPT_RESULT_TYPE, "payload": snapshot_interrupt}
            
    raise RuntimeError(f"{agent_name} 执行失败: {agent_error}")


# ==================== Tier-0.5: 数据域路由器 (Domain Router) ====================
def domain_router_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
    """
    先识别“去哪类数据域”，再做意图路由。

    数据域定义：
    1. YUNYOU_DB: 云柚/holter 业务域
    2. LOCAL_DB: 本地业务库 SQL 域
    3. WEB_SEARCH: 互联网检索域
    4. GENERAL: 通用对话域
    """
    messages = state.get("messages", [])
    session_id = state.get("session_id") or config.get("configurable", {}).get("thread_id", "")
    latest_user_text = _latest_human_message(messages[-12:])

    # follow-up 补充句优先继承上轮域
    if _is_followup_supplement(latest_user_text):
        hinted_intent = _history_hint_intent(messages[-12:])
        if hinted_intent == "yunyou_agent":
            decision = DomainDecision(data_domain="YUNYOU_DB", confidence=0.94, source="history")
        elif hinted_intent == "sql_agent":
            decision = DomainDecision(data_domain="LOCAL_DB", confidence=0.93, source="history")
        else:
            decision = None
        if decision:
            route_metrics_service.record_domain_decision(
                session_id=session_id,
                user_text=latest_user_text,
                domain=decision.data_domain,
                confidence=decision.confidence,
                source=decision.source,
            )
            return {
                "data_domain": decision.data_domain,
                "domain_confidence": decision.confidence,
                "domain_route_source": decision.source,
            }

    # 规则优先
    if _looks_like_holter_request(latest_user_text):
        decision = DomainDecision(data_domain="YUNYOU_DB", confidence=0.98, source="rule")
    elif _looks_like_sql_request(latest_user_text):
        decision = DomainDecision(data_domain="LOCAL_DB", confidence=0.95, source="rule")
    elif _looks_like_search_request(latest_user_text):
        decision = DomainDecision(data_domain="WEB_SEARCH", confidence=0.9, source="rule")
    else:
        # 轻量 LLM 兜底分类
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是数据域分类器。仅输出 JSON："
             "{\"data_domain\":\"YUNYOU_DB|LOCAL_DB|WEB_SEARCH|GENERAL\",\"confidence\":0-1}。"
             "不要输出其它文本。"),
            MessagesPlaceholder(variable_name="messages"),
        ])
        try:
            non_streaming_config = config.copy() if config else {}
            if "callbacks" in non_streaming_config:
                del non_streaming_config["callbacks"]
            response = (prompt | model).invoke({"messages": messages[-8:]}, config=non_streaming_config)
            data = _parse_json_from_text(response.content)
            domain = str(data.get("data_domain", "GENERAL")).upper()
            confidence = float(data.get("confidence", 0.5))
            if domain not in {"YUNYOU_DB", "LOCAL_DB", "WEB_SEARCH", "GENERAL"}:
                domain = "GENERAL"
            decision = DomainDecision(data_domain=domain, confidence=confidence, source="llm")
        except Exception as exc:
            log.warning(f"Domain router fallback: {exc}")
            decision = DomainDecision(data_domain="GENERAL", confidence=0.4, source="fallback")

    route_metrics_service.record_domain_decision(
        session_id=session_id,
        user_text=latest_user_text,
        domain=decision.data_domain,
        confidence=decision.confidence,
        source=decision.source,
    )
    return {
        "data_domain": decision.data_domain,
        "domain_confidence": decision.confidence,
        "domain_route_source": decision.source,
    }


# ==================== Tier-1: 意图路由器 (Intent Router) ====================
def intent_router_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
    """第二级：意图识别节点 (Intent_Router_Node)"""
    messages = state.get("messages", [])
    session_id = state.get("session_id") or config.get("configurable", {}).get("thread_id", "")
    data_domain = (state.get("data_domain") or "GENERAL").upper()
    domain_conf = float(state.get("domain_confidence") or 0.0)
    domain_source = state.get("domain_route_source") or "unknown"
    # 保留最近多轮上下文，避免“用户补充参数”被误判为新问题导致重复追问
    trimmed_messages = messages[-12:]

    latest_user_text = _latest_human_message(trimmed_messages)

    # 先处理“补充参数”场景：继承上轮领域，避免补充句被 SQL fast-path 截走。
    if _is_followup_supplement(latest_user_text):
        hinted_intent = _history_hint_intent(trimmed_messages)
        if hinted_intent:
            log.info(f"Intent follow-up carry-over: route to [{hinted_intent}]")
            route_metrics_service.record_intent_decision(
                session_id=session_id,
                user_text=latest_user_text,
                intent=hinted_intent,
                confidence=0.93,
                source="followup_history",
            )
            return {
                "intent": hinted_intent,
                "intent_confidence": 0.93,
                "is_complex": False,
                "direct_answer": "",
            }

    # 先按数据域强约束路由，避免跨域误查
    if data_domain == "YUNYOU_DB":
        route_metrics_service.record_intent_decision(
            session_id=session_id,
            user_text=latest_user_text,
            intent="yunyou_agent",
            confidence=max(domain_conf, 0.95),
            source=f"domain_{domain_source}",
        )
        return {
            "intent": "yunyou_agent",
            "intent_confidence": max(domain_conf, 0.95),
            "is_complex": False,
            "direct_answer": "",
        }
    if data_domain == "LOCAL_DB" and _looks_like_sql_request(latest_user_text):
        route_metrics_service.record_intent_decision(
            session_id=session_id,
            user_text=latest_user_text,
            intent="sql_agent",
            confidence=max(domain_conf, 0.94),
            source=f"domain_{domain_source}",
        )
        return {
            "intent": "sql_agent",
            "intent_confidence": max(domain_conf, 0.94),
            "is_complex": False,
            "direct_answer": "",
        }
    if data_domain == "WEB_SEARCH":
        fallback_agent = "weather_agent" if "天气" in latest_user_text else "search_agent"
        route_metrics_service.record_intent_decision(
            session_id=session_id,
            user_text=latest_user_text,
            intent=fallback_agent,
            confidence=max(domain_conf, 0.9),
            source=f"domain_{domain_source}",
        )
        return {
            "intent": fallback_agent,
            "intent_confidence": max(domain_conf, 0.9),
            "is_complex": False,
            "direct_answer": "",
        }

    # 业务域优先路由：Holter/云柚相关查询，优先进入 yunyou_agent。
    # 注意：必须放在 SQL fast-path 之前，否则会被“order by/limit/数据库”误路由到 sql_agent。
    if _looks_like_holter_request(latest_user_text):
        log.info("Intent fast-path: 命中 Holter/云柚业务域，直接路由 yunyou_agent")
        route_metrics_service.record_intent_decision(
            session_id=session_id,
            user_text=latest_user_text,
            intent="yunyou_agent",
            confidence=0.96,
            source="rule_holter",
        )
        return {
            "intent": "yunyou_agent",
            "intent_confidence": 0.96,
            "is_complex": False,
            "direct_answer": "",
        }

    # SQL 快速路由：用户明确表达了 SQL/排序/TopN 查询诉求时，优先进入 sql_agent。
    # 业务上“按 id 倒序/前 N 条/order by/limit”这类语句通常是直接查库意图。
    if _looks_like_sql_request(latest_user_text):
        log.info("Intent fast-path: 命中 SQL 语义特征，直接路由 sql_agent")
        route_metrics_service.record_intent_decision(
            session_id=session_id,
            user_text=latest_user_text,
            intent="sql_agent",
            confidence=0.95,
            source="rule_sql",
        )
        return {
            "intent": "sql_agent",
            "intent_confidence": 0.95,
            "is_complex": False,
            "direct_answer": "",
        }

    prompt = ChatPromptTemplate.from_messages([
        ("system", IntentRouterPrompt.get_system_prompt()),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    router_model = model
    
    try:
        non_streaming_config = config.copy() if config else {}
        if "callbacks" in non_streaming_config: del non_streaming_config["callbacks"]
        response = (prompt | router_model).invoke({"messages": trimmed_messages}, config=non_streaming_config)
        data = _parse_json_from_text(response.content)
        
        decision = IntentDecision(
            intent=data.get("intent", "CHAT"),
            confidence=float(data.get("confidence", 0.5)),
            is_complex=bool(data.get("is_complex", False)),
            direct_answer=data.get("direct_answer", "")
        )
    except Exception as exc:
        log.warning(f"Intent parsing fallback: {exc}")
        # 如果解析失败但用户输入比较长，宁可错杀交给 Planner 当做复杂任务，避免直通 CHAT 白屏
        user_input_length = len(trimmed_messages[-1].content) if trimmed_messages else 0
        if user_input_length > 50:
            log.info("Fallack 捕获长句 (>50 chars)，强制路由至 Parent_Planner_Node")
            decision = IntentDecision(intent="CHAT", confidence=0.3, is_complex=True, direct_answer="")
        else:
            decision = IntentDecision(intent="CHAT", confidence=0.3, is_complex=False, direct_answer="")
        
    log.info(f"Tier-1 Router: intent=[{decision.intent}], conf=[{decision.confidence}], complex=[{decision.is_complex}]")
    route_metrics_service.record_intent_decision(
        session_id=session_id,
        user_text=latest_user_text,
        intent=decision.intent,
        confidence=decision.confidence,
        source="llm",
    )
    return {
        "intent": decision.intent,
        "intent_confidence": decision.confidence,
        "is_complex": decision.is_complex,
        "direct_answer": decision.direct_answer,
    }

# ==================== Tier-2: DAG Planner & Dispatcher ====================
def parent_planner_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
    """第三级：父规划器 (Parent_Planner_Node)"""
    messages = state.get("messages", [])
    prompt = ChatPromptTemplate.from_messages([
        ("system", PlannerPrompt.get_system_prompt()),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    try:
        response = (prompt | model).invoke({"messages": messages}, config=config)
        data = _parse_json_from_text(response.content)
        tasks_data = data.get("tasks", [])
        
        task_list: List[SubTask] = []
        for idx, t in enumerate(tasks_data):
            task_list.append({
                "id": str(t.get("id", f"t{idx}")),
                "agent": str(t.get("agent", "CHAT")),
                "input": str(t.get("input", "")),
                "depends_on": [str(x) for x in t.get("depends_on", [])],
                "status": "pending",
                "result": None
            })
            
    except Exception as exc:
        log.warning(f"Planner parsing fallback: {exc}")
        job: SubTask = {
            "id": "t1", "agent": "CHAT", "input": _latest_human_message(messages),
            "depends_on": [], "status": "pending", "result": None
        }
        task_list = [job]
        
    log.info(f"Tier-2 Planner: Generated {len(task_list)} tasks -> {[t['id'] for t in task_list]}")
    return {
        "task_list": task_list,
        "task_results": {},
        "current_wave": 0,
        "max_waves": len(task_list) * 2 + 2
    }


def dispatcher_node(state: GraphState) -> dict:
    """Dispatcher：提取可并发执行的无依赖任务，并标记为 dispatched。"""
    tasks = state.get("task_list", [])
    current_wave = state.get("current_wave", 0)
    done_ids = {t["id"] for t in tasks if t["status"] == "done"}

    active_tasks = []
    new_task_list = []

    for task in tasks:
        new_task = dict(task)
        if new_task["status"] == "pending":
            # 检查所有依赖是否已完成
            deps_met = all(dep_id in done_ids for dep_id in new_task.get("depends_on", []))
            if deps_met:
                new_task["status"] = "dispatched"
                active_tasks.append(new_task)
        new_task_list.append(new_task)

    log.info(f"Dispatcher [Wave {current_wave}]: {len(active_tasks)} tasks ready to dispatch.")
    return {
        "task_list": new_task_list,
        "active_tasks": active_tasks,
        "current_wave": current_wave + 1,
        "worker_results": [],  # 重置本轮 worker 结果缓冲
    }

def dispatch_router(state: GraphState):
    """Conditional Edge：根据 active_tasks 发起扇出，若全部完成则聚合。"""
    active = state.get("active_tasks", [])
    if active:
        # Fan-out 到 worker_node 进行并行执行
        return [Send("worker_node", {"task": t}) for t in active]
        
    # 如果没要执行的任务，查验是等待别人完成，还是全剧终，还是死锁
    tasks = state.get("task_list", [])
    if any(t["status"] in ["pending", "dispatched", "pending_approval"] for t in tasks):
        # 异常：死锁或者波次超限
        max_waves = state.get("max_waves", 10)
        current = state.get("current_wave", 0)
        
        # 波次超限：直接退出
        if current >= max_waves:
            log.warning("Dispatcher: DAG execution reached max waves, force quitting.")
            return "aggregator_node"
            
        # 死锁检测：图里有 pending，且没有 dispatched 在跑（全军覆没卡死在等待依赖上）
        _has_dispatched = any(t["status"] == "dispatched" for t in tasks)
        _has_pending = any(t["status"] == "pending" for t in tasks)
        _has_approval = any(t["status"] == "pending_approval" for t in tasks)
        
        if _has_pending and not _has_dispatched and not _has_approval:
            log.warning("Dispatcher: 💥 Dependency deadlock detected (Pending nodes waiting for ghosts). Force quit.")
            return "aggregator_node"
            
        # 否则如果是还在等待 dispatched 或审批任务完成，直接由于没有 active 返回，这里 LangGraph 会暂停 (因为没有出边可走)
        if _has_dispatched or _has_approval:
            log.info("Dispatcher: Waiting for dispatched or pending_approval tasks to finish. Yielding to Graph runner.")
            # 这是一个关键的改动：如果只有 pending_approval 的任务，我们不能路由到 aggregator_node 结束图，
            # 而是应该抛出真正的 LangGraph Interrupt 或者是结束这一个步骤，让图引擎静默挂起等待外部输入。
            # LangGraph 要求图必须能流转，这里如果卡在空路由会报错，由于我们已经在 __INTERRUPTED_PENDING_APPROVAL__ 用 Exception 模拟过，
            # 所以其实图在这之前就已经被我们的外置 _run_agent_to_completion 所打断了。
            # 但是如果没被打断跑到这里，就让它安全走向终点以便提取状态
            return "aggregator_node"
            
    return "aggregator_node"


# ==================== Worker & Reducer ====================
def worker_node(state: WorkerState, config: RunnableConfig, model: BaseChatModel) -> dict:
    """Worker：执行单个 SubTask 并返回结果。"""
    task = state["task"]
    log.info(f"Worker start: task=[{task['id']}], agent=[{task['agent']}]")

    try:
        res_text = _run_agent_to_completion(task["agent"], task["input"], model, config)
        if isinstance(res_text, dict) and res_text.get("type") == INTERRUPT_RESULT_TYPE:
            log.info(f"Worker [{task['id']}] 中断挂起，等待人工审批")
            payload = _normalize_interrupt_payload(res_text.get("payload"))
            payload["agent_name"] = payload.get("agent_name") or task["agent"]
            return {
                "worker_results": [WorkerResult(task_id=task["id"], result="pending_approval", error=None)],
                "interrupt_payload": payload
            }
    except Exception as exc:
        err_msg = str(exc)
        if "Interrupt(" in err_msg or exc.__class__.__name__ == "GraphInterrupt":
            log.info(f"Worker [{task['id']}] 中断挂起 (通过异常捕获)，等待人工审批")
            return {
                "worker_results": [WorkerResult(task_id=task["id"], result="pending_approval", error=None)],
                "interrupt_payload": {
                    "message": "需要人工审核",
                    "allowed_decisions": ["approve", "reject"],
                    "action_requests": [],
                    "agent_name": task["agent"],
                }
            }
        log.error(f"Worker [{task['id']}] error: {exc}")
        return {"worker_results": [WorkerResult(task_id=task["id"], result="", error=str(exc))]}

    return {"worker_results": [WorkerResult(task_id=task["id"], result=res_text, error=None)]}


def reducer_node(state: GraphState) -> dict:
    """Reducer：回收并发结果，更新全图 task_list 状态。"""
    new_results = state.get("worker_results", [])
    if not new_results:
        return {}
        
    tasks = state.get("task_list", [])
    task_res_map = state.get("task_results", {})
    
    new_task_list = []
    
    for task in tasks:
        new_task = dict(task)
        # 寻找匹配的执行结果
        matched = [r for r in new_results if r["task_id"] == task["id"]]
        if matched:
            worker_res = matched[0] # 取出其中一个
            if new_task["status"] != "done": # 防止重复标记
                if worker_res.get("result") == "pending_approval":
                    new_task["status"] = "pending_approval"  # 更改状态为 pending_approval 阻塞 DAG 但不触发死锁
                    log.info(f"Reducer: task [{task['id']}] 挂起，等待人工审批完成。")
                else:
                    new_task["status"] = "done"
                    if worker_res.get("error"):
                        new_task["result"] = f"Error: {worker_res['error']}"
                        new_task["status"] = "error"
                    else:
                        new_task["result"] = worker_res["result"]
                        
                    task_res_map[task["id"]] = new_task["result"]
                    log.info(f"Reducer: task [{task['id']}] finished.")
            
        new_task_list.append(new_task)
        
    return {"task_list": new_task_list, "task_results": task_res_map}


# ==================== Aggregator ====================
def aggregator_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
    """收尾级：结果聚合节点 (Aggregator_Node)"""
    results = state.get("task_results", {})
    if not results:
        msg = AIMessage(content="没有任何子任务结果可以聚合。", name="Aggregator")
        return {"messages": [msg]}
        
    # Phase 2：将所有任务运行结果抛给大模型润色打包
    res_list = [f"【任务 {k} 的反馈】:\n{v}" for k, v in results.items()]
    agg_msg = "\n\n---\n\n".join(res_list)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", AggregatorPrompt.SYSTEM),
        ("human", f"用户的原始请求：\n{_latest_human_message(state.get('messages', []))}\n\n执行结果反馈：\n{agg_msg}")
    ])
    
    response = (prompt | model).invoke({}, config=config)
    final_content = response.content
    
    msg = AIMessage(content=final_content, name="Aggregator", response_metadata={"synthetic": True})
    return {"messages": [msg], "direct_answer": final_content}


def chat_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
    """单一的对话闲聊节点 (非复杂DAG走这)"""
    direct_ans = state.get("direct_answer", "")
    if direct_ans and len(direct_ans.strip()) > 3:
        msg = AIMessage(content=direct_ans, name="ChatAgent", response_metadata={"synthetic": True})
    else:
        prompt = ChatFallbackPrompt.get_system_prompt()
        response = model.invoke(
            [("system", prompt), ("system", get_agent_date_context())] + state.get("messages", []),
            config=config
        )
        msg = AIMessage(content=response.content, name="ChatAgent")
    return {"messages": [msg]}


def single_agent_node(state: GraphState, agent_name: str, model: BaseChatModel, config: RunnableConfig) -> dict:
    """单一实体 Agent 包装执行 (非复杂 DAG 路线)，复用共享执行逻辑。"""
    user_input = _latest_human_message(state.get("messages", []))
    try:
        content = _run_agent_to_completion(
            agent_name, user_input, model, config,
            session_id=state.get("session_id") or ""
        )
        if isinstance(content, dict) and content.get("type") == INTERRUPT_RESULT_TYPE:
            payload = _normalize_interrupt_payload(content.get("payload"))
            payload["agent_name"] = payload.get("agent_name") or agent_name
            return {"interrupt_payload": payload}
        res_msg = AIMessage(content=content, name=agent_name, response_metadata={"synthetic": True})
    except Exception as exc:
        err_msg = str(exc)
        if "Interrupt(" in err_msg or exc.__class__.__name__ == "GraphInterrupt":
            log.info(f"Single Agent [{agent_name}] 中断挂起，等待人工审批")
            return {
                "interrupt_payload": {
                    "message": "需要人工审核",
                    "allowed_decisions": ["approve", "reject"],
                    "action_requests": [],
                    "agent_name": agent_name,
                }
            }
        res_msg = AIMessage(content=f"⚠️ {agent_name} 执行异常: {exc}", name=agent_name, response_metadata={"synthetic": True})
    return {"messages": [res_msg]}


# ==================== 路由与编排逻辑 ====================
def _route_after_intent(state: GraphState) -> str:
    """
    条件边：Intent_Router → 单兵/DAG/聊天

    决策逻辑（与架构规范对齐）：
    1. is_complex=True → Parent_Planner_Node（DAG 拆解）
    2. confidence >= 0.7 且 intent 在 MEMBERS 中 → 直接路由到专业 Agent
    3. 其余情况 → chat_node（通用对话兜底）
    """
    conf = state.get("intent_confidence", 0.0)
    is_comp = state.get("is_complex", False)
    intent = state.get("intent", "CHAT")

    # 复杂任务 → 无论置信度高低，都进入 DAG 拆解
    if is_comp:
        log.info(f"路由决策: 复杂任务 → Parent_Planner_Node (intent={intent}, conf={conf:.2f})")
        return "Parent_Planner_Node"

    # 高置信单一意图 → 直接路由到对应 Agent
    if conf >= 0.7 and intent in MEMBERS:
        log.info(f"路由决策: 高置信单兵 → {intent} (conf={conf:.2f})")
        return intent

    # 兜底 → 通用对话
    log.info(f"路由决策: 兜底对话 → chat_node (intent={intent}, conf={conf:.2f})")
    return "chat_node"


def create_graph(model_config: Optional[dict] = None):
    """构建遵循生产级 3 层逻辑拓扑结构的融合 StateGraph"""
    config_dict = model_config or {}
    model, _ = create_model_from_config(**config_dict)
    
    # 尝试加载 Tier-1 路由专用小模型 (如 gemini-2.0-flash-lite)
    router_model_name = config_dict.get("router_model")
    router_model = model
    
    if router_model_name and router_model_name != config_dict.get("model"):
        try:
            router_config = dict(config_dict)
            router_config["model"] = router_model_name
            # 这里默认共享主模型的 service_type 和 keys
            temp_router_model, _ = create_model_from_config(**router_config)
            router_model = temp_router_model
            log.info(f"Tier-1 极速路由引擎已挂载小模型: {router_model_name}")
        except Exception as e:
            log.warning(f"挂载小模型路由 [{router_model_name}] 失败，回退至主模型: {e}")

    workflow = StateGraph(GraphState)

    # 上层的决策分析
    workflow.add_node("Domain_Router_Node", functools.partial(domain_router_node, model=router_model))
    workflow.add_node("Intent_Router_Node", functools.partial(intent_router_node, model=router_model))
    workflow.add_node("Parent_Planner_Node", functools.partial(parent_planner_node, model=model))
    
    # 动态 DAG 发牌与执行网络
    workflow.add_node("dispatcher_node", dispatcher_node)
    workflow.add_node("worker_node", functools.partial(worker_node, model=model))
    workflow.add_node("reducer_node", reducer_node)
    workflow.add_node("aggregator_node", functools.partial(aggregator_node, model=model))
    
    # 单兵节点 (非 DAG 路径使用)
    workflow.add_node("chat_node", functools.partial(chat_node, model=model))
    for name in MEMBERS:
        workflow.add_node(name, functools.partial(single_agent_node, agent_name=name, model=model))

    # ================= 编织拓扑关系 =================
    workflow.add_edge(START, "Domain_Router_Node")
    workflow.add_edge("Domain_Router_Node", "Intent_Router_Node")
    
    router_options = {name: name for name in MEMBERS}
    router_options.update({"chat_node": "chat_node", "Parent_Planner_Node": "Parent_Planner_Node"})
    workflow.add_conditional_edges("Intent_Router_Node", _route_after_intent, router_options)
    
    # DAG 循环执行部分
    workflow.add_edge("Parent_Planner_Node", "dispatcher_node")
    workflow.add_conditional_edges("dispatcher_node", dispatch_router, ["worker_node", "aggregator_node"])
    workflow.add_edge("worker_node", "reducer_node")
    workflow.add_edge("reducer_node", "dispatcher_node") # 闭环：拉取下一波任务
    
    # 单兵出口
    workflow.add_edge("chat_node", END)
    for name in MEMBERS:
        workflow.add_edge(name, END)
        
    workflow.add_edge("aggregator_node", END)

    return workflow.compile(checkpointer=checkpointer)
