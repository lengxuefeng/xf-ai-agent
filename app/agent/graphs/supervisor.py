import functools
import re
import json
from typing import Optional, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field

from agent.graph_state import AgentRequest
from agent.graphs.checkpointer import checkpointer
from agent.graphs.state import GraphState
from agent.llm.unified_loader import create_model_from_config
from agent.registry import agent_classes, MEMBERS
from utils.custom_logger import get_logger

log = get_logger(__name__)

"""
【模块说明】
系统的核心大脑。负责接收用户输入，并决定分发给哪个下游 Agent（或直接 Chat 兜底）。
采用规则、关键词、大模型语义三级降级路由策略，兼顾速度与智能。
"""


class RouteDecision(BaseModel):
    route: str = Field(description="下一个路由目标")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="路由置信度")
    reason: str = Field(default="", description="路由原因")


def _latest_human_message(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return (msg.content or "").strip()
    return ""


def _previous_agent_name(messages: List[BaseMessage]) -> Optional[str]:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.name in MEMBERS:
            return msg.name
    return None


def _is_short_followup(user_text: str) -> bool:
    text = (user_text or "").strip().lower()
    short_confirmations = {"好", "好的", "是", "继续", "继续吧", "行", "可以", "确认", "嗯", "对", "然后呢"}
    return text in short_confirmations or len(text) <= 4


def _keyword_route(user_text: str) -> Optional[RouteDecision]:
    query = (user_text or "").lower().strip()
    if not query: return None

    scores = {}
    for name, info in agent_classes.items():
        hit = sum(1 for kw in info.keywords if kw and kw.lower() in query)
        if hit > 0: scores[name] = hit

    if not scores: return None
    best_agent = max(scores, key=scores.get)
    total_hits = sum(scores.values())
    confidence = min(0.98, 0.6 + (scores[best_agent] / max(total_hits, 1)) * 0.4)
    return RouteDecision(route=best_agent, confidence=confidence, reason=f"关键词命中: {best_agent}")


def _llm_route(messages: List[BaseMessage], model: BaseChatModel, config: RunnableConfig) -> RouteDecision:
    options = ["FINISH"] + MEMBERS + ["CHAT"]
    options_str = ", ".join(options)

    system_prompt = (
            "你是 AI Agent 任务路由器。只负责路由，不回答问题。\n"
            "可用目标:\n" + "\n".join([f"- {name}: {info.description}" for name, info in agent_classes.items()]) +
            "\n- CHAT: 通用对话兜底\n- FINISH: 任务结束\n"
            "必须返回 JSON: {\"route\": \"目标\", \"confidence\": 0.9, \"reason\": \"原因\"}"
    )

    # 路由器只需要看最近的 4 条消息，降低干扰，节省 token
    trimmed_messages = trim_messages(messages, max_tokens=4, strategy="last", token_counter=len)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", f"请返回 JSON 格式决策，route 必选: {options_str}")
    ])

    try:
        structured_model = model.with_structured_output(RouteDecision)
        decision = (prompt | structured_model).invoke({"messages": trimmed_messages}, config=config)
        route = (decision.route or "CHAT").strip()
        return RouteDecision(route=route if route in options else "CHAT", confidence=decision.confidence,
                             reason=decision.reason)
    except Exception as exc:
        log.warning(f"结构化路由失败，回退纯文本提取: {exc}")
        response = (prompt | model).invoke({"messages": trimmed_messages}, config=config)
        text = str(response.content).strip()

        # 【核心修复】强大的正则：剔除 GLM 等模型自带的 ```json 标签
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()

        try:
            parsed = json.loads(text)
            route = parsed.get("route", "CHAT") if isinstance(parsed, dict) else "CHAT"
            return RouteDecision(
                route=route if route in options else "CHAT",
                confidence=parsed.get("confidence", 0.4),
                reason=parsed.get("reason", "fallback_json")
            )
        except json.JSONDecodeError:
            log.warning(f"JSON 解析失败: {text}")
            return RouteDecision(route="CHAT", confidence=0.4, reason="fallback_text")


def _supervisor_node(state: GraphState, model: BaseChatModel, config: RunnableConfig) -> dict:
    messages = state.get("messages", [])
    if not messages: return {"next": "CHAT", "routing_confidence": 0.0}

    user_text = _latest_human_message(messages)
    previous_agent = _previous_agent_name(messages)

    if _is_short_followup(user_text) and previous_agent:
        return {"next": previous_agent, "routing_confidence": 0.9, "routing_reason": "短追问保留"}

    kw_dec = _keyword_route(user_text)
    if kw_dec and kw_dec.confidence >= 0.85:
        return {"next": kw_dec.route, "routing_confidence": kw_dec.confidence, "routing_reason": kw_dec.reason}

    llm_dec = _llm_route(messages, model, config)
    next_agent = llm_dec.route
    if next_agent not in {"CHAT", "FINISH"} and llm_dec.confidence < 0.45:
        next_agent = "CHAT"

    return {"next": next_agent, "routing_confidence": llm_dec.confidence, "routing_reason": llm_dec.reason}


def _agent_node(state: GraphState, agent_name: str, model: BaseChatModel, config: RunnableConfig):
    req = AgentRequest(
        user_input=_latest_human_message(state.get("messages", [])),
        model=model, session_id=state.get("session_id") or "",
        subgraph_id=agent_name, llm_config=state.get("llm_config") or {},
    )
    agent_instance = agent_classes[agent_name].cls(req)

    final_response = None
    # 【核心修复】把外层的 config 传进 run() 里，打通全链路 Trace
    for event in agent_instance.run(req, config=config):
        if not isinstance(event, dict): continue
        for node_val in event.values():
            if isinstance(node_val, dict) and "messages" in node_val:
                msgs = node_val["messages"]
                if msgs and isinstance(msgs[-1], AIMessage):
                    final_response = msgs[-1]

    if final_response:
        # Pydantic V2 下安全复制状态，避免 mutated 错误
        safe_response = AIMessage(
            content=final_response.content, name=agent_name,
            tool_calls=getattr(final_response, "tool_calls", []),
            response_metadata=getattr(final_response, "response_metadata", {})
        )
        return {"messages": [safe_response]}
    return {"messages": [AIMessage(content=f"{agent_name} 未生成有效回复。", name=agent_name)]}


def _chat_node(state: GraphState, model: BaseChatModel, config: RunnableConfig):
    prompt = "你是兜底聊天助手。不确定时请直言，禁止编造。"
    response = model.invoke([("system", prompt)] + state.get("messages", []), config=config)
    return {"messages": [AIMessage(content=response.content, name="ChatAgent")]}


def create_graph(model_config: Optional[dict] = None):
    model, _ = create_model_from_config ** (model_config or {})
    workflow = StateGraph(GraphState)

    workflow.add_node("supervisor", functools.partial(_supervisor_node, model=model))
    workflow.add_node("chat_node", functools.partial(_chat_node, model=model))

    for name in MEMBERS:
        workflow.add_node(name, functools.partial(_agent_node, agent_name=name, model=model))

    workflow.add_edge(START, "supervisor")

    conditional_map = {name: name for name in MEMBERS}
    conditional_map["CHAT"] = "chat_node"
    conditional_map["FINISH"] = END

    workflow.add_conditional_edges("supervisor", lambda state: state.get("next", "CHAT"), conditional_map)

    for name in MEMBERS: workflow.add_edge(name, END)
    workflow.add_edge("chat_node", END)

    return workflow.compile(checkpointer=checkpointer)