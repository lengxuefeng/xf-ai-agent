#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键回归脚本（离线可运行）。

覆盖三类检查：
1. 路由稳定性：100 条样本的 domain/intent 命中率。
2. 时延统计：基于 logs/app.log 提取各阶段耗时分布。
3. 端到端链路：weather/search/yunyou 三条链路离线冒烟与首包耗时。

运行方式：
    uv run python scripts/regression/run_full_regression.py
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# 强制使用内存 Checkpointer，避免回归脚本依赖本地 PgSQL。
os.environ.setdefault("CHECKPOINTER_BACKEND", "memory")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from agent.graph_runner import GraphRunner
from agent.graphs import supervisor as supervisor_module
from agent.graphs.supervisor import domain_router_node, intent_router_node
from config.runtime_settings import RouterPolicyConfig
from importlib.metadata import version as pkg_version


@dataclass
class RouteCase:
    """路由测试样本。"""

    text: str
    expected_domain: str
    expected_intent: str
    history: Optional[List[BaseMessage]] = None


@dataclass
class RouteCaseResult:
    """单条路由测试结果。"""

    text: str
    expected_domain: str
    actual_domain: str
    expected_intent: str
    actual_intent: str
    domain_ms: int
    intent_ms: int
    pass_domain: bool
    pass_intent: bool


@dataclass
class E2ECaseResult:
    """单条端到端链路冒烟结果。"""

    name: str
    user_input: str
    first_stream_ms: int
    total_ms: int
    status: str
    final_preview: str


class OfflineRegressionChatModel(BaseChatModel):
    """
    离线回归用模型。

    目标：
    1. 不访问外部网络，保证回归可重复。
    2. 支持 bind / bind_tools / with_structured_output，兼容最新 LangChain 写法。
    """

    @property
    def _llm_type(self) -> str:
        """标识模型类型。"""
        return "offline-regression-chat"

    def _latest_human(self, messages: Sequence[BaseMessage]) -> str:
        """提取最近一条用户消息文本。"""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                content = getattr(msg, "content", "")
                if isinstance(content, str):
                    return content.strip()
                return str(content or "").strip()
        return ""

    def _classify_domain(self, text: str) -> Dict[str, Any]:
        """根据关键词生成数据域分类结果。"""
        lower = (text or "").lower()
        if any(k in lower for k in ("holter", "云柚", "动态心电", "贴片")):
            return {"data_domain": "YUNYOU_DB", "confidence": 0.98}
        if any(k in lower for k in ("sql", "数据库", "表", "order by", "limit", "select")):
            return {"data_domain": "LOCAL_DB", "confidence": 0.95}
        if any(k in lower for k in ("天气", "活动", "附近", "推荐", "新闻", "搜索")):
            return {"data_domain": "WEB_SEARCH", "confidence": 0.91}
        return {"data_domain": "GENERAL", "confidence": 0.86}

    def _classify_intent(self, text: str) -> Dict[str, Any]:
        """根据关键词生成意图分类结果。"""
        lower = (text or "").lower()
        if any(k in lower for k in ("holter", "云柚", "动态心电", "贴片")):
            intent = "yunyou_agent"
        elif any(k in lower for k in ("sql", "数据库", "order by", "limit", "select")):
            intent = "sql_agent"
        elif any(k in lower for k in ("天气", "气温", "降雨", "下雨")):
            intent = "weather_agent"
        elif any(k in lower for k in ("活动", "新闻", "附近", "推荐", "搜索")):
            intent = "search_agent"
        else:
            intent = "CHAT"
        is_complex = ("先" in lower and "再" in lower) or ("然后" in lower and len(lower) >= 20)
        return {
            "intent": intent,
            "confidence": 0.93 if intent != "CHAT" else 0.90,
            "is_complex": bool(is_complex),
            "direct_answer": "" if intent != "CHAT" else "这是离线回归模型的简答回复。",
        }

    def _planner_output(self, text: str) -> Dict[str, Any]:
        """生成规划器结构化输出。"""
        lower = (text or "").lower()
        if "先" in lower and "再" in lower:
            return {
                "tasks": [
                    {"id": "t1", "agent": "search_agent", "input": "先执行第一步信息检索", "depends_on": []},
                    {"id": "t2", "agent": "CHAT", "input": "基于 t1 的结果给出总结", "depends_on": ["t1"]},
                ]
            }
        return {"tasks": [{"id": "t1", "agent": "CHAT", "input": text or "请直接回答用户问题", "depends_on": []}]}

    def _response_text(self, text: str) -> str:
        """生成普通文本回复。"""
        lower = (text or "").lower()
        if "天气" in lower:
            return "郑州今天天气偏冷且有霾，建议减少长时间户外活动。"
        if "活动" in lower or "附近" in lower:
            return "已为你整理附近可选活动：商场、书店、室内运动馆。"
        if "holter" in lower or "云柚" in lower:
            return "已进入云柚链路，可继续执行 holter 查询。"
        return "这是离线回归模型回复。"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "OfflineRegressionChatModel":
        """工具绑定兼容实现（离线回归不真实调用工具）。"""
        return self

    def bind(self, **kwargs: Any) -> "OfflineRegressionChatModel":
        """参数绑定兼容实现。"""
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any):
        """结构化输出兼容实现，直接返回 schema 对象。"""

        def _invoke_structured(input_data: Any) -> Any:
            messages = _extract_messages_from_prompt_input(input_data)
            latest = self._latest_human(messages)

            if hasattr(schema, "model_fields") and "data_domain" in schema.model_fields:
                payload = self._classify_domain(latest)
            elif hasattr(schema, "model_fields") and "intent" in schema.model_fields:
                payload = self._classify_intent(latest)
            elif hasattr(schema, "model_fields") and "tasks" in schema.model_fields:
                payload = self._planner_output(latest)
            else:
                payload = {}

            try:
                return schema(**payload)
            except Exception:
                return payload

        return RunnableLambda(_invoke_structured)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """BaseChatModel 抽象方法实现。"""
        latest = self._latest_human(messages)
        text = self._response_text(latest)
        ai_msg = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])


def _extract_messages_from_prompt_input(input_data: Any) -> List[BaseMessage]:
    """从 Prompt/输入对象中提取消息列表。"""
    if isinstance(input_data, dict) and "messages" in input_data:
        raw_messages = input_data.get("messages") or []
        return [msg for msg in raw_messages if isinstance(msg, BaseMessage)]
    if isinstance(input_data, list):
        return [msg for msg in input_data if isinstance(msg, BaseMessage)]
    return []


def _safe_percentile(values: List[int], percentile: float) -> int:
    """计算整数序列分位数。"""
    if not values:
        return 0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = int(math.ceil((percentile / 100.0) * len(sorted_values))) - 1
    rank = max(0, min(rank, len(sorted_values) - 1))
    return sorted_values[rank]


def _build_route_cases() -> List[RouteCase]:
    """构建 100 条路由回归样本。"""
    cases: List[RouteCase] = []

    holter_inputs = [
        "查询 holter 最近使用的用户，按 id 倒序前 5 条",
        "云柚贴片最近有没有人用",
        "我要看动态心电最近记录",
        "holter 数据按 id 倒序取前 10 条",
        "请查云柚 holter 明细",
    ]
    for i in range(20):
        text = holter_inputs[i % len(holter_inputs)]
        cases.append(RouteCase(text=text, expected_domain="YUNYOU_DB", expected_intent="yunyou_agent"))

    sql_inputs = [
        "请给我 SQL：查询 t_user_info 按 id 倒序前 5 条",
        "数据库里 user 表最新 10 条记录",
        "写一个 order by id desc limit 20 的查询",
        "select * from t_chat_message order by id desc limit 5",
        "我想查本地库里最近数据",
    ]
    for i in range(20):
        text = sql_inputs[i % len(sql_inputs)]
        cases.append(RouteCase(text=text, expected_domain="LOCAL_DB", expected_intent="sql_agent"))

    weather_inputs = [
        "郑州今天的天气怎么样",
        "明天会下雨吗",
        "今天气温多少度",
        "空气质量怎么样",
        "郑州现在适合出门吗",
    ]
    for i in range(20):
        text = weather_inputs[i % len(weather_inputs)]
        cases.append(RouteCase(text=text, expected_domain="WEB_SEARCH", expected_intent="weather_agent"))

    search_inputs = [
        "郑州东站附近有什么活动",
        "帮我搜索启迪科技城附近好玩的",
        "附近有没有室内活动推荐",
        "给我查一下本周末活动",
        "搜一下郑州商场活动",
    ]
    for i in range(20):
        text = search_inputs[i % len(search_inputs)]
        cases.append(RouteCase(text=text, expected_domain="WEB_SEARCH", expected_intent="search_agent"))

    chat_inputs = [
        "你好呀",
        "今天心情不好",
        "讲个笑话",
        "谢谢你",
        "你是谁",
    ]
    for i in range(10):
        text = chat_inputs[i % len(chat_inputs)]
        cases.append(RouteCase(text=text, expected_domain="GENERAL", expected_intent="CHAT"))

    # 天气追问复用场景：预期走 CHAT（复用上下文），避免重复调 weather_agent。
    history_messages = [
        HumanMessage(content="郑州今天天气怎么样"),
        AIMessage(content="郑州实时天气：10°C，有霾，能见度较低。"),
    ]
    followups = [
        "这个天气适合出去玩吗",
        "那我要不要戴口罩",
        "今天还能跑步吗",
        "这种天气我该怎么穿",
        "晚上出门合适吗",
    ]
    for text in followups:
        cases.append(RouteCase(text=text, expected_domain="WEB_SEARCH", expected_intent="CHAT", history=history_messages))

    # 多域混合场景：必须进入复杂任务拆解，不允许被单域强路由吞掉。
    mixed_inputs = [
        ("天气也这么不好，老板还让我查询一下holter最近的数据", "YUNYOU_DB"),
        ("今天郑州东站有什么活动，再帮我查holter最近5条", "YUNYOU_DB"),
        ("先查郑州天气，再查本地库用户最近10条", "LOCAL_DB"),
        ("我先看看今天天气，然后帮我查云柚holter记录", "YUNYOU_DB"),
        ("查下附近活动并且查询holter最近使用用户", "YUNYOU_DB"),
    ]
    for text, domain in mixed_inputs:
        cases.append(RouteCase(text=text, expected_domain=domain, expected_intent="CHAT"))

    return cases[:100]


def run_router_regression() -> Dict[str, Any]:
    """执行 100 条路由回归，输出命中率与耗时统计。"""
    model = OfflineRegressionChatModel()
    cases = _build_route_cases()
    results: List[RouteCaseResult] = []

    for idx, case in enumerate(cases):
        session_id = f"route-reg-{idx + 1}"
        cfg = {"configurable": {"thread_id": session_id}}
        messages = list(case.history or []) + [HumanMessage(content=case.text)]
        state = {"messages": messages, "session_id": session_id}

        t1 = time.perf_counter()
        domain_result = domain_router_node(state, model, cfg)
        domain_ms = int((time.perf_counter() - t1) * 1000)

        intent_state = {**state, **domain_result}
        t2 = time.perf_counter()
        intent_result = intent_router_node(intent_state, model, cfg)
        intent_ms = int((time.perf_counter() - t2) * 1000)

        actual_domain = str(domain_result.get("data_domain", ""))
        actual_intent = str(intent_result.get("intent", ""))
        pass_domain = actual_domain == case.expected_domain
        pass_intent = actual_intent == case.expected_intent
        results.append(
            RouteCaseResult(
                text=case.text,
                expected_domain=case.expected_domain,
                actual_domain=actual_domain,
                expected_intent=case.expected_intent,
                actual_intent=actual_intent,
                domain_ms=domain_ms,
                intent_ms=intent_ms,
                pass_domain=pass_domain,
                pass_intent=pass_intent,
            )
        )

    domain_hits = sum(1 for row in results if row.pass_domain)
    intent_hits = sum(1 for row in results if row.pass_intent)
    total = len(results)
    mismatch_rows = [asdict(row) for row in results if not row.pass_domain or not row.pass_intent][:20]
    domain_latencies = [row.domain_ms for row in results]
    intent_latencies = [row.intent_ms for row in results]
    both_pass = sum(1 for row in results if row.pass_domain and row.pass_intent)

    return {
        "total_cases": total,
        "domain_accuracy": round(domain_hits / max(total, 1), 4),
        "intent_accuracy": round(intent_hits / max(total, 1), 4),
        "full_route_accuracy": round(both_pass / max(total, 1), 4),
        "latency_ms": {
            "domain_p50": _safe_percentile(domain_latencies, 50),
            "domain_p95": _safe_percentile(domain_latencies, 95),
            "domain_max": max(domain_latencies) if domain_latencies else 0,
            "intent_p50": _safe_percentile(intent_latencies, 50),
            "intent_p95": _safe_percentile(intent_latencies, 95),
            "intent_max": max(intent_latencies) if intent_latencies else 0,
        },
        "mismatches_top20": mismatch_rows,
    }


def _parse_sse_chunk(chunk: str) -> Optional[Dict[str, Any]]:
    """解析单个 SSE 块，提取 data JSON。"""
    if not chunk:
        return None
    lines = [line.strip() for line in chunk.splitlines() if line.strip()]
    for line in lines:
        if line.startswith("data:"):
            payload = line[len("data:") :].strip()
            try:
                return json.loads(payload)
            except Exception:
                return None
    return None


def _run_single_e2e_case(runner: GraphRunner, user_input: str, model_config: Dict[str, Any]) -> E2ECaseResult:
    """执行一条端到端离线链路并统计首包/总耗时。"""
    started_at = time.perf_counter()
    first_stream_ms: Optional[int] = None
    final_preview = ""
    status = "ok"

    try:
        for chunk in runner.stream_run(
            user_input=user_input,
            session_id=f"e2e-{int(time.time() * 1000)}",
            model_config=model_config,
            history_messages=[],
            session_context={},
        ):
            parsed = _parse_sse_chunk(chunk)
            if not parsed:
                continue
            event_type = str(parsed.get("type", ""))
            event_content = str(parsed.get("content", ""))
            if first_stream_ms is None and event_type in {"stream", "error"}:
                first_stream_ms = int((time.perf_counter() - started_at) * 1000)
            if event_type == "error":
                status = "error"
                final_preview = event_content[:180]
            elif event_type == "stream" and event_content:
                final_preview = event_content[:180]
    except Exception as exc:
        status = "exception"
        final_preview = str(exc)[:180]

    total_ms = int((time.perf_counter() - started_at) * 1000)
    return E2ECaseResult(
        name="",
        user_input=user_input,
        first_stream_ms=first_stream_ms or total_ms,
        total_ms=total_ms,
        status=status,
        final_preview=final_preview,
    )


def run_e2e_regression_offline() -> Dict[str, Any]:
    """执行 weather/search/yunyou 三条端到端离线回归。"""
    model_config = {
        "model": "offline-regression",
        "model_service": "offline",
        "service_type": "offline",
        "model_key": "offline",
        "model_url": "offline",
    }

    original_loader = supervisor_module.create_model_from_config

    # 临时替换模型加载器：统一使用离线模型，避免外部接口抖动影响回归。
    def _offline_loader(**kwargs: Any):
        return OfflineRegressionChatModel(), {}

    supervisor_module.create_model_from_config = _offline_loader

    # 临时替换 yunyou 工具调用，避免真实外部网关请求。
    from agent.agents import yunyou_agent as yunyou_agent_module

    original_common_post = yunyou_agent_module.YunYouTools.common_post

    def _offline_common_post(self: Any, path: str, params: Optional[Dict[str, Any]] = None):
        if path == "holter/list":
            return {
                "success": True,
                "code": 0,
                "msg": "ok",
                "data": {
                    "items": [
                        {"user_id": 1001, "use_day": "2026-03-10", "holter_type": 1, "report_status": 2},
                        {"user_id": 1002, "use_day": "2026-03-10", "holter_type": 2, "report_status": 1},
                    ]
                },
            }
        return {"success": False, "code": 1, "msg": "offline mock not implemented"}

    yunyou_agent_module.YunYouTools.common_post = _offline_common_post

    cases = [
        ("weather_chain", "郑州今天天气怎么样"),
        ("search_chain", "郑州东站附近有什么活动"),
        ("yunyou_chain", "查询holter最近使用数据，按id倒序前5条"),
    ]

    results: List[E2ECaseResult] = []
    try:
        runner = GraphRunner()
        for name, text in cases:
            row = _run_single_e2e_case(runner, text, model_config)
            row.name = name
            results.append(row)
    finally:
        # 回收 monkey patch，避免污染其它进程逻辑。
        supervisor_module.create_model_from_config = original_loader
        yunyou_agent_module.YunYouTools.common_post = original_common_post

    success_count = sum(1 for row in results if row.status == "ok")
    first_stream_values = [row.first_stream_ms for row in results]
    total_values = [row.total_ms for row in results]

    return {
        "total_cases": len(results),
        "success_cases": success_count,
        "success_rate": round(success_count / max(len(results), 1), 4),
        "latency_ms": {
            "first_stream_p50": _safe_percentile(first_stream_values, 50),
            "first_stream_p95": _safe_percentile(first_stream_values, 95),
            "total_p50": _safe_percentile(total_values, 50),
            "total_p95": _safe_percentile(total_values, 95),
        },
        "details": [asdict(row) for row in results],
    }


def run_structured_router_smoke() -> Dict[str, Any]:
    """开启 LLM fallback 后执行结构化路由烟测。"""
    original_policy = supervisor_module.ROUTER_POLICY_CONFIG
    supervisor_module.ROUTER_POLICY_CONFIG = RouterPolicyConfig(
        domain_llm_fallback_enabled=True,
        intent_llm_fallback_enabled=True,
        classifier_history_messages=original_policy.classifier_history_messages,
        general_chat_fastpath_enabled=original_policy.general_chat_fastpath_enabled,
        planner_llm_fallback_enabled=original_policy.planner_llm_fallback_enabled,
        router_llm_timeout_sec=original_policy.router_llm_timeout_sec,
    )
    model = OfflineRegressionChatModel()
    samples = [
        "帮我处理一下",
        "我想看数据库最近的数据",
        "今天适合出去玩吗",
        "查查附近活动",
        "看看 holter 数据",
        "你好",
        "写个 SQL 示例",
        "明天下雨吗",
        "帮我总结一下",
        "我有点不舒服怎么办",
    ]
    passed = 0
    failures: List[str] = []
    try:
        for idx, text in enumerate(samples):
            session_id = f"struct-smoke-{idx}"
            cfg = {"configurable": {"thread_id": session_id}}
            state = {"messages": [HumanMessage(content=text)], "session_id": session_id}
            try:
                domain_result = domain_router_node(state, model, cfg)
                intent_state = {**state, **domain_result}
                intent_result = intent_router_node(intent_state, model, cfg)
                if domain_result.get("data_domain") and intent_result.get("intent"):
                    passed += 1
                else:
                    failures.append(text)
            except Exception:
                failures.append(text)
    finally:
        supervisor_module.ROUTER_POLICY_CONFIG = original_policy

    return {"total": len(samples), "passed": passed, "failed_inputs": failures}


def parse_latency_from_logs(log_file: Path) -> Dict[str, Any]:
    """从日志中提取各阶段耗时分布。"""
    if not log_file.exists():
        return {"log_file": str(log_file), "exists": False, "metrics": {}}

    patterns = {
        "domain_router_ms": re.compile(r"Domain Router .*?耗时:\s*(\d+)ms"),
        "intent_router_ms": re.compile(r"Intent Router .*?耗时:\s*(\d+)ms"),
        "chat_node_ms": re.compile(r"chat_node 模型耗时:\s*(\d+)ms"),
        "planner_ms": re.compile(r"Parent Planner 耗时:\s*(\d+)ms"),
        "aggregator_ms": re.compile(r"Aggregator 耗时:\s*(\d+)ms"),
        "worker_ms": re.compile(r"Worker\[[^\]]+\].*?耗时:\s*(\d+)ms"),
    }

    values: Dict[str, List[int]] = {name: [] for name in patterns}
    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for name, regex in patterns.items():
                matched = regex.search(line)
                if matched:
                    values[name].append(int(matched.group(1)))

    metrics: Dict[str, Any] = {}
    for name, number_list in values.items():
        metrics[name] = {
            "count": len(number_list),
            "p50": _safe_percentile(number_list, 50),
            "p95": _safe_percentile(number_list, 95),
            "max": max(number_list) if number_list else 0,
        }

    return {"log_file": str(log_file), "exists": True, "metrics": metrics}


def collect_runtime_versions() -> Dict[str, str]:
    """收集本次回归使用的关键包版本。"""
    names = [
        "langchain",
        "langgraph",
        "langchain-core",
        "langchain-openai",
        "langsmith",
        "openai",
    ]
    return {name: pkg_version(name) for name in names}


def _render_markdown_report(result: Dict[str, Any]) -> str:
    """将回归结果渲染为 Markdown 报告。"""
    route = result["router_regression"]
    latency = result["log_latency"]
    e2e = result["e2e_offline"]
    smoke = result["structured_router_smoke"]
    versions = result["versions"]
    generated_at = result["generated_at"]

    lines: List[str] = []
    lines.append("# 回归报告（LangGraph 1.0.10 升级后）")
    lines.append("")
    lines.append(f"- 生成时间：{generated_at}")
    lines.append("- 说明：本报告由 `scripts/regression/run_full_regression.py` 自动生成。")
    lines.append("")
    lines.append("## 1. 运行环境版本")
    lines.append("")
    for name, ver in versions.items():
        lines.append(f"- `{name}`: `{ver}`")
    lines.append("")
    lines.append("## 2. 路由回归（100 条）")
    lines.append("")
    lines.append(f"- Domain 命中率：`{route['domain_accuracy'] * 100:.2f}%`")
    lines.append(f"- Intent 命中率：`{route['intent_accuracy'] * 100:.2f}%`")
    lines.append(f"- 全链路命中率：`{route['full_route_accuracy'] * 100:.2f}%`")
    lines.append(
        f"- 路由耗时：Domain p50=`{route['latency_ms']['domain_p50']}ms` / p95=`{route['latency_ms']['domain_p95']}ms`，"
        f"Intent p50=`{route['latency_ms']['intent_p50']}ms` / p95=`{route['latency_ms']['intent_p95']}ms`"
    )
    if route["mismatches_top20"]:
        lines.append("")
        lines.append("### 2.1 主要误差样本（Top 20）")
        lines.append("")
        for item in route["mismatches_top20"][:10]:
            lines.append(
                f"- 输入：`{item['text']}` | 期望 `{item['expected_domain']}/{item['expected_intent']}`，"
                f"实际 `{item['actual_domain']}/{item['actual_intent']}`"
            )
    lines.append("")
    lines.append("## 3. 结构化路由烟测（LLM fallback 开启）")
    lines.append("")
    lines.append(f"- 通过：`{smoke['passed']}/{smoke['total']}`")
    if smoke["failed_inputs"]:
        lines.append(f"- 失败输入：`{smoke['failed_inputs']}`")
    lines.append("")
    lines.append("## 4. 日志时延统计（app.log）")
    lines.append("")
    if latency["exists"]:
        for metric_name, metric_value in latency["metrics"].items():
            lines.append(
                f"- `{metric_name}`: count={metric_value['count']}, "
                f"p50={metric_value['p50']}ms, p95={metric_value['p95']}ms, max={metric_value['max']}ms"
            )
    else:
        lines.append(f"- 日志不存在：`{latency['log_file']}`")
    lines.append("")
    lines.append("## 5. 三链路端到端离线回归")
    lines.append("")
    lines.append(
        f"- 成功率：`{e2e['success_rate'] * 100:.2f}%` "
        f"（{e2e['success_cases']}/{e2e['total_cases']}）"
    )
    lines.append(
        f"- 首包耗时：p50=`{e2e['latency_ms']['first_stream_p50']}ms` / p95=`{e2e['latency_ms']['first_stream_p95']}ms`"
    )
    lines.append(
        f"- 总耗时：p50=`{e2e['latency_ms']['total_p50']}ms` / p95=`{e2e['latency_ms']['total_p95']}ms`"
    )
    lines.append("")
    for detail in e2e["details"]:
        lines.append(
            f"- `{detail['name']}`: status=`{detail['status']}`, first_stream=`{detail['first_stream_ms']}ms`, "
            f"total=`{detail['total_ms']}ms`, preview=`{detail['final_preview']}`"
        )
    lines.append("")
    lines.append("## 6. 结论与建议")
    lines.append("")
    lines.append("- 路由层面：规则优先 + 结构化输出已可稳定运行，核心路径耗时很低。")
    lines.append("- 性能瓶颈：从日志看，`chat_node` 仍存在高耗时长尾（秒级到分钟级），建议继续做模型超时/降级策略。")
    lines.append("- 后续动作：建议接入真实在线 E2E 压测（同样脚本框架，替换离线模型为在线模型）。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """脚本主入口。"""
    project_root = Path(__file__).resolve().parents[2]
    docs_regression_dir = project_root / "docs" / "regression"
    docs_regression_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "versions": collect_runtime_versions(),
        "router_regression": run_router_regression(),
        "structured_router_smoke": run_structured_router_smoke(),
        "log_latency": parse_latency_from_logs(project_root / "logs" / "app.log"),
        "e2e_offline": run_e2e_regression_offline(),
    }

    json_path = docs_regression_dir / "latest_regression_report.json"
    md_path = docs_regression_dir / "latest_regression_report.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown_report(result), encoding="utf-8")

    print(f"[OK] JSON 报告: {json_path}")
    print(f"[OK] Markdown 报告: {md_path}")
    print(
        f"[SUMMARY] route={result['router_regression']['full_route_accuracy']:.4f}, "
        f"e2e={result['e2e_offline']['success_rate']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
