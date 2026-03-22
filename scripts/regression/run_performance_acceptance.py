#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提速验收脚本（面向 P0/P1/P2 的“可感知速度”指标）。

覆盖三类指标：
1. HTTP SSE 首包/首字时延（p50/p95）
2. thinking 先到但正文长时间空白的 gap 检测
3. 非 [RESUME] 场景审批查询探针（验证每轮减少至少 1 次 DB 热路径查询）

支持 baseline 对比：
- 传入 --baseline-json 后，会自动计算首包 p95 改善率并按阈值判定（默认 >=30%）。

运行示例：
    uv run python scripts/regression/run_performance_acceptance.py --skip-http
    uv run python scripts/regression/run_performance_acceptance.py \
      --base-url http://127.0.0.1:8000/api/v1 \
      --rounds 20 \
      --baseline-json docs/regression/perf_acceptance_latest.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional
from unittest.mock import patch

import requests

from agent.graph_runner import GraphRunner


def _consume_stream_events(stream_obj: Any) -> List[str]:
    """兼容消费同步/异步事件流并返回完整列表。"""
    if hasattr(stream_obj, "__aiter__"):
        async def _collect_async() -> List[str]:
            items: List[str] = []
            async for item in stream_obj:
                items.append(item)
            return items
        return asyncio.run(_collect_async())
    return list(stream_obj)


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


def _safe_percentile_nullable(values: List[int], percentile: float) -> Optional[int]:
    """计算分位数（空序列返回 None）。"""
    if not values:
        return None
    return _safe_percentile(values, percentile)


def _as_int(value: Optional[int], default: int) -> int:
    """安全 int 转换。"""
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _as_int_or_none(value: Any) -> Optional[int]:
    """安全转换为 int，失败返回 None。"""
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _parse_sse_events(resp: requests.Response) -> Iterable[Dict[str, Any]]:
    """把 SSE 行流解析成事件字典。"""
    current_event = ""
    data_lines: List[str] = []

    def _flush() -> Optional[Dict[str, Any]]:
        if not data_lines:
            return None
        payload_str = "\n".join(data_lines).strip()
        payload_obj: Dict[str, Any]
        try:
            loaded = json.loads(payload_str) if payload_str else {}
            payload_obj = loaded if isinstance(loaded, dict) else {"raw": loaded}
        except Exception:
            payload_obj = {"raw": payload_str}
        event_type = str(payload_obj.get("type") or current_event or "").strip()
        return {
            "event": current_event,
            "type": event_type,
            "payload": payload_obj,
        }

    for raw_line in resp.iter_lines(decode_unicode=True):
        line = (raw_line or "").rstrip("\r")
        if line == "":
            flushed = _flush()
            if flushed is not None:
                yield flushed
            current_event = ""
            data_lines = []
            continue

        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
            continue

    flushed = _flush()
    if flushed is not None:
        yield flushed


@dataclass
class HttpProbeRun:
    """单次 HTTP SSE 压测记录。"""

    round_idx: int
    session_id: str
    status: str
    first_packet_ms: int
    first_response_start_ms: int
    first_thinking_ms: int
    first_stream_ms: int
    thinking_to_stream_gap_ms: int
    final_event_type: str
    final_preview: str
    error: str


def _run_single_http_probe(
    *,
    base_url: str,
    timeout_sec: int,
    user_input: str,
    round_idx: int,
    token: str,
    endpoint_path: str,
) -> HttpProbeRun:
    """执行单次 HTTP SSE 请求并测时。"""
    session_id = f"perf-{round_idx}-{int(time.time() * 1000)}"
    started_at = time.perf_counter()
    first_response_start_ms: Optional[int] = None
    first_thinking_ms: Optional[int] = None
    first_stream_ms: Optional[int] = None
    first_packet_ms: Optional[int] = None
    final_event_type = ""
    final_preview = ""
    error_msg = ""
    status = "ok"

    headers = {"Accept": "text/event-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {"user_input": user_input, "session_id": session_id}
    url = f"{base_url.rstrip('/')}/{endpoint_path.lstrip('/')}"

    try:
        with requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=(3, timeout_sec),
            stream=True,
        ) as resp:
            resp.raise_for_status()
            for event in _parse_sse_events(resp):
                now_ms = int((time.perf_counter() - started_at) * 1000)
                event_type = str(event.get("type") or "")
                payload_obj = event.get("payload") or {}
                content = str(payload_obj.get("content") or "")

                if first_packet_ms is None:
                    first_packet_ms = now_ms
                if first_response_start_ms is None and event_type == "response_start":
                    first_response_start_ms = now_ms
                if first_thinking_ms is None and event_type in {"thinking", "log"}:
                    first_thinking_ms = now_ms
                if first_stream_ms is None and event_type in {"stream", "error"}:
                    first_stream_ms = now_ms

                final_event_type = event_type or final_event_type
                if content:
                    final_preview = content[:180]

                if event_type in {"response_end", "error"}:
                    if event_type == "error":
                        status = "error"
                    break
    except Exception as exc:
        status = "exception"
        error_msg = str(exc)
        if first_packet_ms is None:
            first_packet_ms = int((time.perf_counter() - started_at) * 1000)

    resolved_thinking = _as_int(first_thinking_ms, -1)
    resolved_stream = _as_int(first_stream_ms, -1)
    thinking_gap = -1
    if resolved_thinking >= 0 and resolved_stream >= 0:
        thinking_gap = max(0, resolved_stream - resolved_thinking)

    return HttpProbeRun(
        round_idx=round_idx,
        session_id=session_id,
        status=status,
        first_packet_ms=_as_int(first_packet_ms, 0),
        first_response_start_ms=_as_int(first_response_start_ms, -1),
        first_thinking_ms=resolved_thinking,
        first_stream_ms=resolved_stream,
        thinking_to_stream_gap_ms=thinking_gap,
        final_event_type=final_event_type,
        final_preview=final_preview,
        error=error_msg,
    )


def run_http_probe(
    *,
    base_url: str,
    rounds: int,
    timeout_sec: int,
    user_input: str,
    token: str,
    endpoint_path: str,
    thinking_gap_alarm_ms: int,
) -> Dict[str, Any]:
    """批量执行 HTTP SSE 压测。"""
    runs: List[HttpProbeRun] = []
    for i in range(rounds):
        runs.append(
            _run_single_http_probe(
                base_url=base_url,
                timeout_sec=timeout_sec,
                user_input=user_input,
                round_idx=i + 1,
                token=token,
                endpoint_path=endpoint_path,
            )
        )

    packet_values = [x.first_packet_ms for x in runs if x.first_packet_ms >= 0]
    start_values = [x.first_response_start_ms for x in runs if x.first_response_start_ms >= 0]
    stream_values = [x.first_stream_ms for x in runs if x.first_stream_ms >= 0]
    gap_values = [x.thinking_to_stream_gap_ms for x in runs if x.thinking_to_stream_gap_ms >= 0]

    long_gap_runs = [
        x for x in runs
        if x.thinking_to_stream_gap_ms >= 0 and x.thinking_to_stream_gap_ms >= thinking_gap_alarm_ms
    ]
    ok_count = sum(1 for x in runs if x.status == "ok")
    stream_sample_count = len(stream_values)
    packet_sample_count = len(packet_values)
    return {
        "enabled": True,
        "endpoint": endpoint_path,
        "rounds": rounds,
        "success_rate": round(ok_count / max(rounds, 1), 4),
        "packet_sample_count": packet_sample_count,
        "stream_sample_count": stream_sample_count,
        "latency_ms": {
            "first_packet_p50": _safe_percentile_nullable(packet_values, 50),
            "first_packet_p95": _safe_percentile_nullable(packet_values, 95),
            "response_start_p50": _safe_percentile_nullable(start_values, 50),
            "response_start_p95": _safe_percentile_nullable(start_values, 95),
            "first_stream_p50": _safe_percentile_nullable(stream_values, 50),
            "first_stream_p95": _safe_percentile_nullable(stream_values, 95),
            "thinking_to_stream_gap_p95": _safe_percentile_nullable(gap_values, 95),
        },
        "thinking_gap_alarm_ms": thinking_gap_alarm_ms,
        "thinking_gap_alarm_count": len(long_gap_runs),
        "thinking_gap_alarm_top5": [asdict(x) for x in long_gap_runs[:5]],
        "details": [asdict(x) for x in runs],
    }


class _FakeGraph:
    """审批查询探针用假图对象。"""

    def stream(self, _inputs, config=None, stream_mode=None):
        return []

    def get_state(self, _config):
        return SimpleNamespace(values={"direct_answer": "ok"})


def run_approval_query_probe(
    *,
    non_resume_rounds: int,
    resume_rounds: int,
    legacy_calls_per_non_resume_turn: int,
) -> Dict[str, Any]:
    """
    统计审批查询热路径调用次数：
    - 非 RESUME：期望 0 次
    - RESUME：期望 = resume_rounds（仍然可恢复）
    """
    runner = GraphRunner()
    call_counter = {"count": 0}

    def _fake_fetch_latest(_session_id: str) -> Optional[Dict[str, Any]]:
        call_counter["count"] += 1
        return None

    with (
        patch.object(runner, "_get_or_create_supervisor", return_value=_FakeGraph()),
        patch("agent.graph_runner.interrupt_service.fetch_latest_resolved_approval", side_effect=_fake_fetch_latest),
        patch("agent.graph_runner.rule_registry.get_rules", return_value=[]),
    ):
        for i in range(non_resume_rounds):
            _ = _consume_stream_events(
                runner.stream_run(
                    "hello",
                    f"probe-non-resume-{i}",
                    model_config={},
                    emit_response_start=False,
                )
            )
        non_resume_calls = int(call_counter["count"])

        for i in range(resume_rounds):
            _ = _consume_stream_events(
                runner.stream_run(
                    "[RESUME]",
                    f"probe-resume-{i}",
                    model_config={},
                    emit_response_start=False,
                )
            )
        total_calls = int(call_counter["count"])

    resume_calls = max(0, total_calls - non_resume_calls)
    legacy_expected = max(0, non_resume_rounds * legacy_calls_per_non_resume_turn)
    reduced_calls = max(0, legacy_expected - non_resume_calls)

    return {
        "non_resume_rounds": non_resume_rounds,
        "resume_rounds": resume_rounds,
        "legacy_calls_per_non_resume_turn": legacy_calls_per_non_resume_turn,
        "observed_non_resume_calls": non_resume_calls,
        "observed_resume_calls": resume_calls,
        "legacy_expected_non_resume_calls": legacy_expected,
        "reduced_calls_vs_legacy": reduced_calls,
        "pass_non_resume_zero_query": non_resume_calls == 0,
        "pass_resume_still_queries": resume_calls == resume_rounds,
    }


def _load_baseline(path_text: str) -> Optional[Dict[str, Any]]:
    """读取 baseline JSON。"""
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _build_baseline_comparison(
    *,
    current_http_probe: Dict[str, Any],
    baseline_data: Optional[Dict[str, Any]],
    expected_first_packet_improve_ratio: float,
) -> Dict[str, Any]:
    """基于 baseline 计算首包 p95 改善率。"""
    if not baseline_data:
        return {
            "has_baseline": False,
            "first_packet_p95_improve_ratio": None,
            "expected_first_packet_improve_ratio": expected_first_packet_improve_ratio,
            "pass_first_packet_improve_gate": None,
        }

    baseline_http = baseline_data.get("http_probe") or {}
    baseline_latency = baseline_http.get("latency_ms") or {}
    current_latency = current_http_probe.get("latency_ms") or {}

    baseline_p95 = _as_int_or_none(baseline_latency.get("first_packet_p95"))
    current_p95 = _as_int_or_none(current_latency.get("first_packet_p95"))
    if baseline_p95 is None or current_p95 is None or baseline_p95 <= 0:
        improve_ratio = None
        pass_gate = None
    else:
        improve_ratio = round((baseline_p95 - current_p95) / baseline_p95, 4)
        pass_gate = bool(improve_ratio >= expected_first_packet_improve_ratio)

    return {
        "has_baseline": True,
        "baseline_first_packet_p95": baseline_p95,
        "current_first_packet_p95": current_p95,
        "first_packet_p95_improve_ratio": improve_ratio,
        "expected_first_packet_improve_ratio": expected_first_packet_improve_ratio,
        "pass_first_packet_improve_gate": pass_gate,
    }


def _render_markdown_report(result: Dict[str, Any]) -> str:
    """渲染 Markdown 报告。"""
    generated_at = result.get("generated_at")
    http_probe = result.get("http_probe") or {}
    approval_probe = result.get("approval_query_probe") or {}
    baseline_cmp = result.get("baseline_comparison") or {}
    gates = result.get("acceptance_gates") or {}

    lines: List[str] = []
    def _fmt_ms(value: Any) -> str:
        return "N/A" if value is None else f"{value}ms"

    lines.append("# 提速验收报告（P0/P1/P2）")
    lines.append("")
    lines.append(f"- 生成时间：{generated_at}")
    lines.append("")
    lines.append("## 1. HTTP SSE 指标")
    lines.append("")
    if not http_probe.get("enabled"):
        lines.append(f"- 已跳过：{http_probe.get('skip_reason', 'unknown')}")
    else:
        latency = http_probe.get("latency_ms") or {}
        lines.append(f"- 轮数：`{http_probe.get('rounds')}`，成功率：`{http_probe.get('success_rate')}`")
        lines.append(
            f"- 首包耗时：p50=`{_fmt_ms(latency.get('first_packet_p50'))}` / p95=`{_fmt_ms(latency.get('first_packet_p95'))}`"
        )
        lines.append(
            f"- 首字耗时：p50=`{_fmt_ms(latency.get('first_stream_p50'))}` / p95=`{_fmt_ms(latency.get('first_stream_p95'))}`"
        )
        lines.append(
            f"- thinking->stream gap p95：`{_fmt_ms(latency.get('thinking_to_stream_gap_p95'))}`，"
            f"超阈值次数：`{http_probe.get('thinking_gap_alarm_count')}`"
        )

    lines.append("")
    lines.append("## 2. 审批查询热路径探针")
    lines.append("")
    lines.append(
        f"- 非 RESUME 调用次数：`{approval_probe.get('observed_non_resume_calls')}` "
        f"(轮数=`{approval_probe.get('non_resume_rounds')}`)"
    )
    lines.append(
        f"- RESUME 调用次数：`{approval_probe.get('observed_resume_calls')}` "
        f"(轮数=`{approval_probe.get('resume_rounds')}`)"
    )
    lines.append(
        f"- 相比旧模型预期减少：`{approval_probe.get('reduced_calls_vs_legacy')}` 次"
    )

    lines.append("")
    lines.append("## 3. Baseline 对比")
    lines.append("")
    if not baseline_cmp.get("has_baseline"):
        lines.append("- 未提供 baseline，未做首包下降比例判定。")
    else:
        lines.append(
            f"- 首包 p95：baseline=`{baseline_cmp.get('baseline_first_packet_p95')}ms`，"
            f"current=`{baseline_cmp.get('current_first_packet_p95')}ms`，"
            f"improve=`{baseline_cmp.get('first_packet_p95_improve_ratio')}`"
        )
        lines.append(
            f"- 门槛：`{baseline_cmp.get('expected_first_packet_improve_ratio')}`，"
            f"通过：`{baseline_cmp.get('pass_first_packet_improve_gate')}`"
        )

    lines.append("")
    lines.append("## 4. 验收门槛")
    lines.append("")
    lines.append(f"- 非 RESUME 不触发审批查询：`{gates.get('pass_non_resume_zero_query')}`")
    lines.append(f"- RESUME 行为保留：`{gates.get('pass_resume_still_queries')}`")
    lines.append(f"- thinking 有但正文长时间空白：`{gates.get('pass_thinking_gap')}`")
    lines.append(f"- 首包 p95 降幅门槛：`{gates.get('pass_first_packet_improve_gate')}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="性能验收脚本（SSE + 审批查询热路径）")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api/v1", help="API 基地址")
    parser.add_argument("--endpoint-path", default="/chat/stream/anonymous", help="压测接口路径")
    parser.add_argument("--token", default="", help="可选 Bearer Token")
    parser.add_argument("--user-input", default="你好，简单自我介绍一下。", help="压测输入")
    parser.add_argument("--rounds", type=int, default=20, help="HTTP 压测轮数")
    parser.add_argument("--timeout-sec", type=int, default=60, help="单轮请求超时（秒）")
    parser.add_argument("--thinking-gap-alarm-ms", type=int, default=5000, help="thinking->stream 告警阈值（毫秒）")
    parser.add_argument("--skip-http", action="store_true", help="仅执行审批查询探针，跳过 HTTP 压测")
    parser.add_argument("--non-resume-rounds", type=int, default=30, help="非 RESUME 探针轮数")
    parser.add_argument("--resume-rounds", type=int, default=10, help="RESUME 探针轮数")
    parser.add_argument(
        "--legacy-calls-per-non-resume-turn",
        type=int,
        default=1,
        help="旧版本每轮非 RESUME 预期审批查询次数（默认 1）",
    )
    parser.add_argument("--baseline-json", default="", help="baseline 报告 JSON 路径")
    parser.add_argument(
        "--expected-first-packet-improve-ratio",
        type=float,
        default=0.30,
        help="首包 p95 改善率门槛（默认 0.30）",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    docs_regression_dir = project_root / "docs" / "regression"
    docs_regression_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_http:
        http_probe: Dict[str, Any] = {
            "enabled": False,
            "skip_reason": "--skip-http enabled",
        }
    else:
        http_probe = run_http_probe(
            base_url=args.base_url,
            rounds=max(1, args.rounds),
            timeout_sec=max(3, args.timeout_sec),
            user_input=args.user_input,
            token=args.token,
            endpoint_path=args.endpoint_path,
            thinking_gap_alarm_ms=max(0, args.thinking_gap_alarm_ms),
        )

    approval_probe = run_approval_query_probe(
        non_resume_rounds=max(1, args.non_resume_rounds),
        resume_rounds=max(1, args.resume_rounds),
        legacy_calls_per_non_resume_turn=max(0, args.legacy_calls_per_non_resume_turn),
    )

    baseline_data = _load_baseline(args.baseline_json)
    baseline_cmp = _build_baseline_comparison(
        current_http_probe=http_probe,
        baseline_data=baseline_data,
        expected_first_packet_improve_ratio=float(args.expected_first_packet_improve_ratio),
    )

    pass_thinking_gap = None
    if http_probe.get("enabled"):
        if float(http_probe.get("success_rate") or 0.0) > 0:
            pass_thinking_gap = int(http_probe.get("thinking_gap_alarm_count") or 0) == 0

    acceptance_gates = {
        "pass_non_resume_zero_query": bool(approval_probe.get("pass_non_resume_zero_query")),
        "pass_resume_still_queries": bool(approval_probe.get("pass_resume_still_queries")),
        "pass_thinking_gap": pass_thinking_gap,
        "pass_first_packet_improve_gate": baseline_cmp.get("pass_first_packet_improve_gate"),
    }

    result: Dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "http_probe": http_probe,
        "approval_query_probe": approval_probe,
        "baseline_comparison": baseline_cmp,
        "acceptance_gates": acceptance_gates,
        "args": vars(args),
    }

    json_path = docs_regression_dir / "perf_acceptance_latest.json"
    md_path = docs_regression_dir / "perf_acceptance_latest.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown_report(result), encoding="utf-8")

    print(f"[OK] JSON 报告: {json_path}")
    print(f"[OK] Markdown 报告: {md_path}")
    print(
        "[SUMMARY] "
        f"non_resume_zero={acceptance_gates['pass_non_resume_zero_query']} "
        f"resume_ok={acceptance_gates['pass_resume_still_queries']} "
        f"thinking_gap_ok={acceptance_gates['pass_thinking_gap']} "
        f"first_packet_gate={acceptance_gates['pass_first_packet_improve_gate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
