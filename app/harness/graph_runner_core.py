# -*- coding: utf-8 -*-
"""
图执行器模块（GraphRunner）。

【模块职责】
- API 接口的直接消费者，负责拉起 Supervisor 图并驱动其执行。
- 将图执行过程中产生的所有事件（模型输出、工具调用、中断、错误）
  封装为标准 SSE 格式推送给前端。
- 内置前置规则拦截引擎（Zero-LLM 快速响应）。
- 负责 Interrupt 审批流的恢复和状态回填。

【设计要点】
- 生产者-消费者解耦：图执行跑在后台 daemon 线程，主线程只消费队列推 SSE，
  避免同步阻塞导致日志延迟积压。
- Supervisor 编译图按 model_config 指纹缓存，同一配置只编译一次。
- 前置规则拦截仅在输入较短时触发，避免扫描超长文本的性能损耗。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from config.constants.sse_constants import SseEventType
from harness.core.session_manager import runtime_session_manager
from harness.types import RunContext
from common.utils.custom_logger import get_logger

log = get_logger(__name__)


# 规则逻辑处理
def rule_handle(self,
                user_input: str,
                session_id: str,
                rule_result: Optional[Tuple[str, str]] = None,
                emit_response_start: bool = True,
                run_context: RunContext = None,
                run_id: str = None,
                ):
    # 规则命中，直接返回结果，不进入图执行
    thinking_hint, rule_content = rule_result
    try:
        # 发送 response_start 事件
        if emit_response_start:
            yield self._fmt_sse(SseEventType.RESPONSE_START.value, "")

        # 发送"老板下达任务"事件
        yield self._fmt_workflow_event(
            self._build_workflow_event(
                session_id=session_id,
                run_id=run_id,
                phase="user_message_received",
                title="老板下达任务",
                summary=user_input,
                status="completed",
                role="boss",
                meta={"input_length": len(user_input or "")},
            )
        )

        # 发送"总管快速裁决"事件
        yield self._fmt_workflow_event(
            self._build_workflow_event(
                session_id=session_id,
                run_id=run_id,
                phase="rule_intercepted",
                title="总管快速裁决",
                summary=thinking_hint,
                status="completed",
                role="supervisor",
                agent_name="ChatAgent",
            )
        )

        # 发送思考提示
        yield self._fmt_sse(SseEventType.THINKING.value, thinking_hint)

        # 发送规则拦截结果
        yield self._fmt_sse(SseEventType.STREAM.value, rule_content)

        # 发送"总管回禀老板"事件
        yield self._fmt_workflow_event(
            self._build_workflow_event(
                session_id=session_id,
                run_id=run_id,
                phase="final_report_delivered",
                title="总管回禀老板",
                summary=rule_content[:160],
                status="completed",
                role="supervisor",
                agent_name="ChatAgent",
            )
        )

        # 更新运行状态为完成
        runtime_session_manager.mark_completed(
            run_context,
            phase="final_report_delivered",
            summary=rule_content[:160],
            title="总管回禀老板",
        )

        # 发送 response_end 事件
        yield self._fmt_sse(SseEventType.RESPONSE_END.value, "")
    finally:
        # 清理运行上下文
        runtime_session_manager.cleanup_run(run_context)
    return
