import unittest
import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage

from agent.graph_runner import GraphRunner
from agent.tools import sql_tools
from constants.sse_constants import SseEventType
from runtime.core.run_state_store import run_state_store
from runtime.core.session_manager import runtime_session_manager
from services.chat_service import ChatService
from services.request_cancellation_service import request_cancellation_service
from utils.custom_logger import get_logger


class _FakeGraph:
    def stream(self, _inputs, config=None, stream_mode=None):
        return []

    def get_state(self, _config):
        return SimpleNamespace(values={"messages": [AIMessage(content="ok")]})


class _FakeSqlResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeSqlDb:
    def __init__(self):
        self.calls = 0

    def execute(self, _sql, params=None):
        self.calls += 1
        if self.calls == 1:
            return _FakeSqlResult([("t_demo",)])
        return _FakeSqlResult([("id", "bigint", "NO", None)])


class PerformanceOptimizationsTest(unittest.TestCase):
    def test_graph_runner_skips_approval_query_for_non_resume(self):
        runner = GraphRunner()
        with (
            patch.object(runner, "_get_or_create_supervisor", return_value=_FakeGraph()),
            patch.object(runner, "_check_pending_approval", return_value=None) as mock_check,
            patch("agent.graph_runner.rule_registry.get_rules", return_value=[]),
        ):
            async def _collect():
                result = []
                async for chunk in runner.stream_run(
                    "hello", "s1", model_config={}, emit_response_start=False
                ):
                    result.append(chunk)
                return result

            chunks = asyncio.run(_collect())
        mock_check.assert_not_called()
        self.assertTrue(any("event: stream" in chunk for chunk in chunks))

    def test_graph_runner_resume_still_queries_approval(self):
        runner = GraphRunner()
        with (
            patch.object(runner, "_get_or_create_supervisor", return_value=_FakeGraph()),
            patch.object(runner, "_check_pending_approval", return_value=None) as mock_check,
        ):
            async def _collect():
                result = []
                async for chunk in runner.stream_run(
                    "[RESUME]", "s2", model_config={}, emit_response_start=False
                ):
                    result.append(chunk)
                return result

            chunks = asyncio.run(_collect())
        mock_check.assert_called_once_with("s2")
        self.assertTrue(any("event: error" in chunk for chunk in chunks))

    def test_chat_service_emits_response_start_before_context_prewarm(self):
        service = ChatService()

        @contextmanager
        def fake_db_context():
            yield object()

        with (
            patch("services.chat_service.get_db_context", fake_db_context),
            patch("services.chat_service.chat_history_service.get_or_create_session"),
            patch("services.chat_service.chat_history_service.get_recent_session_messages", return_value=[]) as mock_recent,
            patch("services.chat_service.session_state_service.build_runtime_context", return_value={}),
            patch("services.chat_service.chat_history_service.create_chat_message"),
            patch("services.chat_service.session_state_service.update_after_turn"),
            patch.object(service.graph_runner, "stream_run", return_value=iter([])),
        ):
            gen = service.stream_chat_with_history(
                user_input="你好",
                session_id="sess-1",
                model_config={"model": "x"},
                user_id=1,
                is_resume=False,
                history_limit=30,
                emit_response_start=True,
            )
            first_chunk = next(gen)
            self.assertIn("event: response_start", first_chunk)
            mock_recent.assert_not_called()
            gen.close()

    def test_custom_logger_no_propagate_and_shared_handlers(self):
        logger_a = get_logger("perf.logger.a").logger
        logger_b = get_logger("perf.logger.b").logger
        self.assertFalse(logger_a.propagate)
        self.assertFalse(logger_b.propagate)
        self.assertGreaterEqual(len(logger_a.handlers), 1)
        self.assertIs(logger_a.handlers[0], logger_b.handlers[0])

    def test_sql_schema_cache_hit(self):
        sql_tools._SCHEMA_CACHE_VALUE = ""
        sql_tools._SCHEMA_CACHE_EXPIRE_AT = 0.0
        context_call_count = {"count": 0}

        @contextmanager
        def fake_db_context():
            context_call_count["count"] += 1
            yield _FakeSqlDb()

        with patch("agent.tools.sql_tools.get_db_context", fake_db_context):
            first = sql_tools.get_schema()
            second = sql_tools.get_schema()

        self.assertIn("数据库表结构如下", first)
        self.assertEqual(first, second)
        self.assertEqual(context_call_count["count"], 1)

    def test_process_supervisor_event_keeps_tool_agent_synthetic_even_if_live_streamed(self):
        runner = GraphRunner()
        event = {
            "sql_agent": {
                "messages": [
                    AIMessage(
                        content="实时输出后的最终拼装消息",
                        name="sql_agent",
                        response_metadata={"synthetic": True, "live_streamed": True},
                    )
                ]
            }
        }
        chunks = list(
            runner._handle_updates_event(
                event,
                session_id="",
                effective_config={},
                live_streamed_agents={"sql_agent"},
                interrupt_emitted=False,
                active_agent_candidates=set(),
            )
        )
        self.assertTrue(any("event: stream" in chunk for chunk in chunks))

    def test_graph_runner_live_stream_emits_structured_workflow_event(self):
        chunks = GraphRunner._handle_live_stream_event(
            {
                "agent_name": "sql_agent",
                "content": "正在检索财务数据",
                "meta": {"task_id": "task-1"},
            },
            live_streamed_agents=set(),
            session_id="sess-live",
            run_id="sess-live:run",
        )

        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("event: workflow_event"))
        self.assertTrue(chunks[1].startswith("event: stream"))

        payload_line = next(line for line in chunks[0].splitlines() if line.startswith("data: "))
        payload = json.loads(payload_line[6:])
        self.assertEqual(payload["type"], SseEventType.WORKFLOW_EVENT.value)
        self.assertEqual(payload["payload"]["phase"], "worker_streaming")
        self.assertEqual(payload["payload"]["agent_name"], "sql_agent")
        self.assertTrue(payload["payload"]["meta"]["first_stream"])
        self.assertIn("preview", payload["payload"]["meta"])

    def test_chat_service_persists_workflow_trace_in_extra_data(self):
        service = ChatService()
        captured = {}

        @contextmanager
        def fake_db_context():
            yield object()

        workflow_chunk = GraphRunner._fmt_workflow_event(
            GraphRunner._build_workflow_event(
                session_id="sess-workflow",
                run_id="sess-workflow:run",
                phase="tasks_planned",
                title="掌柜拆解任务",
                summary="已拆分检索与汇总两个子任务",
                status="completed",
                role="supervisor",
                agent_name="ChatAgent",
            )
        )

        async def _fake_stream():
            yield GraphRunner._fmt_sse(SseEventType.THINKING.value, "掌柜正在拆解问题")
            yield workflow_chunk
            yield GraphRunner._fmt_sse(SseEventType.STREAM.value, "最终回报")

        def _capture_chat_message(_db, _user_id, chat_data):
            captured["extra_data"] = chat_data.extra_data
            captured["model_content"] = chat_data.model_content
            return SimpleNamespace(id=1)

        with (
            patch("services.chat_service.get_db_context", fake_db_context),
            patch("services.chat_service.chat_history_service.get_or_create_session"),
            patch("services.chat_service.chat_history_service.get_recent_session_messages", return_value=[]),
            patch("services.chat_service.session_state_service.build_runtime_context", return_value={}),
            patch("services.chat_service.chat_history_service.create_chat_message", side_effect=_capture_chat_message),
            patch("services.chat_service.session_state_service.update_after_turn"),
            patch.object(service.graph_runner, "stream_run", return_value=_fake_stream()),
        ):
            chunks = list(
                service.stream_chat_with_history(
                    user_input="给我做一份财务汇总",
                    session_id="sess-workflow",
                    model_config={"model": "x"},
                    user_id=1,
                    is_resume=False,
                    history_limit=30,
                    emit_response_start=False,
                )
            )

        self.assertTrue(any("event: workflow_event" in chunk for chunk in chunks))
        self.assertEqual(captured["model_content"], "最终回报")
        self.assertEqual(captured["extra_data"]["thinking_trace"], "掌柜正在拆解问题")
        self.assertEqual(captured["extra_data"]["workflow_trace"][0]["phase"], "tasks_planned")
        self.assertEqual(captured["extra_data"]["workflow_version"], 1)

    def test_request_cancellation_session_link_cascades_to_run(self):
        session_id = "sess-link-test"
        run_id = "sess-link-test:run"
        try:
            request_cancellation_service.register_request(session_id)
            request_cancellation_service.register_request(run_id)
            request_cancellation_service.link_request(session_id, run_id)

            request_cancellation_service.cancel_request(session_id)

            self.assertTrue(request_cancellation_service.is_cancelled(session_id))
            self.assertTrue(request_cancellation_service.is_cancelled(run_id))
        finally:
            request_cancellation_service.cleanup_request(run_id)
            request_cancellation_service.cleanup_request(session_id)

    def test_runtime_session_manager_tracks_latest_workflow_event(self):
        run_context = runtime_session_manager.create_run_context(
            session_id="sess-runtime",
            user_input="帮我整理一下任务",
            model_config={"model": "x"},
        )
        try:
            runtime_session_manager.register_run(run_context)
            chunk = GraphRunner._fmt_workflow_event(
                GraphRunner._build_workflow_event(
                    session_id=run_context.session_id,
                    run_id=run_context.run_id,
                    phase="tasks_planned",
                    title="掌柜拆解任务",
                    summary="已拆成两个子任务",
                    status="completed",
                    role="supervisor",
                    agent_name="ChatAgent",
                )
            )

            runtime_session_manager.record_workflow_event_chunk(run_context, chunk)

            snapshot = run_state_store.get(run_context.run_id)
            self.assertIsNotNone(snapshot)
            self.assertEqual(snapshot.current_phase, "tasks_planned")
            self.assertEqual(snapshot.title, "掌柜拆解任务")
            self.assertEqual(snapshot.status, "completed")
            self.assertEqual(snapshot.agent_name, "ChatAgent")
        finally:
            runtime_session_manager.cleanup_run(run_context)
            run_state_store.remove(run_context.run_id)

    def test_runtime_session_manager_cancel_marks_snapshot(self):
        run_context = runtime_session_manager.create_run_context(
            session_id="sess-runtime-cancel",
            user_input="取消这轮任务",
            model_config={"model": "x"},
        )
        try:
            runtime_session_manager.register_run(run_context)
            runtime_session_manager.cancel_run(run_context, summary="客户端断开")

            snapshot = run_state_store.get(run_context.run_id)
            self.assertIsNotNone(snapshot)
            self.assertEqual(snapshot.status, "cancelled")
            self.assertEqual(snapshot.current_phase, "cancelled")
            self.assertEqual(snapshot.summary, "客户端断开")
            self.assertTrue(request_cancellation_service.is_cancelled(run_context.run_id))
        finally:
            runtime_session_manager.cleanup_run(run_context)
            run_state_store.remove(run_context.run_id)


if __name__ == "__main__":
    unittest.main()
