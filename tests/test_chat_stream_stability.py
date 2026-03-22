import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from agent.agents.weather_agent import WeatherAgent
from agent.graph_state import AgentRequest
from agent.graph_runner import GraphRunner
from agent.graphs.supervisor import chat_node
from services.agent_stream_bus import agent_stream_bus
from utils.tracing_guard import apply_tracing_env_guard


class _Chunk:
    def __init__(self, content: str):
        self.content = content


class _StreamingModel:
    def stream(self, _messages, config=None):
        yield _Chunk("你")
        yield _Chunk("好")


class _SlowFirstTokenModel:
    def stream(self, _messages, config=None):
        time.sleep(0.2)
        yield _Chunk("late")


class _ErrorOnlyGraph:
    def stream(self, _inputs, config=None, stream_mode=None):
        yield (
            "updates",
            {
                "chat_node": {
                    "error_message": "处理超时，请稍后重试。",
                    "error_detail": "chat_node.first_token_timeout: 0.1s",
                }
            },
        )

    def get_state(self, _config):
        return SimpleNamespace(values={"messages": []})


class _FakeBoundModel:
    def bind_tools(self, _tools):
        return self


class _HeartbeatGraph:
    def stream(self, _inputs, config=None, stream_mode=None):
        yield (
            "updates",
            {"Intent_Router_Node": {"intent": "CHAT", "intent_confidence": 0.92}},
        )
        time.sleep(0.15)
        yield (
            "updates",
            {
                "chat_node": {
                    "messages": [
                        AIMessage(
                            content="完成",
                            name="ChatAgent",
                            response_metadata={"synthetic": True},
                        )
                    ]
                }
            },
        )

    def get_state(self, _config):
        return SimpleNamespace(values={"direct_answer": "完成"})


class ChatStreamStabilityTest(unittest.TestCase):
    def test_tracing_guard_forces_all_tracing_flags_off_by_default(self):
        env = {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING": "true"}
        changed = apply_tracing_env_guard(env)
        self.assertTrue(changed)
        self.assertEqual(env["LANGSMITH_TRACING"], "false")
        self.assertEqual(env["LANGCHAIN_TRACING"], "false")
        self.assertEqual(env["LANGCHAIN_TRACING_V2"], "false")
        self.assertEqual(env["TRACING"], "false")
        self.assertEqual(env["TRACING_V2"], "false")

    def test_chat_node_streams_to_agent_bus(self):
        run_id = "rid-stream"
        captured = []

        def _capture(payload):
            captured.append(payload)

        agent_stream_bus.register_callback(run_id, _capture)
        try:
            with (
                patch("agent.graphs.supervisor.CHAT_NODE_STREAM_ENABLED", True),
                patch("agent.graphs.supervisor.CHAT_NODE_FIRST_TOKEN_TIMEOUT_SEC", 1.0),
                patch("agent.graphs.supervisor.CHAT_NODE_TOTAL_TIMEOUT_SEC", 5.0),
                patch("agent.graphs.supervisor.AGENT_LIVE_STREAM_ENABLED", True),
            ):
                result = chat_node(
                    state={"messages": [HumanMessage(content="你好")], "session_id": "s-chat"},
                    model=_StreamingModel(),
                    config={"configurable": {"run_id": run_id}},
                )
        finally:
            agent_stream_bus.unregister_callback(run_id)

        msg = result["messages"][0]
        self.assertEqual(msg.content, "你好")
        self.assertTrue(msg.response_metadata.get("synthetic"))
        self.assertTrue(msg.response_metadata.get("live_streamed"))
        self.assertEqual("".join(item.get("content", "") for item in captured), "你好")

    def test_chat_node_first_token_timeout_returns_error_state(self):
        run_id = "rid-timeout"
        captured = []

        def _capture(payload):
            captured.append(payload)

        agent_stream_bus.register_callback(run_id, _capture)
        try:
            with (
                patch("agent.graphs.supervisor.CHAT_NODE_STREAM_ENABLED", True),
                patch("agent.graphs.supervisor.CHAT_NODE_FIRST_TOKEN_TIMEOUT_SEC", 0.05),
                patch("agent.graphs.supervisor.CHAT_NODE_TOTAL_TIMEOUT_SEC", 0.2),
                patch("agent.graphs.supervisor.AGENT_LIVE_STREAM_ENABLED", True),
            ):
                result = chat_node(
                    state={"messages": [HumanMessage(content="你好")], "session_id": "s-timeout"},
                    model=_SlowFirstTokenModel(),
                    config={"configurable": {"run_id": run_id}},
                )
        finally:
            agent_stream_bus.unregister_callback(run_id)

        self.assertEqual(result.get("error_message"), "处理超时，请稍后重试。")
        self.assertIn("first_token_timeout", result.get("error_detail", ""))
        self.assertEqual(captured, [])

    def test_graph_runner_emits_heartbeat_after_initial_progress(self):
        runner = GraphRunner()
        fast_tuning = SimpleNamespace(
            rule_scan_max_len=60,
            chars_per_intent=15,
            queue_poll_timeout_sec=0.01,
            idle_heartbeat_sec=0.03,
            idle_timeout_sec=1.0,
            idle_timeout_enabled=False,
            hard_timeout_sec=3.0,
            post_run_interrupt_scan_enabled=False,
        )
        with (
            patch.object(runner, "_get_or_create_supervisor", return_value=_HeartbeatGraph()),
            patch("agent.graph_runner.rule_registry.get_rules", return_value=[]),
            patch("agent.graph_runner.GRAPH_RUNNER_TUNING", fast_tuning),
        ):
            async def _collect():
                result = []
                async for chunk in runner.stream_run(
                    "hello", "s-heartbeat", model_config={}, emit_response_start=False
                ):
                    result.append(chunk)
                return result

            chunks = asyncio.run(_collect())

        self.assertTrue(any("正在等待模型首包返回" in chunk for chunk in chunks))
        self.assertTrue(any("event: response_end" in chunk for chunk in chunks))

    def test_graph_runner_emits_error_event_when_chat_node_reports_error_message(self):
        runner = GraphRunner()
        with (
            patch.object(runner, "_get_or_create_supervisor", return_value=_ErrorOnlyGraph()),
            patch("agent.graph_runner.rule_registry.get_rules", return_value=[]),
        ):
            async def _collect():
                result = []
                async for chunk in runner.stream_run(
                    "hello", "s-error", model_config={}, emit_response_start=False
                ):
                    result.append(chunk)
                return result

            chunks = asyncio.run(_collect())

        self.assertTrue(any("event: error" in chunk for chunk in chunks))
        self.assertTrue(any("处理超时，请稍后重试" in chunk for chunk in chunks))
        self.assertTrue(any("event: response_end" in chunk for chunk in chunks))

    def test_weather_agent_build_graph_does_not_raise_name_error(self):
        req = AgentRequest.model_construct(
            user_input="今天天气如何",
            session_id="s-weather",
            subgraph_id="weather_agent",
            model=_FakeBoundModel(),
            state={},
            llm_config={},
        )
        agent = WeatherAgent(req)
        self.assertIsNotNone(agent.graph)

    def test_weather_agent_uses_history_city_on_confirmation(self):
        req = AgentRequest.model_construct(
            user_input="是的",
            session_id="s-weather-follow",
            subgraph_id="weather_agent",
            model=_FakeBoundModel(),
            state={
                "messages": [
                    HumanMessage(content="今天天气怎么样"),
                    AIMessage(content="我需要先确认城市"),
                    HumanMessage(content="南京"),
                ]
            },
            llm_config={},
        )
        agent = WeatherAgent(req)

        with patch("agent.agents.weather_agent.location_search", return_value="南京的实时天气详情：晴，26°C"):
            events = list(agent.run(req, config={"configurable": {"run_id": "rid-weather-follow"}}))

        # 断言：已命中强制直查路径，不再重复追问城市。
        flattened = str(events)
        self.assertIn("南京的实时天气详情", flattened)

    def test_weather_agent_queries_multi_cities_without_llm_loop(self):
        req = AgentRequest.model_construct(
            user_input="南京，北京，上海",
            session_id="s-weather-multi",
            subgraph_id="weather_agent",
            model=_FakeBoundModel(),
            state={
                "messages": [
                    HumanMessage(content="今天天气怎么样"),
                    AIMessage(content="我需要先确认城市"),
                ]
            },
            llm_config={},
        )
        agent = WeatherAgent(req)

        def _fake_location(city: str) -> str:
            return f"{city}的实时天气详情：晴"

        with patch("agent.agents.weather_agent.location_search", side_effect=_fake_location):
            events = list(agent.run(req, config={"configurable": {"run_id": "rid-weather-multi"}}))

        flattened = str(events)
        self.assertIn("南京的实时天气详情", flattened)
        self.assertIn("北京的实时天气详情", flattened)
        self.assertIn("上海的实时天气详情", flattened)

    def test_weather_agent_extracts_city_from_wrapped_task_input(self):
        req = AgentRequest.model_construct(
            user_input="你是天气执行子任务，仅允许输出天气结论。\n用户子任务：南京",
            session_id="s-weather-wrap",
            subgraph_id="weather_agent",
            model=_FakeBoundModel(),
            state={"messages": [HumanMessage(content="你是天气执行子任务，仅允许输出天气结论。\n用户子任务：南京")]},
            llm_config={},
        )
        agent = WeatherAgent(req)

        with patch("agent.agents.weather_agent.location_search", return_value="南京的实时天气详情：晴，26°C"):
            events = list(agent.run(req, config={"configurable": {"run_id": "rid-weather-wrap"}}))

        flattened = str(events)
        self.assertIn("南京的实时天气详情", flattened)

    def test_weather_agent_no_longer_depends_on_llm_bind_tools(self):
        req = AgentRequest.model_construct(
            user_input="南京天气怎么样",
            session_id="s-weather-no-llm",
            subgraph_id="weather_agent",
            model=None,
            state={"messages": [HumanMessage(content="南京天气怎么样")]},
            llm_config={},
        )
        agent = WeatherAgent(req)

        with patch("agent.agents.weather_agent.location_search", return_value="南京的实时天气详情：晴，26°C"):
            events = list(agent.run(req, config={"configurable": {"run_id": "rid-weather-no-llm"}}))

        flattened = str(events)
        self.assertIn("南京的实时天气详情", flattened)


if __name__ == "__main__":
    unittest.main()
