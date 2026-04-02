from types import SimpleNamespace
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

from api.v1.chat_api import _build_ws_request, _enrich_ws_event
from harness.context.context_builder import runtime_context_builder
from harness.graph_runner import GraphRunner
from harness.graph_runner_events import WsEventFormatter, parse_sse_chunk
from harness.core.run_context import build_run_context
from models.schemas.agent_runtime_schemas import AgentRequest
from models.schemas.chat_schemas import StreamChatRequest
from planner.planner import PlannerNode
from prompts.runtime_prompts.prompt_cache_adapter import PromptCacheAdapter, RuntimePromptBlock
from runtime.langgraph import checkpoint_bridge, langgraph_supervisor_shell, resume_bridge
from runtime.modes import runtime_mode_profile_resolver
from runtime.query import QueryBudget, runtime_engine
from runtime.tools import runtime_tool_orchestrator, tool_permission_resolver
from runtime.tools.models import ToolExecutionRequest, ToolExecutionReport, ToolPermissionDecision
from runtime.workers.search_worker import SearchWorker
from runtime.workers.weather_worker import WeatherWorker
from services.chat_service import ChatService
from services.chat_stream_support import extract_ai_content_from_chunk
from supervisor.agents.search_agent import SearchAgent
from supervisor.agents.weather_agent import WeatherAgent
from supervisor.supervisor import (
    _looks_like_holter_request,
    _looks_like_search_request,
    _resolve_forced_specialist_agent,
    _run_agent_to_completion,
    dispatch_node,
)
from tools.runtime_tools.search_gateway import search_gateway
from tools.runtime_tools.tool_registry import ToolDescriptor, ToolType


class WebSocketRequestContractsTest(unittest.TestCase):
    def test_build_ws_request_rejects_mismatched_session_id(self):
        with self.assertRaisesRegex(ValueError, "session_id"):
            _build_ws_request(
                {
                    "user_input": "hello",
                    "session_id": "session_payload",
                    "model": "glm-4.7",
                    "service_type": "zhipu",
                },
                session_id="session_path",
                anonymous=False,
            )


class WsEventFormatterContractsTest(unittest.TestCase):
    def test_interrupt_event_is_preserved_as_structured_approval(self):
        chunk = (
            'event: interrupt\n'
            'data: {"message":"需要人工审核","message_id":"msg_1","allowed_decisions":["approve","reject"],'
            '"action_requests":[{"id":"call_1","action_name":"run_shell","action_args":{"cmd":"ls -la"}}],'
            '"agent_name":"code_agent"}\n\n'
        )

        events = WsEventFormatter().consume(chunk)

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["type"], "approval_required")
        self.assertEqual(event["status"], "waiting_approval")
        self.assertEqual(event["message_id"], "msg_1")
        self.assertEqual(event["allowed_decisions"], ["approve", "reject"])
        self.assertEqual(event["action_requests"][0]["action_name"], "run_shell")
        self.assertEqual(event["action_requests"][0]["args"], {"cmd": "ls -la"})
        self.assertEqual(event["payload"]["agent_name"], "code_agent")

    def test_ws_event_enrichment_bubbles_up_run_identity(self):
        enriched = _enrich_ws_event(
            {
                "type": "workflow_event",
                "payload": {
                    "run_id": "run_contract_1",
                    "meta": {"request_id": "req_contract_1"},
                },
            },
            session_id="session_contract_1",
            sequence=3,
        )

        self.assertEqual(enriched["run_id"], "run_contract_1")
        self.assertEqual(enriched["request_id"], "req_contract_1")


class ChatServiceContractsTest(unittest.TestCase):
    def setUp(self):
        self.service = ChatService()

    def test_authenticated_request_requires_bound_user_model(self):
        req = StreamChatRequest(
            user_input="hello",
            session_id="session_auth_contract",
            model="glm-4.7",
            service_type="zhipu",
            model_service="zhipu",
            model_key="sk-test",
            model_url="https://example.invalid/v1",
        )

        with self.assertRaisesRegex(ValueError, "当前未绑定可用模型"):
            self.service._resolve_model_config(req, None, user_id=1)

    def test_runtime_overrides_keep_workspace_and_resume_message(self):
        req = StreamChatRequest(
            user_input="[RESUME]",
            session_id="session_resume_contract",
            model="gpt-4o-mini",
            service_type="openai",
            model_service="openai",
            model_key="sk-test",
            resume_message_id="approval_msg_1",
            workspace_root="/tmp/runtime-contract",
        )

        effective_config = self.service._attach_runtime_overrides(
            {"model": "gpt-4o-mini", "service_type": "openai"},
            req,
            user_id=9,
        )

        self.assertEqual(effective_config["resume_message_id"], "approval_msg_1")
        self.assertEqual(effective_config["workspace_root"], "/tmp/runtime-contract")
        self.assertEqual(effective_config["runtime_user_id"], 9)

    def test_request_model_config_carries_model_url(self):
        req = StreamChatRequest(
            user_input="hello",
            session_id="session_model_url_contract",
            model="gpt-4o-mini",
            service_type="openai",
            model_service="openai",
            model_key="sk-test",
            model_url="https://example.invalid/v1",
        )

        model_config = self.service._build_model_config_from_request(req)

        self.assertEqual(model_config["model_url"], "https://example.invalid/v1")

    def test_ai_content_extraction_strips_internal_tool_noise(self):
        chunk = (
            'event: stream\n'
            'data: {"type":"stream","content":"web_search_proxy{\\"query\\":\\"上海天气\\",\\"source\\":\\"external_weather\\"}上海今天多云"}\n\n'
        )

        extracted = extract_ai_content_from_chunk(chunk)

        self.assertEqual(extracted, "上海今天多云")


class PlannerContractsTest(unittest.IsolatedAsyncioTestCase):
    async def test_multi_domain_request_is_split_instead_of_passthrough(self):
        result = await PlannerNode.planner_node(
            {
                "messages": [HumanMessage(content="今天天气怎么样？南京的天气呢？Holter特今天有人使用吗")],
                "route_strategy": "multi_domain_split",
                "task_list": [],
            },
            model=object(),
            config={},
        )

        self.assertEqual(result["planner_source"], "rule_split")
        self.assertEqual(
            result["plan"],
            ["今天天气怎么样", "南京的天气呢", "Holter特今天有人使用吗"],
        )
        self.assertEqual(
            [task["agent"] for task in result["task_list"]],
            ["weather_agent", "weather_agent", "yunyou_agent"],
        )

    async def test_company_info_query_is_classified_as_search(self):
        self.assertTrue(_looks_like_search_request("帮我查询南京云佑公司的企业信息和公开资料"))
        self.assertTrue(SearchAgent._requires_live_search("帮我查询南京云佑公司的企业信息和公开资料"))
        self.assertTrue(_looks_like_holter_request("请告诉我今天浩特有人使用吗"))
        self.assertEqual(_resolve_forced_specialist_agent("请告诉我今天浩特有人使用吗"), "yunyou_agent")

    async def test_company_relation_query_stays_passthrough_single(self):
        result = await PlannerNode.planner_node(
            {
                "messages": [HumanMessage(content="帮我查询一下南京云佑公司和上海支线智康公司关系")],
                "route_strategy": "single_domain",
                "task_list": [],
            },
            model=object(),
            config={},
        )

        self.assertTrue(PlannerNode._is_simple_request("帮我查询一下南京云佑公司和上海支线智康公司关系"))
        self.assertEqual(result["planner_source"], "passthrough_single")
        self.assertEqual(result["plan"], ["帮我查询一下南京云佑公司和上海支线智康公司关系"])

    async def test_complex_single_request_skips_llm_when_disabled(self):
        class NoPlannerModel:
            def with_structured_output(self, *args, **kwargs):
                raise AssertionError("planner llm should not be called when disabled")

        with patch("planner.planner.ROUTER_POLICY_CONFIG", SimpleNamespace(planner_llm_fallback_enabled=False)):
            result = await PlannerNode.planner_node(
                {
                    "messages": [HumanMessage(content="先查询南京云佑公司，再查询上海支线智康公司关系")],
                    "route_strategy": "complex_single_domain",
                    "task_list": [],
                },
                model=NoPlannerModel(),
                config={},
            )

        self.assertEqual(result["planner_source"], "rule_split_disabled_llm")
        self.assertEqual(
            result["plan"],
            ["查询南京云佑公司", "查询上海支线智康公司关系"],
        )
        self.assertIn("runtime_task_plan", result["memory"])
        self.assertEqual(result["memory"]["runtime_task_plan"]["steps"], result["plan"])


class SearchAgentContractsTest(unittest.IsolatedAsyncioTestCase):
    async def test_search_agent_summarizes_after_first_tool_result(self):
        class FakeModel:
            def __init__(self):
                self.bound_calls = 0
                self.invoke_calls = 0

            def bind_tools(self, tools, **kwargs):
                self.bound_calls += 1
                return self

            async def ainvoke(self, payload, config=None):
                self.invoke_calls += 1
                return AIMessage(content="已基于现有搜索结果总结答案。")

        agent = SearchAgent.__new__(SearchAgent)
        agent.llm = FakeModel()
        agent.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "test search system"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        agent.guard_prompt = agent.prompt
        agent.tools = []

        result = await agent._model_node(
            {
                "messages": [
                    HumanMessage(content="帮我查询南京云佑公司和上海支线智康公司关系"),
                    AIMessage(content="", tool_calls=[{"name": "tavily_search_tool", "id": "call_1", "args": {"query": "南京云佑公司 上海支线智康 关系"}}]),
                    ToolMessage(content="搜索结果：两家公司公开资料摘要", tool_call_id="call_1"),
                ],
                "tool_loop_count": 1,
                "current_task": "帮我查询南京云佑公司和上海支线智康公司关系",
            },
            config={},
        )

        self.assertEqual(agent.llm.bound_calls, 0)
        self.assertEqual(agent.llm.invoke_calls, 1)
        self.assertEqual(result["messages"][0].content, "已基于现有搜索结果总结答案。")
        self.assertFalse(getattr(result["messages"][0], "tool_calls", None))


class WeatherAgentContractsTest(unittest.IsolatedAsyncioTestCase):
    async def test_weather_agent_summarizes_after_first_tool_result(self):
        class FakeModel:
            def __init__(self):
                self.bound_calls = 0
                self.invoke_calls = 0

            def bind_tools(self, tools, **kwargs):
                self.bound_calls += 1
                return self

            async def ainvoke(self, payload, config=None):
                self.invoke_calls += 1
                return AIMessage(content="已基于现有天气结果直接总结。")

        agent = WeatherAgent.__new__(WeatherAgent)
        agent.llm = FakeModel()
        agent.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "weather system"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        agent.tools = []

        result = await agent._model_node(
            {
                "messages": [
                    HumanMessage(content="帮我看一下今天上海的天气"),
                    AIMessage(content="", tool_calls=[{"name": "get_weathers", "id": "call_weather_1", "args": {"city_names": ["上海"]}}]),
                    ToolMessage(content="上海当前多云，22°C，适合正常出行。", tool_call_id="call_weather_1"),
                ],
                "tool_loop_count": 1,
                "current_task": "帮我看一下今天上海的天气",
            },
            config={},
        )

        self.assertEqual(agent.llm.bound_calls, 0)
        self.assertEqual(agent.llm.invoke_calls, 1)
        self.assertEqual(result["messages"][0].content, "已基于现有天气结果直接总结。")


class QueryRuntimeContractsTest(unittest.TestCase):
    def test_runtime_engine_bootstraps_legacy_query_state(self):
        run_context = build_run_context(
            session_id="session_runtime_contract",
            user_input="帮我查询南京云佑公司关系",
            model_config={"model": "glm-4.7", "service_type": "zhipu"},
            history_messages=[{"user_content": "你好"}],
            request_id="req_runtime_contract",
        )

        query_state = runtime_engine.bootstrap_query(run_context, budget=QueryBudget(max_model_turns=4, max_tool_calls=2))

        self.assertEqual(query_state.executor_name, runtime_engine.LEGACY_EXECUTOR_NAME)
        self.assertEqual(query_state.status, "initialized")
        self.assertEqual(query_state.budget.max_model_turns, 4)
        self.assertEqual(query_state.meta["request_id"], "req_runtime_contract")


class LangGraphShellContractsTest(unittest.TestCase):
    def test_checkpoint_bridge_describes_run_context(self):
        run_context = build_run_context(
            session_id="session_shell_contract",
            user_input="hello",
            model_config={"model": "glm-4.7", "service_type": "zhipu"},
            request_id="req_shell_contract",
        )

        payload = checkpoint_bridge.describe_run(run_context)

        self.assertEqual(payload["thread_id"], "session_shell_contract")
        self.assertEqual(payload["request_id"], "req_shell_contract")
        self.assertFalse(payload["is_resume"])

    def test_graph_runner_uses_langgraph_shell_to_compile(self):
        graph_runner = GraphRunner()
        fake_graph = object()

        with patch.object(graph_runner, "_build_config_key", return_value="config_shell_contract"), patch(
            "harness.graph_runner.session_pool.borrow",
            return_value=None,
        ), patch(
            "harness.graph_runner.session_pool.register_config",
        ) as register_mock, patch(
            "harness.graph_runner.langgraph_supervisor_shell.compile_supervisor",
            return_value=fake_graph,
        ) as compile_mock:
            graph = graph_runner._get_or_create_supervisor({"model": "glm-4.7", "service_type": "zhipu"})

        self.assertIs(graph, fake_graph)
        compile_mock.assert_called_once()
        register_mock.assert_called_once()


class RuntimeModeContractsTest(unittest.TestCase):
    def test_mode_profile_resolver_for_cloud_mode(self):
        with patch("runtime.modes.get_run_mode", return_value=SimpleNamespace(value="cloud")):
            profile = runtime_mode_profile_resolver.resolve(
                tool_catalog=[
                    {"name": "web_search_proxy", "category": "search", "tool_type": "native"},
                    {"name": "user_mcp", "category": "mcp", "tool_type": "mcp"},
                ]
            )

        self.assertEqual(profile["run_mode"], "cloud")
        self.assertEqual(profile["transport"], "remote")
        self.assertTrue(profile["supports_search"])
        self.assertTrue(profile["supports_mcp"])
        self.assertFalse(profile["supports_workspace"])

    def test_mode_profile_resolver_for_local_mode(self):
        with patch("runtime.modes.get_run_mode", return_value=SimpleNamespace(value="local")):
            profile = runtime_mode_profile_resolver.resolve(
                tool_catalog=[
                    {"name": "workspace_read_file", "category": "workspace", "tool_type": "native"},
                    {"name": "workspace_run_command", "category": "exec", "tool_type": "native"},
                ]
            )

        self.assertEqual(profile["run_mode"], "local")
        self.assertEqual(profile["transport"], "hybrid")
        self.assertTrue(profile["supports_workspace"])
        self.assertTrue(profile["supports_shell"])


class ToolRuntimeContractsTest(unittest.TestCase):
    def test_search_gateway_snapshot_reports_runtime_tool(self):
        snapshot = search_gateway.capability_snapshot()

        self.assertTrue(snapshot["enabled"])
        self.assertEqual(snapshot["tool_name"], "web_search_proxy")
        self.assertEqual(snapshot["provider"], "tavily_proxy")

    def test_permission_resolver_requires_approval_when_tool_is_sensitive(self):
        fake_descriptor = ToolDescriptor(
            name="workspace_run_command",
            category="exec",
            description="run a workspace command",
            source="runtime.exec",
            tool_type=ToolType.NATIVE,
            requires_approval=True,
        )
        request = ToolExecutionRequest(tool_name="workspace_run_command", args={"command": "ls"})

        with patch("runtime.tools.permissions.runtime_tool_registry.get_tool", return_value=fake_descriptor), patch(
            "runtime.tools.permissions.runtime_tool_registry.is_tool_allowed_in_current_mode",
            return_value=True,
        ):
            decision = tool_permission_resolver.decide(request)

        self.assertEqual(decision.decision, "ask")
        self.assertTrue(decision.requires_approval)
        self.assertIn("人工审批", decision.reason)

    def test_tool_orchestrator_executes_web_search_proxy_once(self):
        run_context = build_run_context(
            session_id="session_tool_runtime",
            user_input="帮我查询南京云佑公司的公开资料",
            model_config={"model": "glm-4.7", "service_type": "zhipu"},
            request_id="req_tool_runtime",
        )

        with patch(
            "runtime.tools.executor.tool_executor.execute",
            return_value={
                "ok": True,
                "tool": {"name": "web_search_proxy"},
                "result": [{"title": "南京云佑", "url": "https://example.invalid", "content": "企业信息摘要"}],
            },
        ) as execute_mock:
            report = runtime_tool_orchestrator.execute_tool(
                "web_search_proxy",
                args={"query": "南京云佑公司", "topic": "general"},
                run_context=run_context,
                source_agent="search_worker",
            )

        self.assertTrue(report.ok)
        self.assertEqual(report.status, "completed")
        self.assertEqual(report.result[0]["title"], "南京云佑")
        execute_mock.assert_called_once_with("web_search_proxy", query="南京云佑公司", topic="general")


class ContextGovernanceContractsTest(unittest.TestCase):
    def test_context_builder_exposes_layered_context_meta(self):
        run_context = build_run_context(
            session_id="session_context_contract",
            user_input="帮我总结南京云佑公司的公开资料",
            model_config={"model": "glm-4.7", "service_type": "zhipu", "workspace_root": "/tmp/demo"},
            history_messages=[
                {"user_content": "你好", "model_content": "你好，我在。"},
                {"user_content": "帮我看南京云佑公司", "model_content": "好的。"},
            ],
            session_context={"context_summary": "用户关注企业工商与医疗方向。"},
        )

        with patch(
            "harness.context.context_builder.runtime_memory_service.build_memory_snippets",
            return_value=[],
        ), patch(
            "harness.context.context_builder.runtime_memory_service.render_memory_block",
            return_value="## Session Memory\n- 用户偏好查看企业公开信息。",
        ):
            messages, memory_snippets, context_meta = runtime_context_builder.build_messages(
                run_context=run_context,
                history_messages=[
                    {"user_content": "你好", "model_content": "你好，我在。"},
                    {"user_content": "帮我看南京云佑公司", "model_content": "好的。"},
                ],
                session_context={"context_summary": "用户关注企业工商与医疗方向。"},
                max_tokens=1200,
                max_chars=4000,
            )

        self.assertEqual(len(memory_snippets), 0)
        self.assertTrue(messages)
        self.assertIn("prompt_segments", context_meta)
        self.assertIn("context_layers", context_meta)
        self.assertIn("compact_boundary", context_meta)
        self.assertIn("memory", context_meta["prompt_segments"])
        self.assertTrue(any(layer["segment"] == "compacted_history" for layer in context_meta["context_layers"]))

    def test_prompt_cache_adapter_honors_cacheable_boundary(self):
        adapter = PromptCacheAdapter()
        messages = adapter.to_messages(
            [
                RuntimePromptBlock(segment="static", content="static prompt", boundary="cacheable"),
                RuntimePromptBlock(segment="memory", content="memory prompt", boundary="volatile"),
            ],
            provider_family="anthropic",
        )

        self.assertEqual(messages[0].additional_kwargs["cache_control"], {"type": "ephemeral"})
        self.assertEqual(messages[1].additional_kwargs, {})


class RuntimeWorkerContractsTest(unittest.IsolatedAsyncioTestCase):
    async def test_search_worker_uses_single_runtime_search(self):
        worker = SearchWorker()
        req = AgentRequest(
            user_input="帮我查询南京云佑公司的公开资料",
            session_id="session_worker_contract",
            model=RunnableLambda(lambda _: AIMessage(content="已根据单轮搜索结果整理答案。")),
            llm_config={"model": "glm-4.7", "service_type": "zhipu"},
            state={"messages": [HumanMessage(content="帮我查询南京云佑公司的公开资料")]},
        )
        fake_report = ToolExecutionReport(
            request=ToolExecutionRequest(tool_name="web_search_proxy", args={"query": "南京云佑公司"}),
            permission=ToolPermissionDecision(tool_name="web_search_proxy", decision="allow", allowed=True),
            ok=True,
            status="completed",
            result=[{"title": "南京云佑", "url": "https://example.invalid", "content": "企业信息摘要"}],
        )

        with patch("runtime.workers.search_worker.search_gateway.search_once", return_value=fake_report) as search_mock:
            result = await worker.run(req, config={"configurable": {}})

        self.assertEqual(result["content"], "已根据单轮搜索结果整理答案。")
        self.assertEqual(result["response_metadata"]["runtime_worker"], "search_worker")
        search_mock.assert_called_once()

    async def test_weather_worker_queries_once_and_returns_direct_result(self):
        worker = WeatherWorker()
        req = AgentRequest(
            user_input="帮我看一下今天上海的天气",
            session_id="session_weather_worker",
            model=None,
            llm_config={"model": "glm-4.7", "service_type": "zhipu"},
            state={"messages": [HumanMessage(content="帮我看一下今天上海的天气")]},
        )

        with patch(
            "runtime.workers.weather_worker._invoke_weather_tool",
            return_value=["上海当前多云，22°C，适合正常出行。"],
        ) as weather_mock:
            result = await worker.run(req, config={"configurable": {}})

        self.assertEqual(result["content"], "上海当前多云，22°C，适合正常出行。")
        self.assertEqual(result["response_metadata"]["runtime_worker"], "weather_worker")
        weather_mock.assert_called_once_with(["上海"])


class RuntimeWorkerSupervisorContractsTest(unittest.TestCase):
    def test_run_agent_to_completion_prefers_runtime_worker(self):
        class FakeWorker:
            worker_name = "fake_search_worker"

            def supports(self, req):
                return True

            async def run(self, req, *, config):
                return {"content": "runtime worker handled", "response_metadata": {"runtime_worker": self.worker_name}}

        with patch("supervisor.supervisor.runtime_worker_registry.get_worker", return_value=FakeWorker()), patch.dict(
            "supervisor.supervisor.agent_classes",
            {"search_agent": SimpleNamespace(cls=lambda req: (_ for _ in ()).throw(AssertionError("legacy should not run")))}
        ):
            result = _run_agent_to_completion(
                "search_agent",
                "帮我查询南京云佑公司的公开资料",
                model=object(),
                config={"configurable": {"run_id": "run_worker_contract"}},
                session_id="session_worker_contract",
                history_messages=[HumanMessage(content="帮我查询南京云佑公司的公开资料")],
                llm_config={"model": "glm-4.7", "service_type": "zhipu"},
            )

        self.assertEqual(result["content"], "runtime worker handled")
        self.assertEqual(result["response_metadata"]["runtime_worker"], "fake_search_worker")


class RuntimeTaskContractsTest(unittest.TestCase):
    def test_dispatch_node_consumes_runtime_task_plan_shape(self):
        state = {
            "plan": ["查询南京云佑公司", "查询上海支线智康公司关系"],
            "task_list": [
                {"id": "t1", "input": "查询南京云佑公司", "agent": "search_agent", "status": "pending", "depends_on": []},
                {"id": "t2", "input": "查询上海支线智康公司关系", "agent": "search_agent", "status": "pending", "depends_on": []},
            ],
            "task_results": {},
        }

        result = dispatch_node(state, config={})

        self.assertEqual(result["current_task"], "查询南京云佑公司")
        self.assertEqual(result["current_task_id"], "t1")
        self.assertEqual(result["current_step_agent"], "search_agent")
        self.assertEqual(result["plan"], ["查询上海支线智康公司关系"])


class GraphRunnerContractsTest(unittest.IsolatedAsyncioTestCase):
    async def test_query_runtime_initialization_uses_legacy_executor(self):
        run_context = build_run_context(
            session_id="session_graph_runtime",
            user_input="hello",
            model_config={"model": "glm-4.7", "service_type": "zhipu"},
            request_id="req_graph_runtime",
        )

        query_state = GraphRunner()._initialize_query_runtime(run_context)

        self.assertEqual(query_state.executor_name, runtime_engine.LEGACY_EXECUTOR_NAME)
        self.assertEqual(query_state.meta["request_id"], "req_graph_runtime")

    async def test_final_answer_fallback_rejects_snapshot_from_other_run(self):
        state_snapshot = SimpleNamespace(
            config={"configurable": {"run_id": "run_old"}},
            metadata={},
            values={
                "messages": [AIMessage(content="这是旧答案", name="Aggregator")],
                "task_results": {},
                "task_list": [],
            },
        )

        class FakeGraph:
            async def aget_state(self, config):
                return state_snapshot

        answer = await GraphRunner()._extract_final_answer_from_state(
            FakeGraph(),
            "session_contract",
            run_id="run_current",
        )

        self.assertEqual(answer, "")

    async def test_final_answer_fallback_accepts_matching_run(self):
        state_snapshot = SimpleNamespace(
            config={"configurable": {"run_id": "run_current"}},
            metadata={},
            values={
                "messages": [AIMessage(content="这是本轮答案", name="Aggregator")],
                "task_results": {},
                "task_list": [],
            },
        )

        class FakeGraph:
            async def aget_state(self, config):
                return state_snapshot

        answer = await GraphRunner()._extract_final_answer_from_state(
            FakeGraph(),
            "session_contract",
            run_id="run_current",
        )

        self.assertEqual(answer, "这是本轮答案")

    async def test_user_visible_formatter_strips_internal_tool_invocation(self):
        content = (
            'web_search_proxy{"query":"南京云佑公司企业信息","source":"external_search"}'
            "南京云佑公司的公开资料显示其主营方向与智能医疗有关。"
        )

        formatted = GraphRunner._format_user_visible_text(content)

        self.assertEqual(formatted, "南京云佑公司的公开资料显示其主营方向与智能医疗有关。")

    async def test_worker_result_emits_incremental_body_stream_for_orchestrated_run(self):
        runner = GraphRunner()
        execution_state = {"orchestrated": True, "direct_body_agents": set()}

        chunks = list(
            runner._handle_updates_event(
                event={
                    "worker_node": {
                        "worker_results": [
                            {
                                "task_id": "t1",
                                "task": "帮我写一个Java的slow方法",
                                "result": "```java\npublic class SlowMethod {}\n```",
                                "error": None,
                                "error_payload": None,
                                "agent": "code_agent",
                                "elapsed_ms": 1234,
                                "usage": None,
                            }
                        ]
                    }
                },
                session_id="session_stream_contract",
                effective_config={},
                live_streamed_agents=set(),
                execution_state=execution_state,
                interrupt_emitted=False,
                active_agent_candidates=set(),
                run_id="run_stream_contract",
            )
        )

        stream_payloads = [
            parse_sse_chunk(chunk)
            for chunk in chunks
            if chunk.startswith("event: stream")
        ]

        self.assertEqual(len(stream_payloads), 1)
        self.assertIn("## 已完成的任务结果", stream_payloads[0]["content"])
        self.assertIn("### 1. 帮我写一个Java的slow方法", stream_payloads[0]["content"])
        self.assertIn("```java", stream_payloads[0]["content"])
        self.assertTrue(execution_state["partial_task_body_emitted"])

    async def test_aggregator_body_stream_is_suppressed_after_partial_task_streams(self):
        runner = GraphRunner()
        execution_state = {
            "orchestrated": True,
            "direct_body_agents": set(),
            "partial_task_body_emitted": True,
            "streamed_task_result_ids": {"t1"},
        }

        chunks = list(
            runner._handle_updates_event(
                event={
                    "aggregator_node": {
                        "messages": [
                            AIMessage(
                                content="## 已并行完成 2 个子任务\n\n### 1. 代码处理\n示例结果",
                                name="Aggregator",
                                response_metadata={"synthetic": True, "force_emit": True},
                            )
                        ]
                    }
                },
                session_id="session_stream_contract",
                effective_config={},
                live_streamed_agents=set(),
                execution_state=execution_state,
                interrupt_emitted=False,
                active_agent_candidates=set(),
                run_id="run_stream_contract",
            )
        )

        self.assertFalse(any(chunk.startswith("event: stream") for chunk in chunks))
        self.assertTrue(any(chunk.startswith("event: workflow_event") for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
