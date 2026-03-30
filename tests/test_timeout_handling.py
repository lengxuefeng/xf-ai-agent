import unittest

from agent.graphs.supervisor import _ChatNodeStreamFailure, _build_partial_chat_message
from agent.llm.loader_llm_multi import _common_transport_kwargs
from agent.llm.model_config import ModelConfig
from config.runtime_settings import (
    CHAT_NODE_TOTAL_TIMEOUT_SEC,
    ROUTER_POLICY_CONFIG,
    WORKFLOW_REFLECTION_CONFIG,
)


class TimeoutHandlingTest(unittest.TestCase):
    def test_transport_timeout_has_headroom_above_runtime_budgets(self):
        config = ModelConfig(
            model="test-model",
            model_service="test-service",
            service_type="openrouter",
        )

        transport = _common_transport_kwargs(config)

        self.assertGreaterEqual(transport["timeout"], float(CHAT_NODE_TOTAL_TIMEOUT_SEC) + 10.0)
        self.assertGreaterEqual(
            transport["timeout"],
            float(ROUTER_POLICY_CONFIG.router_llm_timeout_sec) + 5.0,
        )
        self.assertGreaterEqual(
            transport["timeout"],
            float(WORKFLOW_REFLECTION_CONFIG.llm_timeout_sec) + 5.0,
        )

    def test_partial_chat_message_keeps_partial_content_with_timeout_note(self):
        exc = _ChatNodeStreamFailure(
            "chat_node.total_timeout: 90.0s",
            partial_output_emitted=True,
            partial_content="第一段回答",
            live_streamed=True,
        )

        message = _build_partial_chat_message("第一段回答", exc, live_streamed=True)

        self.assertIsNotNone(message)
        self.assertIn("第一段回答", message.content)
        self.assertIn("响应超时提前结束", message.content)
        self.assertTrue(message.response_metadata.get("partial_failure"))
        self.assertTrue(message.response_metadata.get("live_streamed"))


if __name__ == "__main__":
    unittest.main()
