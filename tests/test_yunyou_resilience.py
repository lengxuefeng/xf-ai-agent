import os
import unittest
from unittest.mock import patch

import requests

from agent.tools.yunyou_tools import YunYouTools
from config.runtime_settings import YUNYOU_HTTP_CONFIG


class _FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return {"data": self._payload}


class YunyouResilienceTest(unittest.TestCase):
    def setUp(self):
        os.environ["YY_BASE_URL"] = "http://yunyou.local"
        YunYouTools._circuit_state.clear()
        YunYouTools._response_cache.clear()

    def test_retry_then_cache_hit(self):
        tool = YunYouTools()
        params = {"startUseDay": "2026-03-01", "endUseDay": "2026-03-10"}
        retry_attempts = max(1, int(YUNYOU_HTTP_CONFIG.retry_attempts))
        with patch(
            "agent.tools.yunyou_tools.requests.post",
            side_effect=[
                requests.exceptions.ConnectionError("boom"),
                _FakeResponse({"rows": [1, 2, 3]}),
            ],
        ) as mocked_post:
            with patch("agent.tools.yunyou_tools.time.sleep", return_value=None):
                if retry_attempts > 1:
                    first = tool.common_post("holter/list", params)
                    self.assertEqual(first, {"rows": [1, 2, 3]})
                else:
                    with self.assertRaises(ValueError):
                        tool.common_post("holter/list", params)
            second = tool.common_post("holter/list", params)
            third = tool.common_post("holter/list", params)

        self.assertEqual(second, {"rows": [1, 2, 3]})
        self.assertEqual(third, {"rows": [1, 2, 3]})
        # 若重试次数 >1，成功发生在一次调用内；若重试次数 =1，则第二次调用成功，第三次命中缓存。
        self.assertEqual(mocked_post.call_count, 2)

    def test_circuit_breaker_opens_after_consecutive_failures(self):
        tool = YunYouTools()
        params = {"startUseDay": "2026-03-01", "endUseDay": "2026-03-10"}
        threshold = max(1, int(YUNYOU_HTTP_CONFIG.circuit_breaker_threshold))
        retry_attempts = max(1, int(YUNYOU_HTTP_CONFIG.retry_attempts))

        with patch("agent.tools.yunyou_tools.requests.post", side_effect=requests.exceptions.ConnectionError("boom")) as mocked_post:
            with patch("agent.tools.yunyou_tools.time.sleep", return_value=None):
                for _ in range(threshold):
                    with self.assertRaises(ValueError):
                        tool.common_post("holter/list", params)
                # 下一次应命中熔断，不再真正请求
                with self.assertRaises(ValueError) as ctx:
                    tool.common_post("holter/list", params)

        self.assertIn("熔断", str(ctx.exception))
        # 触发阈值前每次会执行 retry_attempts 次请求，熔断命中后不再触发 requests.post
        self.assertEqual(mocked_post.call_count, threshold * retry_attempts)


if __name__ == "__main__":
    unittest.main()
