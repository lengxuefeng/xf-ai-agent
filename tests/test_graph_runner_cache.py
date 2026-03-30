import threading
import time
import unittest
from unittest.mock import patch

from agent.graph_runner import GraphRunner


class _DummySessionPool:
    def __init__(self) -> None:
        self.borrow_calls = 0
        self.register_calls = 0

    def borrow(self, model_config, timeout=None):
        self.borrow_calls += 1
        return None

    def register_config(self, model_config):
        self.register_calls += 1


class GraphRunnerCacheTest(unittest.TestCase):
    def test_get_or_create_supervisor_builds_once_for_same_config_under_concurrency(self):
        runner = GraphRunner(model_config={})
        model_config = {"model": "unit-test-model", "provider": "stub"}
        pool = _DummySessionPool()
        build_started = threading.Event()
        allow_finish = threading.Event()
        build_count = 0
        build_count_lock = threading.Lock()
        results = []
        errors = []

        def fake_create_graph(config):
            nonlocal build_count
            with build_count_lock:
                build_count += 1
                build_id = build_count
            build_started.set()
            self.assertTrue(allow_finish.wait(timeout=1.0))
            time.sleep(0.02)
            return {"config": dict(config), "build_id": build_id}

        def worker():
            try:
                results.append(runner._get_or_create_supervisor(model_config, borrow_timeout=0.01))
            except Exception as exc:  # pragma: no cover - failure path asserted below
                errors.append(exc)

        with patch("agent.graph_runner.create_supervisor_graph", side_effect=fake_create_graph), patch(
            "services.session_pool.session_pool",
            pool,
        ):
            first = threading.Thread(target=worker)
            second = threading.Thread(target=worker)

            first.start()
            self.assertTrue(build_started.wait(timeout=1.0))
            second.start()
            time.sleep(0.05)
            allow_finish.set()

            first.join(timeout=1.0)
            second.join(timeout=1.0)

        self.assertFalse(errors)
        self.assertEqual(2, len(results))
        self.assertIs(results[0], results[1])
        self.assertEqual(1, build_count)
        self.assertEqual(1, pool.borrow_calls)
        self.assertEqual(1, pool.register_calls)


if __name__ == "__main__":
    unittest.main()
