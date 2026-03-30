import unittest

from runtime.core.run_state_store import RunStateStore
from runtime.types import RunContext, RunStatus


class RunStateStoreTest(unittest.TestCase):
    def test_register_run_prunes_oldest_terminal_snapshots_when_over_limit(self):
        store = RunStateStore(max_runs=2)

        run1 = RunContext(session_id="s1", run_id="r1", user_input="first")
        run2 = RunContext(session_id="s2", run_id="r2", user_input="second")
        run3 = RunContext(session_id="s3", run_id="r3", user_input="third")

        store.register_run(run1)
        store.mark_status("r1", RunStatus.COMPLETED.value)
        store.register_run(run2)
        store.mark_status("r2", RunStatus.COMPLETED.value)
        store.register_run(run3)

        self.assertIsNone(store.get("r1"))
        self.assertIsNotNone(store.get("r2"))
        self.assertIsNotNone(store.get("r3"))

    def test_register_run_prunes_oldest_snapshot_even_if_all_are_running(self):
        store = RunStateStore(max_runs=2)

        run1 = RunContext(session_id="s1", run_id="r1", user_input="first")
        run2 = RunContext(session_id="s2", run_id="r2", user_input="second")
        run3 = RunContext(session_id="s3", run_id="r3", user_input="third")

        store.register_run(run1)
        store.register_run(run2)
        store.register_run(run3)

        self.assertIsNone(store.get("r1"))
        self.assertIsNotNone(store.get("r2"))
        self.assertIsNotNone(store.get("r3"))


if __name__ == "__main__":
    unittest.main()
