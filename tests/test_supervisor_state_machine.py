import unittest

from agent.graphs.supervisor import aggregator_node, dispatcher_node, reducer_node
from constants.workflow_constants import TaskStatus, WORKER_CANCELLED_RESULT


class SupervisorStateMachineTest(unittest.TestCase):
    def test_dispatcher_marks_dependency_blocked_task_as_error(self):
        state = {
            "task_list": [
                {
                    "id": "t1",
                    "agent": "yunyou_agent",
                    "input": "q1",
                    "depends_on": [],
                    "status": TaskStatus.ERROR.value,
                    "result": "Error: upstream failed",
                },
                {
                    "id": "t2",
                    "agent": "CHAT",
                    "input": "q2",
                    "depends_on": ["t1"],
                    "status": TaskStatus.PENDING.value,
                    "result": None,
                },
            ],
            "task_results": {"t1": "Error: upstream failed"},
            "current_wave": 0,
        }

        out = dispatcher_node(state)
        task2 = next(item for item in out["task_list"] if item["id"] == "t2")
        self.assertEqual(task2["status"], TaskStatus.ERROR.value)
        self.assertIn("依赖任务失败或取消", task2["result"])
        self.assertIn("t2", out["task_results"])
        self.assertEqual(out["active_tasks"], [])

    def test_reducer_marks_cancelled_task(self):
        state = {
            "task_list": [
                {
                    "id": "t1",
                    "agent": "yunyou_agent",
                    "input": "q1",
                    "depends_on": [],
                    "status": TaskStatus.DISPATCHED.value,
                    "result": None,
                }
            ],
            "task_results": {},
            "worker_results": [
                {
                    "task_id": "t1",
                    "result": WORKER_CANCELLED_RESULT,
                    "error": None,
                    "agent": "yunyou_agent",
                    "elapsed_ms": 10,
                }
            ],
        }

        out = reducer_node(state)
        task1 = out["task_list"][0]
        self.assertEqual(task1["status"], TaskStatus.CANCELLED.value)
        self.assertIn("取消", task1["result"])
        self.assertIn("t1", out["task_results"])

    def test_aggregator_skips_output_when_pending_approval_exists(self):
        state = {
            "task_list": [
                {
                    "id": "t1",
                    "agent": "yunyou_agent",
                    "input": "q1",
                    "depends_on": [],
                    "status": TaskStatus.PENDING_APPROVAL.value,
                    "result": None,
                }
            ],
            "task_results": {},
        }
        out = aggregator_node(state, model=None, config={})
        self.assertEqual(out.get("direct_answer"), "")
        self.assertNotIn("messages", out)


if __name__ == "__main__":
    unittest.main()
