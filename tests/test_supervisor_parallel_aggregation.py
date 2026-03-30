import unittest

from agent.graphs.supervisor import (
    _build_deterministic_aggregation,
    _should_skip_reflection_after_parallel_success,
)
from constants.workflow_constants import TaskStatus


class SupervisorParallelAggregationTest(unittest.TestCase):
    def test_parallel_aggregation_formats_merged_sections(self):
        tasks_by_id = {
            "t1": {
                "id": "t1",
                "agent": "weather_agent",
                "input": "查询明天上海天气",
                "depends_on": [],
                "status": TaskStatus.DONE.value,
                "result": None,
            },
            "t2": {
                "id": "t2",
                "agent": "search_agent",
                "input": "检索这周本地活动",
                "depends_on": [],
                "status": TaskStatus.DONE.value,
                "result": None,
            },
        }

        merged = _build_deterministic_aggregation(
            "帮我看看明天天气并找找这周活动",
            [
                ("t1", "明天多云，早晚偏凉，建议带薄外套。"),
                ("t2", "- 周末有两场市集\n- 周日有露天电影"),
            ],
            tasks_by_id=tasks_by_id,
        )

        self.assertIn("## 已并行完成 2 个子任务", merged)
        self.assertIn("### 1. 天气信息", merged)
        self.assertIn("### 2. 联网检索", merged)
        self.assertIn("系统已按可并行部分同时执行", merged)

    def test_parallel_aggregation_cleans_broken_structured_output(self):
        tasks_by_id = {
            "t1": {
                "id": "t1",
                "agent": "yunyou_agent",
                "input": "查询 holter 是否有人使用",
                "depends_on": [],
                "status": TaskStatus.DONE.value,
                "result": None,
            },
        }

        merged = _build_deterministic_aggregation(
            "告诉我 holter 是否有人使用",
            [
                (
                    "t1",
                    "[object Object],\nSELECT id, user_id FROM t_holter_use_record\nLIMIT [object Object];",
                )
            ],
            tasks_by_id=tasks_by_id,
        )

        self.assertIn("损坏的结构化片段", merged)
        self.assertNotIn("[object Object]", merged)

    def test_parallel_success_can_skip_reflection(self):
        tasks = [
            {
                "id": "t1",
                "agent": "weather_agent",
                "input": "查天气",
                "depends_on": [],
                "status": TaskStatus.DONE.value,
                "result": "晴",
            },
            {
                "id": "t2",
                "agent": "search_agent",
                "input": "查活动",
                "depends_on": [],
                "status": TaskStatus.DONE.value,
                "result": "有活动",
            },
        ]

        self.assertTrue(
            _should_skip_reflection_after_parallel_success(
                tasks,
                {"t1": "晴", "t2": "有活动"},
                planner_source="rule_split",
            )
        )
        self.assertFalse(
            _should_skip_reflection_after_parallel_success(
                tasks,
                {"t1": "晴", "t2": "有活动"},
                planner_source="llm",
            )
        )


if __name__ == "__main__":
    unittest.main()
