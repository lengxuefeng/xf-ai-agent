import unittest

from agent.graphs.supervisor import _analyze_request, _build_rule_based_multidomain_tasks
from constants.workflow_constants import RouteStrategy


class SupervisorMultiTaskSplitTest(unittest.TestCase):
    def _build_tasks(self, text: str):
        analysis = _analyze_request(text)
        tasks = _build_rule_based_multidomain_tasks(
            text,
            candidate_agents=analysis.candidate_agents,
            route_strategy=analysis.route_strategy,
        )
        return analysis, tasks

    def test_screenshot_case_should_split_yunyou_and_chat(self):
        text = "天气也这么不好，老板还让我查询一下holter最近的数据，怎样让老板自己去查呢"
        analysis, tasks = self._build_tasks(text)

        self.assertEqual(analysis.route_strategy, RouteStrategy.MULTI_DOMAIN_SPLIT.value)
        self.assertIsNotNone(tasks)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["agent"], "yunyou_agent")
        self.assertEqual(tasks[1]["agent"], "CHAT")

    def test_weather_background_should_not_force_weather_task(self):
        text = "天气也这么不好，老板还让我查询一下holter最近的数据"
        analysis, tasks = self._build_tasks(text)

        self.assertEqual(analysis.route_strategy, RouteStrategy.SINGLE_DOMAIN.value)
        self.assertEqual(analysis.candidate_agents, ["yunyou_agent"])
        # 规则拆分器在单域且无显式多任务时返回 None，后续由 deterministic fallback 生成单任务。
        self.assertIsNone(tasks)

    def test_interrogative_clause_without_action_hint_should_be_kept(self):
        text = "老板让我查holter最近的数据，并且怎么让他自己操作"
        analysis, tasks = self._build_tasks(text)

        self.assertEqual(analysis.route_strategy, RouteStrategy.MULTI_DOMAIN_SPLIT.value)
        self.assertIsNotNone(tasks)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["agent"], "yunyou_agent")
        self.assertEqual(tasks[1]["agent"], "CHAT")


if __name__ == "__main__":
    unittest.main()
