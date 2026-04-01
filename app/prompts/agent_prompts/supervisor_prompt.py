"""
系统级路由、规划器与聚合器的系统提示词管理。

【三层决策架构提示词体系】
Tier-0: 规则引擎 (Zero-LLM, 由 rules.yaml 配置，无需提示词)
Tier-1: IntentRouterPrompt  → 小模型快速分类与路由
Tier-2: PlannerPrompt       → 大模型 DAG 任务拆解
       ReflectionPrompt    → 大模型执行后反思与追加步骤
       AggregatorPrompt     → 大模型结果聚合润色
兜底:   ChatFallbackPrompt  → 通用对话降级
"""
from supervisor.registry import agent_classes
from common.utils.date_utils import get_current_time_context
from prompts.prompt_loader import render_prompt_template


class IntentRouterPrompt:
    """
    Tier-1: 小模型意图分类器提示词。
    该提示词将被注入到轻量小模型 (如 gemini-2.0-flash-lite) 中，
    负责对用户请求进行初步分析、意图分类和复杂度判定。
    """
    agents_desc = "\n".join([f"- {name}: {info.description}" for name, info in agent_classes.items()])

    @classmethod
    def get_system_prompt(cls) -> str:
        return render_prompt_template(
            "agent_prompts/templates/intent_router_system.txt",
            time_info=get_current_time_context(),
            agents_desc=cls.agents_desc,
        )


class PlannerPrompt:
    """
    Tier-2: 父规划器（Parent Planner）提示词。
    负责将复杂用户请求拆解为具有依赖关系的子任务 DAG。
    核心原则：最小粒度拆分、精确 Agent 指派、显式依赖声明。
    """
    agents_desc = "\n".join([f"- {name}: {info.description}" for name, info in agent_classes.items()])

    @classmethod
    def get_system_prompt(cls) -> str:
        return render_prompt_template(
            "agent_prompts/templates/planner_system.txt",
            time_info=get_current_time_context(),
            agents_desc=cls.agents_desc,
        )


class AggregatorPrompt:
    """
    Tier-2 收尾: 结果聚合器提示词。
    在所有子任务执行完毕后，将零散结果合成自然流畅的最终回复。
    """
    @classmethod
    def get_system_prompt(cls) -> str:
        return render_prompt_template("agent_prompts/templates/aggregator_system.txt")


class ReflectionPrompt:
    """Tier-2 执行后反思器提示词。"""

    agents_desc = "\n".join([f"- {name}: {info.description}" for name, info in agent_classes.items()])

    @classmethod
    def get_system_prompt(cls) -> str:
        return render_prompt_template(
            "agent_prompts/templates/reflection_system.txt",
            agents_desc=cls.agents_desc,
        )


class ChatFallbackPrompt:
    """通用对话降级提示词：当请求无法匹配任何专业 Agent 时使用。"""

    @classmethod
    def get_system_prompt(cls) -> str:
        return render_prompt_template(
            "agent_prompts/templates/chat_fallback_system.txt",
            time_info=get_current_time_context(),
        )
