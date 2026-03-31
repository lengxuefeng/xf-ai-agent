"""
Rule Engine 规则配置索引中心 (Hot-Reloadable)
"""
import os
import re
import yaml
from pydantic import BaseModel, Field, model_validator
from typing import List, Pattern
from common.utils.custom_logger import get_logger

log = get_logger(__name__)


class RuleConfig(BaseModel):
    id: str = Field(description="规则唯一标识")
    patterns: List[str] = Field(description="触发该规则的正则表达式列表")
    intent: str = Field(description="命中的意图标识")
    priority: int = Field(description="匹配优先级（值越大越优先拦截）")
    action: str = Field(description="命中的动作处理器 URI (如 'static_reply', 'call_tool://system/date')")
    response_template: str = Field(description="响应的话术模板 (支持 {kwarg} 占位符)")

    # 运行时缓存属性，不持久化
    _compiled_patterns: List[Pattern] = []

    @model_validator(mode="after")
    def compile_patterns(self) -> 'RuleConfig':
        """在初始化后对正则配置进行预编译并缓存，提升引擎运行时拦截性能。"""
        self._compiled_patterns = [re.compile(p) for p in self.patterns]
        return self


class RuleRegistry:
    """
    独立化、配置驱动的规则注册表。
    支持每次调用 get_rules() 时检查文件 mtime 实现配置热更。
    """

    def __init__(self, config_path: str = "app/config/rules.yaml"):
        # 将相对路径转为绝对路径（以本项目根目录为基准）
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.config_path = os.path.join(base_dir, config_path)

        self._rules_cache: List[RuleConfig] = []
        self._last_mtime: float = 0.0

    def get_rules(self) -> List[RuleConfig]:
        """
        获取按优先级降序排列的可用规则。
        如果在运行时修改了 rules.yaml，可以在下一次请求自动重载配置。
        """
        try:
            if not os.path.exists(self.config_path):
                log.warning(f"本地规则配置文件不存在: {self.config_path}")
                return self._rules_cache

            current_mtime = os.path.getmtime(self.config_path)

            # 文件无变动，且缓存有内容，直接返回
            if current_mtime == self._last_mtime and self._rules_cache:
                return self._rules_cache

            # 读取最新 YAML 重新加载
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            raw_rules = data.get("rules", [])
            new_rules = [RuleConfig(**r) for r in raw_rules]

            # 按 priority 降序排列并覆盖缓存
            new_rules.sort(key=lambda x: x.priority, reverse=True)
            self._rules_cache = new_rules
            self._last_mtime = current_mtime

            log.info(f"成功加载并刷新本地规则引擎配置表，共 {len(self._rules_cache)} 条规则。")

        except Exception as exc:
            log.error(f"RuleRegistry 加载 {self.config_path} 失败，将使用旧缓存或抛空: {exc}")

        return self._rules_cache


# 全局单例管理器
rule_registry = RuleRegistry()
