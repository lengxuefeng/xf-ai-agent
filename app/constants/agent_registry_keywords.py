# -*- coding: utf-8 -*-
"""Agent 注册关键词常量。"""
from enum import Enum
from typing import Dict, Tuple


class AgentKeywordGroup(str, Enum):
    """Agent 关键词分组。"""

    YUNYOU = "yunyou"
    MEDICAL = "medical"
    CODE = "code"
    SQL = "sql"
    WEATHER = "weather"
    SEARCH = "search"


AGENT_KEYWORDS: Dict[AgentKeywordGroup, Tuple[str, ...]] = {
    AgentKeywordGroup.YUNYOU: ("云柚", "holter", "智纤", "心电", "审核", "上传", "holter类型", "报告"),
    AgentKeywordGroup.MEDICAL: ("病", "症状", "药", "健康", "血压", "心率", "诊断", "治疗"),
    AgentKeywordGroup.CODE: ("代码", "函数", "bug", "报错", "Python", "Java", "编程", "脚本"),
    AgentKeywordGroup.SQL: ("SQL", "数据库", "查询", "表", "索引", "select", "insert"),
    AgentKeywordGroup.WEATHER: ("天气", "气温", "预报", "降雨", "温度", "几度"),
    AgentKeywordGroup.SEARCH: ("搜索", "搜一下", "查一下最新", "新闻", "百度", "谷歌", "最新消息"),
}

