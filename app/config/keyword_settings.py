# -*- coding: utf-8 -*-
"""Routing, tool, and guard keyword configuration.

Decision logic should live in code, while keywords and regex patterns live here so
they can be tuned without rewriting business logic.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Tuple


class AgentKeywordGroup(str, Enum):
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


class SupervisorKeywordGroup(str, Enum):
    SQL_REGEX_PATTERNS = "sql_regex_patterns"
    HOLTER_DOMAIN = "holter_domain"
    WEATHER_DOMAIN = "weather_domain"
    SEARCH_DOMAIN = "search_domain"
    FOLLOWUP_ORDER_HINT = "followup_order_hint"
    HISTORY_HOLTER = "history_holter"
    HISTORY_SQL = "history_sql"
    HISTORY_WEATHER = "history_weather"
    HISTORY_SEARCH = "history_search"
    FOLLOWUP_CONFIRM = "followup_confirm"
    FOLLOWUP_REFERENCE = "followup_reference"
    FOLLOWUP_LOCATION_REGEX = "followup_location_regex"
    WEATHER_REUSE_QUERY = "weather_reuse_query"
    WEATHER_REFRESH_HINT = "weather_refresh_hint"
    HISTORY_WEATHER_FACT = "history_weather_fact"
    COMPLEX_CONNECTOR_HINT = "complex_connector_hint"
    REQUEST_ACTION_HINT = "request_action_hint"
    NON_LOCATION_EMOTION_WORDS = "non_location_emotion_words"


SUPERVISOR_KEYWORDS: Dict[SupervisorKeywordGroup, Tuple[str, ...]] = {
    SupervisorKeywordGroup.SQL_REGEX_PATTERNS: (
        r"\border\s+by\b",
        r"\blimit\b",
        r"\bselect\b",
        r"\bwhere\b",
        r"\bgroup\s+by\b",
        r"\bjoin\b",
        r"数据库",
        r"数据表",
        r"表里",
        r"按id",
        r"按\s*id",
        r"根据\s*id",
        r"倒序",
        r"倒叙",
        r"降序",
        r"前\d+条",
        r"最新\d+条",
        r"sql",
        r"本地库",
        r"库里",
    ),
    SupervisorKeywordGroup.HOLTER_DOMAIN: ("holter", "云柚", "动态心电", "心电报告", "报告状态", "贴片"),
    SupervisorKeywordGroup.WEATHER_DOMAIN: (
        "天气",
        "气温",
        "温度",
        "下雨",
        "降雨",
        "风力",
        "空气质量",
        "湿度",
        "体感",
        "适合出门",
        "适合外出",
        "戴口罩",
        "跑步",
    ),
    SupervisorKeywordGroup.SEARCH_DOMAIN: (
        "活动",
        "新闻",
        "什么活动",
        "演出",
        "展览",
        "赛事",
        "附近",
        "周边",
        "好玩",
        "推荐",
    ),
    SupervisorKeywordGroup.FOLLOWUP_ORDER_HINT: (
        "按id",
        "按 id",
        "根据id",
        "根据 id",
        "倒序",
        "倒叙",
        "降序",
        "前",
        "limit",
        "order by",
    ),
    SupervisorKeywordGroup.HISTORY_HOLTER: ("holter", "云柚", "心电", "报告状态", "审核"),
    SupervisorKeywordGroup.HISTORY_SQL: ("sql", "数据库", "查询语句", "数据表"),
    SupervisorKeywordGroup.HISTORY_WEATHER: (
        "天气",
        "气温",
        "温度",
        "下雨",
        "降雨",
        "风力",
        "空气质量",
        "在哪个城市",
        "城市名称",
        "weather_agent",
    ),
    SupervisorKeywordGroup.HISTORY_SEARCH: (
        "搜索",
        "搜一下",
        "查一下",
        "活动",
        "新闻",
        "search_agent",
        "附近",
        "周边",
        "好玩",
        "去哪",
        "推荐",
        "景点",
        "商场",
        "小区",
        "房价",
    ),
    SupervisorKeywordGroup.FOLLOWUP_CONFIRM: ("是", "否", "好的", "继续", "确认", "yes", "no", "ok"),
    SupervisorKeywordGroup.FOLLOWUP_REFERENCE: (
        "那个小区",
        "这个小区",
        "后面那个",
        "前面那个",
        "你刚说的",
        "上面说的",
        "刚才那个",
        "附近那个",
        "后面那个小区",
        "你说的这些",
        "你说这些",
        "这些地方",
        "这些是哪里",
        "哪个城市",
        "离我远吗",
        "离我近吗",
        "在不在附近",
        "这些都在哪",
    ),
    SupervisorKeywordGroup.FOLLOWUP_LOCATION_REGEX: (
        r"^[\u4e00-\u9fa5]{2,12}(?:市|县|区|州|盟|旗)?$",
        r"^[A-Za-z][A-Za-z\s\-]{1,40}$",
    ),
    SupervisorKeywordGroup.WEATHER_REUSE_QUERY: (
        "适合出去",
        "适合出门",
        "要不要出门",
        "要不要外出",
        "需要带伞",
        "需要穿什么",
        "穿什么",
        "冷不冷",
        "热不热",
        "空气怎么样",
        "会不会下雨",
        "要不要戴口罩",
        "戴口罩",
        "还能跑步吗",
        "可以跑步吗",
        "晚上出门合适吗",
        "怎么穿",
        "穿衣建议",
    ),
    SupervisorKeywordGroup.WEATHER_REFRESH_HINT: (
        "现在天气",
        "实时天气",
        "最新天气",
        "重新查",
        "再查一下",
        "更新一下",
        "此刻天气",
    ),
    SupervisorKeywordGroup.HISTORY_WEATHER_FACT: (
        "温度",
        "湿度",
        "风速",
        "风力",
        "风向",
        "能见度",
        "体感温度",
        "气压",
        "天气状况",
        "实时天气",
    ),
    SupervisorKeywordGroup.COMPLEX_CONNECTOR_HINT: (
        "并且",
        "而且",
        "同时",
        "然后",
        "再帮我",
        "顺便",
        "另外",
        "以及",
        "先",
        "再",
    ),
    SupervisorKeywordGroup.REQUEST_ACTION_HINT: (
        "帮我",
        "请",
        "麻烦",
        "查询",
        "查",
        "统计",
        "获取",
        "给我",
        "告诉我",
        "输出",
        "列出",
        "看看",
        "搜",
        "推荐",
        "分析",
        "执行",
        "需要",
        "让我",
    ),
    SupervisorKeywordGroup.NON_LOCATION_EMOTION_WORDS: (
        "不开心",
        "不高兴",
        "郁闷",
        "烦",
        "生气",
        "难受",
        "糟糕",
        "不好",
        "无语",
        "崩溃",
        "绝望",
    ),
}


SUPERVISOR_SQL_EXPLICIT_ANCHORS: Tuple[str, ...] = (
    r"\bselect\b",
    r"\border\s+by\b",
    r"\blimit\b",
    r"\bwhere\b",
    r"\bgroup\s+by\b",
    r"\bjoin\b",
    r"\bsql\b",
    r"数据库",
    r"数据表",
    r"本地库",
    r"库里",
    r"\bt_[a-z0-9_]+\b",
)

SUPERVISOR_SEARCH_GENERIC_QUERY_HINTS: Tuple[str, ...] = ("查一下", "搜一下", "查一查", "搜一搜")

SUPERVISOR_WEATHER_ACTION_PATTERNS: Tuple[str, ...] = (
    r"(查|查询|看看|看下|看一下|告诉我|帮我|搜|获取).{0,8}(天气|气温|温度|下雨|降雨|风力|湿度|空气质量|体感|能见度)",
    r"(天气|气温|温度|下雨|降雨|风力|湿度|空气质量|体感|能见度).{0,8}(如何|怎么样|多少|几度|吗|呢|建议|适合)",
    r"(适合出门|适合外出|可以出门|会不会下雨|冷不冷|热不热|空气怎么样)",
)

SUPERVISOR_SEARCH_ACTION_HINTS: Tuple[str, ...] = (
    "查",
    "查询",
    "搜",
    "搜索",
    "帮我",
    "给我",
    "推荐",
    "看看",
    "列出",
    "有哪些",
    "哪里",
    "去哪",
)

SUPERVISOR_CODE_LEARNING_HINTS: Tuple[str, ...] = (
    "能学吗",
    "值得学吗",
    "好学吗",
    "难学吗",
    "怎么学",
    "学习路线",
    "学习路径",
    "入门",
    "前景",
    "就业",
    "转行",
    "零基础",
)

SUPERVISOR_CODE_ACTION_HINTS: Tuple[str, ...] = (
    "写",
    "实现",
    "生成",
    "代码",
    "函数",
    "脚本",
    "程序",
    "接口",
    "类",
    "编译",
    "运行",
    "执行",
    "报错",
    "bug",
    "异常",
    "修复",
    "调试",
    "优化",
    "重构",
    "算法",
)

SUPERVISOR_CODE_SNIPPET_PATTERNS: Tuple[str, ...] = (
    r"(class\s+\w+|def\s+\w+|function\s+\w+|if\s*\(|for\s*\(|while\s*\(|print\s*\()",
)

SUPERVISOR_CODE_LANGUAGE_ONLY_MARKERS: Tuple[str, ...] = (
    "python",
    "java",
    "javascript",
    "typescript",
    "go",
    "rust",
    "c++",
    "c#",
    "php",
    "ruby",
)

SUPERVISOR_AGENT_FAILURE_TIMEOUT_MARKERS: Tuple[str, ...] = (
    "timeout",
    "timed out",
    "readtimeout",
    "first_token_timeout",
    "total_timeout",
    "超时",
)

SUPERVISOR_AGENT_FAILURE_CONNECTION_MARKERS: Tuple[str, ...] = (
    "connection",
    "connecterror",
    "connectionerror",
    "apiconnectionerror",
    "连接失败",
    "连接断开",
)

SUPERVISOR_SUMMARY_HINTS: Tuple[str, ...] = (
    "总结",
    "分析",
    "解释",
    "建议",
    "报告",
    "结论",
    "对比",
    "方案",
    "归纳",
)

SUPERVISOR_SEQUENTIAL_HINT_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("先", "再"),
    ("先", "然后"),
    ("第一步", "第二步"),
)

SUPERVISOR_CLAUSE_CONNECTOR_PATTERNS: Tuple[str, ...] = (
    "并且",
    "而且",
    "同时",
    "然后",
    "再帮我",
    "顺便",
    "另外",
    "以及",
)

SEARCH_WEATHER_KEYWORDS: Tuple[str, ...] = (
    "天气",
    "气温",
    "温度",
    "湿度",
    "风力",
    "风向",
    "降雨",
    "能见度",
    "体感",
    "气压",
)

SEARCH_REAL_ESTATE_KEYWORDS: Tuple[str, ...] = (
    "小区",
    "房价",
    "租金",
    "楼盘",
    "公寓",
    "二手房",
    "酒店",
    "亚朵",
    "附近",
    "周边",
    "后面",
    "前面",
)

WEATHER_QUERY_KEYWORDS: Tuple[str, ...] = (
    "天气",
    "气温",
    "温度",
    "下雨",
    "降雨",
    "风力",
    "空气质量",
    "湿度",
    "体感",
    "雾",
    "霾",
    "冷不冷",
    "热不热",
)

CITY_INVALID_SUFFIXES: Tuple[str, ...] = ("吗", "呢", "呀", "吧", "嘛", "么")

CITY_STOPWORDS: Tuple[str, ...] = (
    "城市",
    "这里",
    "那里",
    "这个",
    "那个",
    "附近",
    "周边",
    "出去",
    "出去玩",
    "玩",
    "玩吗",
    "玩吧",
    "活动",
    "天气",
    "今天",
    "明天",
    "后天",
)

WEATHER_FOLLOWUP_CONFIRM_TOKENS: Tuple[str, ...] = (
    "是",
    "是的",
    "好的",
    "好",
    "确认",
    "继续",
    "然后呢",
    "嗯",
    "ok",
    "yes",
)

CODE_EXECUTE_HINTS: Tuple[str, ...] = (
    "执行",
    "运行",
    "run",
    "test",
    "测试",
    "验证",
    "试运行",
    "帮我跑",
    "帮我执行",
    "帮我测试",
)

CODE_GENERATE_ONLY_HINTS: Tuple[str, ...] = (
    "写",
    "生成",
    "示例",
    "模板",
    "hello world",
)

TOOL_GUARD_SUSPICIOUS_KEYWORDS: Tuple[str, ...] = (
    "drop table",
    "rm -rf",
    "删除数据库",
    "清空数据",
)
