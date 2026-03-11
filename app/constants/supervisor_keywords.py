# -*- coding: utf-8 -*-
"""Supervisor 路由关键词与模式常量。"""
from enum import Enum
from typing import Dict, Tuple


class SupervisorKeywordGroup(str, Enum):
    """Supervisor 关键词分组枚举"""

    # SQL 识别正则模式
    SQL_REGEX_PATTERNS = "sql_regex_patterns"

    # Holter 域关键词
    HOLTER_DOMAIN = "holter_domain"

    # 天气域关键词
    WEATHER_DOMAIN = "weather_domain"

    # 搜索域关键词
    SEARCH_DOMAIN = "search_domain"

    # 排序提示关键词
    FOLLOWUP_ORDER_HINT = "followup_order_hint"

    # Holter 历史关键词
    HISTORY_HOLTER = "history_holter"

    # SQL 历史关键词
    HISTORY_SQL = "history_sql"

    # 天气历史关键词
    HISTORY_WEATHER = "history_weather"

    # 搜索历史关键词
    HISTORY_SEARCH = "history_search"

    # 确认回复关键词
    FOLLOWUP_CONFIRM = "followup_confirm"

    # 指代补充关键词
    FOLLOWUP_REFERENCE = "followup_reference"

    # 地点位置正则
    FOLLOWUP_LOCATION_REGEX = "followup_location_regex"

    # 可直接复用天气上下文的问题关键词
    WEATHER_REUSE_QUERY = "weather_reuse_query"

    # 需要重新拉取实时天气的关键词
    WEATHER_REFRESH_HINT = "weather_refresh_hint"

    # 历史消息中“已包含天气事实数据”的识别关键词
    HISTORY_WEATHER_FACT = "history_weather_fact"

    # 复杂多任务连接词（用于防止简单问句误入 DAG）
    COMPLEX_CONNECTOR_HINT = "complex_connector_hint"

    # 显式请求动作词（用于识别“真正要执行的任务子句”）
    REQUEST_ACTION_HINT = "request_action_hint"

    # 非地名情绪词（用于过滤“地点补充误判”）
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
    SupervisorKeywordGroup.FOLLOWUP_ORDER_HINT: ("按id", "按 id", "根据id", "根据 id", "倒序", "倒叙", "降序", "前", "limit", "order by"),
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
