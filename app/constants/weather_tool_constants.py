# -*- coding: utf-8 -*-
"""天气工具参数治理常量。"""

from config.keyword_settings import CITY_INVALID_SUFFIXES, CITY_STOPWORDS, WEATHER_QUERY_KEYWORDS

# 城市缺失时的标准提示语（不给模型暴露错误参数）
WEATHER_CITY_REQUIRED_MESSAGE = (
    "我需要先确认城市，才能查询实时天气。"
    "请直接告诉我城市名（例如：郑州、北京、上海）。"
)

# 城市无效时的标准提示语
WEATHER_CITY_NOT_FOUND_MESSAGE = (
    "未识别到可用的城市名。"
    "请提供明确城市（例如：郑州、北京、上海）后我再为您查询。"
)
