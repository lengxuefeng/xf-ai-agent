# app/utils/date_utils.py
from datetime import datetime, timedelta

import dateparser

from config.constants.date_parse_keywords import DATE_PREPROCESS_MAP
from common.utils.custom_logger import get_logger

log = get_logger(__name__)

"""
日期时间工具
"""


def get_current_time_context() -> str:
    """获取当前时间上下文"""
    now = datetime.now()
    return (
        f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"今天是: {now.strftime('%A')} (星期)\n"
        f"当前年份: {now.year}年"
    )


def get_agent_date_context() -> str:
    """生成Agent日期上下文，统一相对时间解释"""
    now = datetime.now().astimezone()
    today = now.date()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)
    tz_name = now.tzname() or "Local"

    return (
        "【时间基准（必须遵守）】\n"
        f"- 当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- 当前时区: {tz_name}\n"
        f"- 今天: {today.strftime('%Y-%m-%d')}\n"
        f"- 明天: {tomorrow.strftime('%Y-%m-%d')}\n"
        f"- 昨天: {yesterday.strftime('%Y-%m-%d')}\n"
        "当用户提到“今天/明天/昨天/本周”等相对时间时，必须基于以上日期解释，不可自行假设。"
    )


def parse_semantic_date(text: str) -> str:
    """解析语义日期，支持中文口语"""
    if not text:
        return text

    # 前置语义拦截与正则替换
    processed_text = text
    for k, v in DATE_PREPROCESS_MAP.items():
        if k in processed_text:
            processed_text = processed_text.replace(k, v)

    # RELATIVE_BASE 告诉解析器"今天"是哪一天
    dt = dateparser.parse(
        processed_text,
        settings={
            'RELATIVE_BASE': datetime.now(),
            'PREFER_DATES_FROM': 'past',  # 模糊词优先指向过去
            'LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD': 0.5
        }
    )

    if dt:
        parsed_date = dt.strftime("%Y-%m-%d")
        log.info(f"时间语义解析: 原文[{text}] -> 预处理[{processed_text}] -> 结果[{parsed_date}]")
        return parsed_date

    return text
