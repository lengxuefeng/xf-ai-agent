# -*- coding: utf-8 -*-
"""日期语义预处理常量。"""
from typing import Dict


DATE_PREPROCESS_MAP: Dict[str, str] = {
    "大大前天": "4天前",
    "大前天": "3天前",
    "大大后天": "4天后",
    "大后天": "3天后",
    "跨年": "12月31日",
    "明儿": "明天",
    "昨儿": "昨天",
    "今儿": "今天",
}

