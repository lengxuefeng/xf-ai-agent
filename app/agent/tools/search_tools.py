# -*- coding: utf-8 -*-
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from config.runtime_settings import SEARCH_TOOL_TIMEOUT_SECONDS

"""搜索相关工具。"""

load_dotenv()


@tool
def tavily_search_tool(query: str, topic: str = "general") -> List[Dict[str, str]]:
    """
    使用 Tavily API 执行实时网页搜索（同步版）。

    参数:
        query (str): 搜索查询字符串
        topic (str): 搜索主题，可选 "general"（默认）、"news"、"finance"

    返回:
        List[Dict[str, str]]: 包含搜索结果的列表，每个结果包含标题、URL 和内容片段
    """
    max_results = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
    timeout_sec = int(SEARCH_TOOL_TIMEOUT_SECONDS)

    tavily = TavilySearch(
        api_key=os.getenv("TAVILY_API_KEY"),
        max_results=max_results,
        topic=topic,
        include_answer=False,
        include_raw_content=False,
        include_images=False,
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(tavily.invoke, {"query": query})
            result = future.result(timeout=timeout_sec)
        if isinstance(result, list):
            return result
        return [{"title": "搜索结果", "url": "", "content": str(result)}]
    except FuturesTimeoutError:
        return [{
            "title": "搜索超时",
            "url": "",
            "content": f"联网检索超过 {timeout_sec} 秒未返回，请稍后重试或缩小关键词。",
        }]
    except Exception as exc:
        return [{
            "title": "搜索失败",
            "url": "",
            "content": f"联网检索异常：{exc}",
        }]
