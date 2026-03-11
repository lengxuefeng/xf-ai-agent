import os
import concurrent.futures
from typing import List, Dict

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

"""
搜索相关工具
"""

load_dotenv()


@tool
def tavily_search_tool(query: str, topic: str = "general") -> List[Dict[str, str]]:
    """
    使用 Tavily API 执行实时网页搜索。

    参数:
        query (str): 搜索查询字符串
        max_results (int): 返回的最大结果数，默认为 5
        topic (str): 搜索主题，可选 "general"(（默认）：通用搜索，适用于广泛的查询，搜索结果来自各种来源（如网页、博客、维基等）。)、
        "news"（新闻）：专注于最新的新闻文章，搜索结果主要来自新闻网站和新闻聚合器。
        "finance"（金融）：专注于金融相关的内容，搜索结果主要来自金融网站和新闻聚合器。
        默认为 "general"

    返回:
        List[Dict[str, str]]: 包含搜索结果的列表，每个结果包含标题、URL 和内容片段
    """
    # 初始化 Tavily 搜索工具
    max_results = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
    timeout_sec = int(os.getenv("TAVILY_TIMEOUT_SEC", "10"))

    tavily_tool = TavilySearch(
        api_key=os.getenv("TAVILY_API_KEY"),
        max_results=max_results,
        # 搜索主题，topic 为
        topic=topic,
        # 是否在搜索结果中包含答案
        # 答案可能包含直接的回答或解释
        # 例如，“北京天气”可能返回“北京天气晴，气温 25~32°C”
        include_answer=False,
        # 是否在搜索结果中包含网页的原始内容
        # 原始内容可能包含 HTML 标签和其他元数据
        # 例如，“北京天气”可能返回“<p>北京天气晴，气温 25~32°C</p>”
        include_raw_content=False,
        # 是否在搜索结果中包含图片 URL
        # 图片 URL 可以用于生成图片摘要或可视化搜索结果
        # 例如，“北京天气”可能返回“https://example.com/weather.jpg”
        include_images=False
    )

    # 执行搜索（增加超时保护，避免前端长期无最终答复）
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(tavily_tool.invoke, {"query": query})
            result = future.result(timeout=timeout_sec)
            if isinstance(result, list):
                return result
            return [{"title": "搜索结果", "url": "", "content": str(result)}]
    except concurrent.futures.TimeoutError:
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
