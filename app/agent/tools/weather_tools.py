import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

from constants.weather_tool_constants import (
    WEATHER_CITY_NOT_FOUND_MESSAGE,
    WEATHER_CITY_REQUIRED_MESSAGE,
)
from utils.custom_logger import get_logger
from utils.location_parser import extract_valid_city_candidate, normalize_city_candidate

"""
定义天气查询相关的工具。

该文件通过调用 和风天气 API 来获取实时天气信息。
"""

load_dotenv()
log = get_logger(__name__)


class WeatherAPIClient:
    """和风天气 API 客户端，封装公用的 API 请求和错误处理逻辑"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('HF_API_KEY')
        if not self.api_key:
            raise ValueError("和风天气 API 密钥未设置，请在 .env 文件中设置 HF_API_KEY")
        self.base_params = {
            "key": self.api_key,
            "lang": "zh"
        }

    def _make_request(self, url: str, params: Dict) -> Dict:
        """执行 API 请求并处理错误"""
        try:
            response = requests.get(url, params={**self.base_params, **params}, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 401:
                raise ValueError("错误：无效的和风天气 API 密钥")
            elif response.status_code == 404:
                raise ValueError(f"错误：找不到资源，参数：{params}")
            else:
                raise ValueError(f"HTTP 错误: {http_err}")
        except Exception as e:
            raise ValueError(f"查询时发生未知错误: {e}")

    def get_weather_data(self, location: str) -> Dict:
        """获取实时天气数据"""
        url = os.getenv('HF_REAL_TIME_URL')
        return self._make_request(url, {"location": location})

    def search_city(self, city_name: str) -> str:
        """根据城市名搜索位置标识"""
        url = os.getenv('HF_LOCATION_URL')
        data = self._make_request(url, {"location": city_name})
        if data.get("code") == "200" and data["location"]:
            return data["location"][0]["id"]
        raise ValueError(f"未找到城市：{city_name}")


def format_weather_data(weather_data: Dict, city: str) -> str:
    """格式化天气数据为字符串"""
    fields = {
        "obsTime": "观测时间",
        "temp": "温度",
        "feelsLike": "体感温度",
        "text": "天气状况",
        "icon": "图标代码",
        "humidity": "湿度",
        "windDir": "风向",
        "wind360": "风向角度",
        "windScale": "风力等级",
        "windSpeed": "风速",
        "precip": "降水量",
        "pressure": "气压",
        "vis": "能见度",
        "cloud": "云量",
        "dew": "露点温度"
    }

    formatted = f"{city}的实时天气详情：\n"
    for key, label in fields.items():
        value = weather_data.get(key, "未知")
        unit = {
            "temp": "°C",
            "feelsLike": "°C",
            "humidity": "%",
            "wind360": "°",
            "windScale": "级",
            "windSpeed": " m/s",
            "precip": " mm",
            "pressure": " hPa",
            "vis": " km",
            "cloud": "%",
            "dew": "°C"
        }.get(key, "")
        if key == "text" and weather_data.get("icon"):
            value = f"{value}（图标代码：{weather_data['icon']}）"
        formatted += f"- {label}：{value}{unit}\n"
    log.info(f"天气工具格式化完成，城市={city}")
    return formatted


@lru_cache(maxsize=512)
def _cached_city_id_lookup(city_name: str) -> str:
    """缓存城市名到 location id 的映射，减少重复地理编码请求。"""
    client = WeatherAPIClient()
    return client.search_city(city_name)


def location_search(city_or_location: str) -> str:
    """
    根据指定城市名称查询实时天气。

    Args:
        city_or_location (str): 城市名称或位置标识（例如 "北京" 或 "101010100"）。

    Returns:
        str: 格式化的天气信息或错误信息。
    """
    # 先做参数治理：如果没有可用城市，直接返回标准提示，不访问外部 API
    resolved_city_or_id = extract_valid_city_candidate(city_or_location)
    if not resolved_city_or_id:
        return WEATHER_CITY_REQUIRED_MESSAGE

    try:
        client = WeatherAPIClient()
        # 如果输入是城市名称，转换为 location id；如果本身是 id，直接查询
        location = resolved_city_or_id
        display_city = normalize_city_candidate(resolved_city_or_id) or resolved_city_or_id
        if not resolved_city_or_id.isdigit():
            try:
                location = _cached_city_id_lookup(resolved_city_or_id)
            except Exception:
                return WEATHER_CITY_NOT_FOUND_MESSAGE
        data = client.get_weather_data(location)
        return format_weather_data(data.get("now", {}), display_city)
    except ValueError as e:
        # 对外返回稳定文案，避免把内部异常细节直接暴露给用户
        log.warning(f"天气工具调用异常: {e}")
        return WEATHER_CITY_NOT_FOUND_MESSAGE
    except Exception as e:
        log.error(f"天气工具未知异常: {e}")
        return "天气服务暂时不可用，请稍后重试。"


@tool
def get_weathers(city_names: List[str], max_threads: int = 4) -> List[str]:
    """
    根据指定城市名称查询实时天气。

    Args:
        city_names (List[str]): 城市名称列表。
        max_threads (int): 最大并发线程数，默认 4。

    Returns:
        str: 格式化的天气信息或错误信息。
    """
    # 参数兜底：空列表时直接返回标准追问文案
    normalized_city_names = [str(item).strip() for item in (city_names or []) if str(item).strip()]
    if not normalized_city_names:
        return [WEATHER_CITY_REQUIRED_MESSAGE]

    # 并发参数保护，避免极值误配置
    safe_max_threads = max(1, min(int(max_threads or 1), 8))
    with ThreadPoolExecutor(max_workers=safe_max_threads) as executor:
        results = list(executor.map(location_search, normalized_city_names))
    return results


if __name__ == "__main__":
    # 多城市并发查询示例
    cities = ["北京", "上海", "杞县"]
    results = get_weathers(city_names=cities, max_threads=3)
    for city, result in zip(cities, results):
        print(f"\n{city} 的天气：\n{result}")
