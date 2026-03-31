import asyncio
import os
from functools import lru_cache
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.tools import tool

from common.http import HttpRequestError, common_http_client
from config.runtime_settings import WEATHER_TOOL_TIMEOUT_SECONDS
from config.constants.weather_tool_constants import (
    WEATHER_CITY_NOT_FOUND_MESSAGE,
    WEATHER_CITY_REQUIRED_MESSAGE,
)
from common.utils.custom_logger import get_logger
from common.utils.location_parser import extract_valid_city_candidate, normalize_city_candidate
from common.utils.retry_utils import execute_with_retry

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
            response = execute_with_retry(
                lambda: common_http_client.request(
                    {
                        "method": "GET",
                        "url": url,
                        "params": {**self.base_params, **params},
                        "timeout_seconds": float(WEATHER_TOOL_TIMEOUT_SECONDS),
                    }
                ),
                label=f"weather_api:{url}",
            )
            if isinstance(response.json_body, dict):
                return response.json_body
            raise ValueError("天气服务返回了无法解析的响应。")
        except HttpRequestError as http_err:
            if http_err.status_code == 401:
                raise ValueError("错误：无效的和风天气 API 密钥")
            elif http_err.status_code == 404:
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


def _coerce_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _coerce_int(value: object) -> Optional[int]:
    try:
        return int(float(value))
    except Exception:
        return None


def format_weather_data(weather_data: Dict, city: str) -> str:
    """格式化天气数据为更自然、更适合直接展示给用户的文本。"""

    temp = weather_data.get("temp", "未知")
    feels_like = weather_data.get("feelsLike", "未知")
    weather_text = weather_data.get("text", "天气情况未知")
    humidity = weather_data.get("humidity", "未知")
    wind_dir = weather_data.get("windDir", "风向未知")
    wind_scale = weather_data.get("windScale", "未知")
    precip = weather_data.get("precip", "未知")
    vis = weather_data.get("vis", "未知")
    obs_time = weather_data.get("obsTime", "未知")

    tips: list[str] = []
    temp_num = _coerce_float(temp)
    feels_like_num = _coerce_float(feels_like)
    humidity_num = _coerce_int(humidity)
    wind_scale_num = _coerce_int(wind_scale)
    precip_num = _coerce_float(precip)

    if precip_num is not None and precip_num > 0:
        tips.append("外出建议带伞")
    if any(token in str(weather_text) for token in ("雨", "雪", "雷")) and "外出建议带伞" not in tips:
        tips.append("外出记得留意降水")
    if temp_num is not None:
        if temp_num <= 10:
            tips.append("体感偏凉，出门建议加外套")
        elif temp_num >= 30:
            tips.append("气温偏高，注意补水防晒")
    if humidity_num is not None and humidity_num >= 80:
        tips.append("空气比较潮，体感可能会有些闷")
    if wind_scale_num is not None and wind_scale_num >= 5:
        tips.append("风有点大，出门注意防风")

    if not tips:
        tips.append("整体来看比较适合正常出行")

    lines = [
        f"帮你看了下，{city}的实时天气是{weather_text}，气温 {temp}°C，体感 {feels_like}°C。",
        f"湿度 {humidity}%，{wind_dir}{wind_scale}级，能见度 {vis} km。",
        f"小提示：{'；'.join(tips)}。",
        f"观测时间：{obs_time}。",
    ]
    formatted = "\n".join(lines)
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
async def get_weathers(city_names: List[str], max_threads: int = 4) -> List[str]:
    """
    根据指定城市名称查询实时天气（异步并发版）。

    Args:
        city_names (List[str]): 城市名称列表。
        max_threads (int): 保留参数（兼容旧接口），异步版本忽略此参数，
                           直接用 asyncio.gather 并发，无需线程池。

    Returns:
        List[str]: 格式化的天气信息列表。
    """
    normalized_city_names = [str(item).strip() for item in (city_names or []) if str(item).strip()]
    if not normalized_city_names:
        return [WEATHER_CITY_REQUIRED_MESSAGE]

    # 用 asyncio.to_thread 将同步 location_search 包裹为协程，asyncio.gather 并发执行
    tasks = [asyncio.to_thread(location_search, city) for city in normalized_city_names]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 将异常结果转换为友好文案
    return [
        r if isinstance(r, str) else f"天气服务暂时不可用：{r}"
        for r in results
    ]


if __name__ == "__main__":
    # 多城市并发查询示例
    cities = ["北京", "上海", "杞县"]
    results = get_weathers(city_names=cities, max_threads=3)
    for city, result in zip(cities, results):
        print(f"\n{city} 的天气：\n{result}")
