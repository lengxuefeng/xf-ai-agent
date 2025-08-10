from langchain_core.tools import tool

"""
搜索相关工具
"""


@tool
def get_weather(city: str) -> str:
    """获取天气信息

    Args:
        city: 城市

    Returns:
        str: 天气信息
    """
    print(f"--- 正在调用工具 get_weather，参数: {city} ---")
    if "北京" in city:
        return "今天北京天气：晴，气温 25~32°C。"
    elif "上海" in city:
        return "今天上海天气：多云转小雨，气温 22~28°C。"
    else:
        return f"抱歉，我暂时查不到 {city} 的天气信息。"
