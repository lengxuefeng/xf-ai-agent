import os
from typing import Dict, Optional

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


class YunYouTools:
    def __init__(self):
        self.base_url = os.getenv("YY_BASE_URL")

    def common_post(self, url: str, params: Dict) -> Dict:
        """执行 API 请求并处理错误"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            # 过滤掉值为 None 的参数
            filtered_params = {k: v for k, v in params.items() if v is not None}

            url = f"{self.base_url}{url}"
            response = requests.post(url, json=filtered_params, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("data")
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 401:
                raise ValueError("错误：无效的云柚 API 密钥")
            elif response.status_code == 404:
                raise ValueError(f"错误：找不到资源，参数：{params}")
            else:
                raise ValueError(f"HTTP 错误: {http_err}")
        except Exception as e:
            raise ValueError(f"查询时发生未知错误: {e}")


@tool
def holter_list(startUseDay: str, endUseDay: str, isUploaded: Optional[int] = None, reportStatus: Optional[int] = None,
                holterType: Optional[int] = None) -> Dict:
    """
    云柚 holter 数据列表，根据具体参数查询holter信息

    Args:
        startUseDay (str): 开始日期，格式如 "2020-07-30"
        endUseDay (str): 结束日期，格式如 "2020-07-30"
        isUploaded (Optional[int]): 数据是否传完 0：否 1：是。如果不需要则忽略。
        reportStatus (Optional[int]): 报告审核状态 0:待审核 1:审核中 2:人工审核完成 3:自动审核完成。如果不需要则忽略。
        holterType (Optional[int]): holter类型 0:24小时 1:2小时 2:24小时（夜间）3:48小时。如果不需要则忽略。
    """
    params = {
        "startUseDay": startUseDay,
        "endUseDay": endUseDay,
        "isUploaded": isUploaded,
        "reportStatus": reportStatus,
        "holterType": holterType,
    }
    return YunYouTools().common_post("/holter/list", params)


@tool
def holter_type_count(startUseDay: str, endUseDay: str) -> Dict:
    """
    云柚 获取holter类型统计，根据时间范围查询holter类型统计
    Args:
        startUseDay (str): 开始日期，格式如 "2020-07-30"
        endUseDay (str): 结束日期，格式如 "2020-07-30"
    Returns:
        Dict: 响应数据，包含holter报告状态统计
    """
    params = {
        "startUseDay": startUseDay,
        "endUseDay": endUseDay,
    }
    print(f"holter_type_count params startUseDay: {startUseDay}, endUseDay: {endUseDay}")
    return YunYouTools().common_post("/holter/holterTypeCount", params)


@tool
def holter_report_count(startUseDay: str, endUseDay: str) -> Dict:
    """
    云柚 获取holter报告状态统计，根据时间范围查询holter报告状态统计
    Args:
        startUseDay (str): 开始日期，格式如 "2020-07-30"
        endUseDay (str): 结束日期，格式如 "2020-07-30"
    Returns:
        Dict: 响应数据，包含holter报告状态统计
    """
    params = {
        "startUseDay": startUseDay,
        "endUseDay": endUseDay,
    }
    return YunYouTools().common_post("/holter/holterReportCount", params)


if __name__ == '__main__':
    llm = ChatOpenAI(
        model="GLM-4.6",
        temperature=0.5,
        api_key="372825e177b84308934ad6564cab60f7.1yrv1Sv8mL7YhHZA",
        # base_url="https://open.bigmodel.cn/api/paas/v4",
        base_url="https://open.bigmodel.cn/api/coding/paas/v4",
    )
    tools = [holter_type_count]
    a = create_agent(
        llm,
        tools=tools,
        system_prompt="你是一个专业的医疗数据分析师，你需要根据用户的问题，调用云柚的API来获取数据。"
    )

    response = a.invoke({
        "messages": [HumanMessage(content="今天holter的数据有多少条")]
    })
    print(response["messages"][-1].content)

    # print(holter_type_count.invoke({"startUseDay": "2025-07-30", "endUseDay": "2025-08-30"}))
