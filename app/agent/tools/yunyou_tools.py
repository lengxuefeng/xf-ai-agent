import os
from typing import Dict

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()


class YunYouTools:
    def __init__(self):
        self.base_url = os.getenv("YY_BASE_URL")

    def _common_post(self, url: str, params: Dict) -> Dict:
        """执行 API 请求并处理错误"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            params = {**params}

            url = f"{self.base_url}{url}"
            response = requests.post(url, json=params, headers=headers)
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
def holter_list(params: Dict) -> Dict:
    """
    云柚 holter 数据列表，根据具体参数查询holter信息
    Args:
        params (Dict): 请求参数，参数包含{
        "startUseDay": "2020-07-30", # 开始日期-必须
        "endUseDay": "2020-07-30", # 解释日期-必须
        "isUploaded": 0, # 数据是否传完 0：否 1：是  -1：没有数据
        "reportStatus": 0, # 报告审核状态 -1:无数据 0:待审核 1:审核中 2:人工审核完成 3:自动审核完成
        "holterType": 0, # holter类型 0:24小时 1:2小时 2:24小时（夜间）3:48小时
    }
    Returns:
        Dict: 响应数据
    """
    print(f"params: {params}")
    return YunYouTools()._common_post("/holter/list", params)


@tool
def holter_type_count(params: Dict) -> Dict:
    """
    云柚 获取holter类型统计，根据时间范围查询holter类型统计
    Args:
        params (Dict): 请求参数，参数包含{
        "startUseDay": "2020-07-30", # 开始日期-必须
        "endUseDay": "2020-07-30", # 解释日期-必须
    }
    Returns:
        Dict: 响应数据，包含holter报告状态统计，参数包含[{
            "count": 0, # 总数量
            "holterType": 0, # holter类型 0:24小时 1:2小时 2:24小时（夜间）3:48小时
        }]
    """
    return YunYouTools()._common_post("/holter/holterTypeCount", params)


@tool
def holter_report_count(params: Dict) -> Dict:
    """
    云柚 获取holter报告状态统计，根据时间范围查询holter报告状态统计
    Args:
        params (Dict): 请求参数，参数包含{
        "startUseDay": "2020-07-30", # 开始日期-必须
        "endUseDay": "2020-07-30", # 解释日期-必须
    }
    Returns:
        Dict: 响应数据，包含holter报告状态统计，参数包含[{
            "count": 0, # 总数量
            "reportStatus": 0, # 报告审核状态 -1:无数据 0:待审核 1:审核中 2:人工审核完成 3:自动审核完成
        }]
    """
    return YunYouTools()._common_post("/holter/holterReportCount", params)


if __name__ == '__main__':
    params = {
        "startUseDay": "2025-07-30",
        "endUseDay": "2025-08-30",
    }
    print(holter_list(params))
