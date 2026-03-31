# -*- coding: utf-8 -*-
from typing import Optional, Dict, Any

from tools.agent_tools.sql_tools import execute_sql
from tools.agent_tools.yunyou_tools import YunyouDbTools
from common.utils.custom_logger import get_logger

log = get_logger(__name__)


class FederatedQueryGateway:
    """
    多数据域查询网关（轻量版）。

    目标：
    1. 对不同数据域提供统一调用入口。
    2. 屏蔽 Agent 对底层连接细节的直接依赖。
    """

    @staticmethod
    def execute_local_sql(sql: str) -> str:
        return execute_sql(sql, domain="LOCAL_DB")

    @staticmethod
    def query_yunyou_holter_recent(
        limit: int = 5,
        order_desc: bool = True,
        start_use_day: Optional[str] = None,
        end_use_day: Optional[str] = None,
    ) -> Dict[str, Any]:
        return YunyouDbTools.query_holter_recent(
            limit=limit,
            order_desc=order_desc,
            start_use_day=start_use_day,
            end_use_day=end_use_day,
        )

    @staticmethod
    def query_yunyou_log_info(user_id: str) -> list[dict]:
        from mcp.yunyou.yunyou_log_mcp import get_log_info
        return get_log_info(user_id)


federated_query_gateway = FederatedQueryGateway()

