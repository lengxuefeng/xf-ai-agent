# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict


class SearchGateway:
    """
    统一搜索网关。

    说明：
    - 当前项目暂未引入真实 Web Search runtime；
    - 这里先提供统一能力描述，便于前后端观测与后续替换。
    """

    @staticmethod
    def capability_snapshot() -> Dict[str, Any]:
        return {
            "enabled": False,
            "provider": "not_configured",
            "reason": "当前项目未接入真实 web search provider，保留统一网关接口以便后续扩展。",
        }


search_gateway = SearchGateway()

