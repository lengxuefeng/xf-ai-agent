# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List

from services.user_mcp_service import user_mcp_service


class MCPGateway:
    """统一 MCP 网关包装。"""

    @staticmethod
    def list_user_connectors(db, user_id: int) -> List[Dict[str, Any]]:
        connectors = user_mcp_service.get_user_mcps(db, user_id)
        results = []
        for item in connectors or []:
            results.append(
                {
                    "id": getattr(item, "id", None),
                    "name": getattr(item, "name", "") or getattr(item, "server_name", ""),
                    "server_url": getattr(item, "server_url", ""),
                    "enabled": bool(getattr(item, "enabled", True)),
                }
            )
        return results


mcp_gateway = MCPGateway()

