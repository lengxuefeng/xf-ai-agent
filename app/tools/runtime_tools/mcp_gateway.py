# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from services.user_mcp_service import user_mcp_service


class MCPGateway:
    """统一 MCP 网关包装。"""

    def __init__(self) -> None:
        self._tool_cache: Dict[str, List[BaseTool]] = {}
        self._cache_lock = threading.RLock()

    @staticmethod
    def list_user_connectors(db, user_id: int, active_only: bool = False) -> List[Dict[str, Any]]:
        connectors = user_mcp_service.get_user_mcps(db, user_id, active_only=active_only)
        results = []
        for item in connectors or []:
            results.append(
                {
                    "id": int(getattr(item, "id", 0) or 0),
                    "name": str(getattr(item, "name", "") or "").strip() or f"mcp-{getattr(item, 'id', 'unknown')}",
                    "transport": str(getattr(item, "transport", "sse") or "sse").strip().lower(),
                    "url": getattr(item, "url", None),
                    "command": getattr(item, "command", None),
                    "args": list(getattr(item, "args", []) or []),
                    "is_active": bool(getattr(item, "is_active", True)),
                }
            )
        return results

    def load_langchain_tools_sync(self, connectors: Optional[List[Dict[str, Any]]]) -> List[BaseTool]:
        normalized_connectors = self._normalize_connectors(connectors)
        if not normalized_connectors:
            return []

        cache_key = json.dumps(normalized_connectors, ensure_ascii=False, sort_keys=True)
        with self._cache_lock:
            cached_tools = self._tool_cache.get(cache_key)
            if cached_tools is not None:
                return list(cached_tools)

        loaded_tools = self._run_async(self._load_langchain_tools_async(normalized_connectors))
        with self._cache_lock:
            self._tool_cache[cache_key] = list(loaded_tools)
        return list(loaded_tools)

    async def _load_langchain_tools_async(self, connectors: List[Dict[str, Any]]) -> List[BaseTool]:
        client = MultiServerMCPClient(
            connections=self._build_connections(connectors),
            tool_name_prefix=True,
        )
        return await client.get_tools()

    @staticmethod
    def _build_connections(connectors: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        connections: Dict[str, Dict[str, Any]] = {}
        for connector in connectors:
            name = connector["name"]
            transport = connector["transport"]
            if transport == "sse":
                connections[name] = {
                    "transport": "sse",
                    "url": connector["url"],
                }
                continue
            connections[name] = {
                "transport": "stdio",
                "command": connector["command"],
                "args": connector.get("args") or [],
            }
        return connections

    @staticmethod
    def _normalize_connectors(connectors: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for connector in connectors or []:
            name = str(connector.get("name") or "").strip()
            transport = str(connector.get("transport") or "sse").strip().lower()
            if not name or transport not in {"sse", "stdio"}:
                continue

            if transport == "sse":
                url = str(connector.get("url") or "").strip()
                if not url:
                    continue
                normalized.append(
                    {
                        "name": name,
                        "transport": "sse",
                        "url": url,
                    }
                )
                continue

            command = str(connector.get("command") or "").strip()
            args = [str(item).strip() for item in (connector.get("args") or []) if str(item).strip()]
            if not command:
                continue
            normalized.append(
                {
                    "name": name,
                    "transport": "stdio",
                    "command": command,
                    "args": args,
                }
            )
        return normalized

    @staticmethod
    def _run_async(awaitable):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(awaitable)
        finally:
            loop.close()


mcp_gateway = MCPGateway()
