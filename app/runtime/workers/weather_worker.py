# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from common.utils.location_parser import extract_valid_city_candidate
from models.schemas.agent_runtime_schemas import AgentRequest
from runtime.workers.base import RuntimeWorker
from supervisor.supervisor_support import extract_city_from_context_slots
from tools.agent_tools.weather_tools import get_weathers


async def _invoke_weather_tool(city_names: List[str]) -> List[str]:
    return await get_weathers.ainvoke({"city_names": city_names})


class WeatherWorker(RuntimeWorker):
    """单轮天气查询 Worker。"""

    worker_name = "weather_worker"
    agent_name = "weather_agent"

    def supports(self, req: AgentRequest) -> bool:
        return True

    @staticmethod
    def _normalize_messages(req: AgentRequest) -> List[BaseMessage]:
        messages = list((req.state or {}).get("messages") or [])
        if messages:
            return messages
        user_input = str(req.user_input or "").strip()
        return [HumanMessage(content=user_input)] if user_input else []

    @staticmethod
    def _extract_city_list(text: str) -> List[str]:
        raw = str(text or "").strip()
        if not raw:
            return []
        candidates: List[str] = []

        def _push(candidate_text: str) -> None:
            city = extract_valid_city_candidate(candidate_text)
            if city and city not in candidates:
                candidates.append(city)

        weather_keywords = (
            "天气",
            "气温",
            "温度",
            "冷不冷",
            "热不热",
            "下雨",
            "会下雨",
            "要下雨",
            "冷吗",
            "热吗",
        )
        helper_prefix_pattern = (
            r"(?:请帮我|帮我|麻烦|我想|想|给我|帮忙|请问|能不能|可以|是否|"
            r"看一下|看下|看看|查一下|查下|查询一下|查询|告诉我|说下|说说|一下)"
        )

        for keyword in weather_keywords:
            for matched in re.finditer(re.escape(keyword), raw):
                prefix_window = raw[max(0, matched.start() - 18):matched.start()]
                normalized_window = re.sub(helper_prefix_pattern, "", prefix_window)
                normalized_window = re.sub(r"(今天|明天|后天|昨天|现在|最近|当前)", "", normalized_window)
                normalized_window = re.sub(r"[的啊呀吧吗呢嘛\s]+$", "", normalized_window).strip()
                _push(normalized_window)

                token_candidates = re.findall(r"[\u4e00-\u9fa5]{2,5}(?:市|县|区|州|盟|旗)?", normalized_window)
                for token in reversed(token_candidates):
                    _push(token)

        if candidates:
            return candidates

        parts = [part.strip() for part in re.split(r"[，、；;\s]+", raw) if part.strip()]
        for part in parts:
            _push(part)

        _push(raw)
        return candidates

    @staticmethod
    def _extract_city_from_history(messages: List[BaseMessage]) -> str:
        for message_item in reversed(messages[-12:]):
            if not isinstance(message_item, HumanMessage):
                continue
            city = extract_valid_city_candidate(str(getattr(message_item, "content", "") or ""))
            if city:
                return city
        return ""

    async def run(self, req: AgentRequest, *, config: RunnableConfig):
        source_messages = self._normalize_messages(req)
        latest_text = str((req.state or {}).get("current_task") or req.user_input or "").strip()
        explicit_cities = self._extract_city_list(latest_text)
        context_city = extract_city_from_context_slots((req.state or {}).get("context_slots") or {})
        history_city = self._extract_city_from_history(source_messages)
        city_names = explicit_cities or ([context_city] if context_city else []) or ([history_city] if history_city else [])
        if not city_names:
            return {
                "content": "请告诉我你想查询哪个城市的天气。",
                "response_metadata": {"runtime_worker": self.worker_name, "city_names": []},
            }

        weather_results = await _invoke_weather_tool(city_names)
        normalized_results = [str(item or "").strip() for item in (weather_results or []) if str(item or "").strip()]
        return {
            "content": "\n\n".join(normalized_results).strip(),
            "response_metadata": {
                "runtime_worker": self.worker_name,
                "city_names": city_names,
                "weather_result_count": len(normalized_results),
            },
        }


weather_worker = WeatherWorker()
