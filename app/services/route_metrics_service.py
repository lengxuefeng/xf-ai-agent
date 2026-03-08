# -*- coding: utf-8 -*-
import threading
import time
from collections import Counter, deque
from typing import Dict, Any

from utils.custom_logger import get_logger, LogTarget

log = get_logger(__name__)


class RouteMetricsService:
    """
    路由指标服务（轻量版）。

    目标：
    1. 统计数据域路由与意图路由命中分布。
    2. 通过“用户纠错语句”近似估算误路由率。
    3. 提供快照给运维接口查看。
    """

    CORRECTION_KEYWORDS = (
        "不对",
        "错了",
        "查错",
        "不是这个",
        "不是我要的",
        "还是不对",
        "你查错了",
    )

    def __init__(self):
        self._lock = threading.Lock()
        self._domain_counter = Counter()
        self._intent_counter = Counter()
        self._domain_source_counter = Counter()
        self._intent_source_counter = Counter()
        self._total_domain = 0
        self._total_intent = 0
        self._correction_count = 0
        self._session_last_route: Dict[str, Dict[str, Any]] = {}
        self._recent_events = deque(maxlen=500)

    def record_domain_decision(
        self,
        session_id: str,
        user_text: str,
        domain: str,
        confidence: float,
        source: str,
    ):
        with self._lock:
            self._total_domain += 1
            self._domain_counter[domain] += 1
            self._domain_source_counter[source] += 1
            if session_id:
                self._session_last_route.setdefault(session_id, {})
                self._session_last_route[session_id].update(
                    {
                        "domain": domain,
                        "domain_confidence": confidence,
                        "domain_source": source,
                        "last_user_text": user_text,
                        "ts": time.time(),
                    }
                )
            self._recent_events.append(
                {
                    "type": "domain",
                    "session_id": session_id,
                    "domain": domain,
                    "confidence": confidence,
                    "source": source,
                    "ts": time.time(),
                }
            )

    def record_intent_decision(
        self,
        session_id: str,
        user_text: str,
        intent: str,
        confidence: float,
        source: str,
    ):
        with self._lock:
            self._total_intent += 1
            self._intent_counter[intent] += 1
            self._intent_source_counter[source] += 1
            if session_id:
                self._session_last_route.setdefault(session_id, {})
                self._session_last_route[session_id].update(
                    {
                        "intent": intent,
                        "intent_confidence": confidence,
                        "intent_source": source,
                        "last_user_text": user_text,
                        "ts": time.time(),
                    }
                )
            self._recent_events.append(
                {
                    "type": "intent",
                    "session_id": session_id,
                    "intent": intent,
                    "confidence": confidence,
                    "source": source,
                    "ts": time.time(),
                }
            )

    def detect_and_record_correction(self, session_id: str, user_text: str):
        t = (user_text or "").strip().lower()
        if not t or not session_id:
            return

        if not any(k in t for k in self.CORRECTION_KEYWORDS):
            return

        with self._lock:
            last = self._session_last_route.get(session_id)
            if not last:
                return
            self._correction_count += 1
            self._recent_events.append(
                {
                    "type": "correction",
                    "session_id": session_id,
                    "user_text": user_text,
                    "last_route": dict(last),
                    "ts": time.time(),
                }
            )
            log.warning(
                f"ROUTE_METRIC_CORRECTION session={session_id} text={user_text} last={last}",
                target=LogTarget.LOG,
            )

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            total_intent = max(self._total_intent, 1)
            misroute_rate = self._correction_count / total_intent
            hit_rate = 1.0 - misroute_rate
            return {
                "total_domain_decisions": self._total_domain,
                "total_intent_decisions": self._total_intent,
                "domain_distribution": dict(self._domain_counter),
                "intent_distribution": dict(self._intent_counter),
                "domain_source_distribution": dict(self._domain_source_counter),
                "intent_source_distribution": dict(self._intent_source_counter),
                "correction_count": self._correction_count,
                "estimated_misroute_rate": round(misroute_rate, 4),
                "estimated_hit_rate": round(hit_rate, 4),
                "recent_events": list(self._recent_events)[-100:],
            }


route_metrics_service = RouteMetricsService()

