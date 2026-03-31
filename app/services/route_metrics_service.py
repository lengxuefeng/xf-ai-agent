# -*- coding: utf-8 -*-
import threading
import time
from collections import Counter, deque
from typing import Dict, Any

from config.runtime_settings import ROUTE_METRICS_CONFIG
from common.utils.custom_logger import get_logger, LogTarget

log = get_logger(__name__)


class RouteMetricsService:
    """
    路由指标服务（轻量版）。

    目标：
    1. 统计数据域路由与意图路由命中分布。
    2. 通过“用户纠错语句”近似估算误路由率。
    3. 提供快照给运维接口查看。
    """

    def __init__(self):
        """初始化路由指标统计容器。"""
        self._lock = threading.Lock()
        self._domain_counter = Counter()
        self._intent_counter = Counter()
        self._domain_source_counter = Counter()
        self._intent_source_counter = Counter()
        self._total_domain = 0
        self._total_intent = 0
        self._correction_count = 0
        self._session_last_route: Dict[str, Dict[str, Any]] = {}
        self._recent_events = deque(maxlen=ROUTE_METRICS_CONFIG.max_events)
        self._correction_keywords = tuple(k.lower() for k in ROUTE_METRICS_CONFIG.correction_keywords if k)

    def record_domain_decision(
        self,
        session_id: str,
        user_text: str,
        domain: str,
        confidence: float,
        source: str,
    ):
        """记录一次数据域路由决策。"""
        now_ts = time.time()
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
                        "ts": now_ts,
                    }
                )
            self._recent_events.append(
                {
                    "type": "domain",
                    "session_id": session_id,
                    "domain": domain,
                    "confidence": confidence,
                    "source": source,
                    "ts": now_ts,
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
        """记录一次意图路由决策。"""
        now_ts = time.time()
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
                        "ts": now_ts,
                    }
                )
            self._recent_events.append(
                {
                    "type": "intent",
                    "session_id": session_id,
                    "intent": intent,
                    "confidence": confidence,
                    "source": source,
                    "ts": now_ts,
                }
            )

    def detect_and_record_correction(self, session_id: str, user_text: str):
        """识别用户纠错语句并记录疑似误路由事件。"""
        t = (user_text or "").strip().lower()
        if not t or not session_id:
            return

        if not any(k in t for k in self._correction_keywords):
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
        """导出当前路由指标快照。"""
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

    def get_last_route(self, session_id: str) -> Dict[str, Any]:
        """读取某会话最近一次路由快照（线程安全拷贝）。"""
        if not session_id:
            return {}
        with self._lock:
            # 返回副本，避免外部误改内部状态
            return dict(self._session_last_route.get(session_id, {}) or {})


route_metrics_service = RouteMetricsService()
