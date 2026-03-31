# -*- coding: utf-8 -*-
"""LangGraph retry compatibility helpers."""

from __future__ import annotations

import random
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Callable, Sequence, TypeVar

import httpx
import requests

from common.utils.custom_logger import get_logger

try:
    from langgraph.pregel import RetryPolicy  # type: ignore[attr-defined]
except ImportError:
    from langgraph.types import RetryPolicy  # type: ignore[no-redef]

log = get_logger(__name__)

T = TypeVar("T")


class RetryableOperationError(RuntimeError):
    """Explicit marker for externally retryable operations."""


def is_retryable_exception(exc: Exception) -> bool:
    """Decide whether an external operation should be retried."""
    if isinstance(exc, RetryableOperationError):
        return True

    if isinstance(
        exc,
        (
            ConnectionError,
            TimeoutError,
            FuturesTimeoutError,
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.TimeoutException,
            httpx.NetworkError,
            requests.Timeout,
            requests.ConnectionError,
        ),
    ):
        return True

    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600

    if isinstance(exc, requests.HTTPError):
        if exc.response is None:
            return True
        return 500 <= exc.response.status_code < 600

    if isinstance(exc, (ImportError, NameError, SyntaxError, TypeError, AttributeError)):
        return False

    if isinstance(exc, (KeyError, IndexError)):
        return False

    if isinstance(exc, ValueError):
        lower = str(exc or "").lower()
        retry_markers = (
            "timeout",
            "timed out",
            "超时",
            "connection",
            "connecterror",
            "connectionerror",
            "read timeout",
            "connect timeout",
            "temporarily unavailable",
            "暂时不可用",
            "503",
            "502",
            "504",
        )
        return any(marker in lower for marker in retry_markers)

    return True


GRAPH_RETRY_POLICY = RetryPolicy(
    initial_interval=0.5,
    backoff_factor=2.0,
    max_interval=8.0,
    max_attempts=3,
    jitter=True,
    retry_on=is_retryable_exception,
)


def retry_policy_allows_retry(
    retry_policy: RetryPolicy,
    exc: Exception,
) -> bool:
    """Evaluate the retry policy's retry_on field safely."""
    retry_on = retry_policy.retry_on
    if isinstance(retry_on, type):
        return isinstance(exc, retry_on)
    if isinstance(retry_on, Sequence) and not isinstance(retry_on, (str, bytes)):
        return any(isinstance(exc, item) for item in retry_on if isinstance(item, type))
    if callable(retry_on):
        return bool(retry_on(exc))
    return False


def compute_retry_delay(retry_policy: RetryPolicy, interval: float) -> float:
    """Compute the next retry backoff delay."""
    safe_interval = max(0.0, min(interval, float(retry_policy.max_interval)))
    if not retry_policy.jitter:
        return safe_interval
    return random.uniform(0.0, safe_interval)


def execute_with_retry(
    operation: Callable[[], T],
    *,
    retry_policy: RetryPolicy | None = None,
    label: str = "operation",
) -> T:
    """Execute an operation with LangGraph-compatible retry semantics."""
    policy = retry_policy or GRAPH_RETRY_POLICY
    max_attempts = max(1, int(policy.max_attempts))
    interval = max(0.0, float(policy.initial_interval))

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= max_attempts or not retry_policy_allows_retry(policy, exc):
                raise

            delay = compute_retry_delay(policy, interval)
            log.warning(
                "retrying external operation label=%s attempt=%s/%s delay=%.2fs error=%s",
                label,
                attempt,
                max_attempts,
                delay,
                exc,
            )
            if delay > 0:
                time.sleep(delay)
            interval = min(max(interval, float(policy.initial_interval)) * float(policy.backoff_factor), float(policy.max_interval))

    raise RuntimeError(f"{label} retry loop exited unexpectedly")
