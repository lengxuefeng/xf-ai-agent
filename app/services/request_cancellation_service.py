# -*- coding: utf-8 -*-
"""兼容层：统一转发到 runtime.core.cancel_manager。"""

from harness.core.cancel_manager import (
    RequestCancellationService,
    runtime_cancel_manager,
)

request_cancellation_service = runtime_cancel_manager

__all__ = ["RequestCancellationService", "request_cancellation_service"]
