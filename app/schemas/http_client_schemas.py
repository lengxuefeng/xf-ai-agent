# -*- coding: utf-8 -*-
"""Common HTTP client request and response schemas."""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import Field

from schemas.base import BaseSchema


class HttpRequestConfig(BaseSchema):
    method: Literal["GET", "POST"]
    url: str = Field(..., min_length=1)
    params: Dict[str, Any] = Field(default_factory=dict)
    headers: Dict[str, str] = Field(default_factory=dict)
    json_body: Optional[Any] = None
    timeout_seconds: float = Field(default=10.0, gt=0.0, le=300.0)
    connect_timeout_seconds: Optional[float] = Field(default=None, gt=0.0, le=300.0)
    follow_redirects: bool = False
    verify_ssl: bool = True
    disable_proxy: bool = False
    retry_attempts: int = Field(default=1, ge=1, le=5)
    retry_backoff_ms: int = Field(default=0, ge=0, le=60000)


class HttpResponsePayload(BaseSchema):
    status_code: int
    headers: Dict[str, str] = Field(default_factory=dict)
    text: str = ""
    json_body: Optional[Any] = None
