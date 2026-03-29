# -*- coding: utf-8 -*-
"""Shared HTTP client wrapper built on httpx."""
from __future__ import annotations

import time
from typing import Any

import httpx

from schemas.http_client_schemas import HttpRequestConfig, HttpResponsePayload


class HttpRequestError(RuntimeError):
    """Raised when a common HTTP request fails."""

    def __init__(self, message: str, *, status_code: int = 0, response_text: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class CommonHttpClient:
    """Small synchronous HTTP wrapper with consistent timeout and retry behavior."""

    @staticmethod
    def _build_timeout(config: HttpRequestConfig) -> httpx.Timeout:
        connect_timeout = (
            float(config.connect_timeout_seconds)
            if config.connect_timeout_seconds is not None
            else float(config.timeout_seconds)
        )
        return httpx.Timeout(
            timeout=float(config.timeout_seconds),
            connect=connect_timeout,
            read=float(config.timeout_seconds),
            write=float(config.timeout_seconds),
            pool=float(config.timeout_seconds),
        )

    def request(self, request_config: HttpRequestConfig | dict[str, Any]) -> HttpResponsePayload:
        config = (
            request_config
            if isinstance(request_config, HttpRequestConfig)
            else HttpRequestConfig.model_validate(request_config)
        )
        last_error: Exception | None = None

        for attempt_idx in range(1, config.retry_attempts + 1):
            try:
                with httpx.Client(
                    follow_redirects=config.follow_redirects,
                    verify=config.verify_ssl,
                    trust_env=not config.disable_proxy,
                    timeout=self._build_timeout(config),
                ) as client:
                    response = client.request(
                        method=config.method,
                        url=config.url,
                        params=config.params or None,
                        headers=config.headers or None,
                        json=config.json_body,
                    )
                    response.raise_for_status()
                    json_body = None
                    try:
                        json_body = response.json()
                    except ValueError:
                        json_body = None
                    return HttpResponsePayload(
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        text=response.text,
                        json_body=json_body,
                    )
            except httpx.HTTPStatusError as exc:
                response = exc.response
                raise HttpRequestError(
                    f"HTTP错误 {response.status_code}: {exc}",
                    status_code=response.status_code,
                    response_text=response.text,
                ) from exc
            except httpx.ConnectTimeout as exc:
                last_error = HttpRequestError(f"connect_timeout: {exc}")
            except httpx.ReadTimeout as exc:
                last_error = HttpRequestError(f"read_timeout: {exc}")
            except httpx.ConnectError as exc:
                last_error = HttpRequestError(f"connection_error: {exc}")
            except httpx.HTTPError as exc:
                last_error = HttpRequestError(str(exc))

            if attempt_idx < config.retry_attempts and config.retry_backoff_ms > 0:
                time.sleep((config.retry_backoff_ms / 1000.0) * attempt_idx)

        if isinstance(last_error, HttpRequestError):
            raise last_error
        raise HttpRequestError("HTTP请求失败")


common_http_client = CommonHttpClient()
