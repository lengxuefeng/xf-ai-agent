# -*- coding: utf-8 -*-
"""
健康检查接口。

提供 /health 端点，供负载均衡器、容器编排（k8s liveness/readiness probe）
和运维监控系统使用。返回服务整体状态、数据库连接状态和 SessionPool 状态。
"""
import time
from typing import Any, Dict

from fastapi import APIRouter
from starlette.responses import JSONResponse

from utils.custom_logger import get_logger

log = get_logger(__name__)

health_router = APIRouter()

# 服务启动时间（用于计算 uptime）
_START_TIME = time.time()
_SERVICE_VERSION = "1.0.0"


def _check_database() -> Dict[str, Any]:
    """检查数据库连接是否正常。"""
    try:
        from db import engine
        with engine.connect() as conn:
            conn.execute(__import__('sqlalchemy').text("SELECT 1"))
        return {"status": "ok"}
    except Exception as exc:
        log.warning(f"健康检查：数据库连接异常: {exc}")
        return {"status": "error", "detail": str(exc)[:200]}


def _check_session_pool() -> Dict[str, Any]:
    """检查 SessionPool 状态。"""
    try:
        from services.session_pool import session_pool
        pool_size = session_pool.size()
        return {
            "status": "ok",
            "pool_size": pool_size,
            "started": session_pool._started,
        }
    except Exception as exc:
        log.warning(f"健康检查：SessionPool 状态异常: {exc}")
        return {"status": "error", "detail": str(exc)[:200]}


def _check_cancellation_service() -> Dict[str, Any]:
    """检查取消服务状态。"""
    try:
        from services.request_cancellation_service import request_cancellation_service
        return {
            "status": "ok",
            "active_requests": request_cancellation_service.active_count(),
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)[:200]}


def _check_runtime_harness() -> Dict[str, Any]:
    """检查 Runtime/Harness 主骨架状态。"""
    try:
        from runtime.exec.command_session_service import command_session_service
        from runtime.core.run_state_store import run_state_store
        from runtime.tools.tool_registry import runtime_tool_registry
        from runtime.workspace.manager import workspace_manager

        return {
            "status": "ok",
            "workspace_root": str(workspace_manager.root_dir),
            "tool_registry": runtime_tool_registry.stats(),
            "terminal_runtime": command_session_service.stats(),
            "latest_run_snapshot_cached": bool(run_state_store._latest_by_session),  # noqa: SLF001
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)[:200]}


@health_router.get("/health", summary="服务健康检查", include_in_schema=True)
async def health_check() -> JSONResponse:
    """
    健康检查端点。

    返回各子系统状态：
    - `status`: overall 状态，ok / degraded / error
    - `uptime_seconds`: 服务已运行秒数
    - `version`: 服务版本号
    - `checks.database`: 数据库连接状态
    - `checks.session_pool`: Supervisor 图预热池状态
    - `checks.cancellation_service`: 请求取消服务状态
    """
    import asyncio

    # 并发执行所有检查，避免串行等待
    db_result, pool_result, cancel_result, runtime_result = await asyncio.gather(
        asyncio.to_thread(_check_database),
        asyncio.to_thread(_check_session_pool),
        asyncio.to_thread(_check_cancellation_service),
        asyncio.to_thread(_check_runtime_harness),
    )

    checks = {
        "database": db_result,
        "session_pool": pool_result,
        "cancellation_service": cancel_result,
        "runtime_harness": runtime_result,
    }

    # 有任一子系统 error 则整体 degraded
    all_ok = all(v.get("status") == "ok" for v in checks.values())
    overall = "ok" if all_ok else "degraded"

    payload = {
        "status": overall,
        "version": _SERVICE_VERSION,
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "checks": checks,
    }

    http_status = 200 if overall == "ok" else 503
    return JSONResponse(content=payload, status_code=http_status)


@health_router.get("/health/live", summary="存活检查（liveness probe）", include_in_schema=False)
async def liveness() -> JSONResponse:
    """轻量存活检查，只要进程正常响应即返回 200，供 k8s liveness probe 使用。"""
    return JSONResponse(content={"status": "ok"})


@health_router.get("/health/ready", summary="就绪检查（readiness probe）", include_in_schema=False)
async def readiness() -> JSONResponse:
    """就绪检查，验证 DB 和 SessionPool 就绪后才返回 200，供 k8s readiness probe 使用。"""
    import asyncio
    db_result, pool_result = await asyncio.gather(
        asyncio.to_thread(_check_database),
        asyncio.to_thread(_check_session_pool),
    )
    all_ok = db_result.get("status") == "ok" and pool_result.get("status") == "ok"
    payload = {
        "status": "ok" if all_ok else "not_ready",
        "database": db_result,
        "session_pool": pool_result,
    }
    return JSONResponse(content=payload, status_code=200 if all_ok else 503)
