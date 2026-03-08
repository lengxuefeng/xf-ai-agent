from fastapi import APIRouter, Depends

from core.security import verify_token
from schemas.response_model import ResponseModel
from services.route_metrics_service import route_metrics_service
from services.semantic_cache_service import semantic_cache_service

metrics_router = APIRouter()


@metrics_router.get("/metrics/router", response_model=ResponseModel, summary="获取路由指标快照")
def get_router_metrics(user_id: int = Depends(verify_token)):
    """
    获取 Domain Router / Intent Router 的运行指标。
    """
    _ = user_id
    data = route_metrics_service.snapshot()
    data["semantic_cache"] = semantic_cache_service.snapshot()
    return ResponseModel.success(data)
