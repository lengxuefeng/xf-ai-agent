from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.security import verify_token
from db import get_db
from schemas.model_setting_schemas import ModelServiceOut, ModelServiceCreate, ModelServiceUpdate, \
    TestConnectionRequest, ToggleServiceRequest
from schemas.response_model import ResponseModel
from services.model_setting_service import model_setting_service

router = APIRouter(prefix="/model_setting", tags=["用户设置"])

"""
模型配置接口
"""


@router.get("/", response_model=ResponseModel[List[ModelServiceOut]], summary="获取系统预定义的模型服务")
def get_model_services(db: Session = Depends(get_db), user_id: int = Depends(verify_token)):
    """
    获取系统预定义的模型服务配置（不再按用户过滤）
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        model_services = model_setting_service.get_model_services(db)
        logger.info(f"Retrieved {len(model_services)} model services")
        
        # 数据格式已在服务层处理，这里直接返回
        
        return ResponseModel(data=model_services)
    except Exception as e:
        logger.error(f"Error retrieving model services: {e}", exc_info=True)
        raise


@router.post("/", response_model=ResponseModel[ModelServiceOut], summary="创建新的模型服务")
def create_model_service(service_data: ModelServiceCreate, db: Session = Depends(get_db),
                         user_id: int = Depends(verify_token)):
    """
    为用户创建一个新的模型服务配置
    """
    model_service = model_setting_service.create_model_service(db, service_data=service_data, user_id=user_id)
    return ResponseModel(data=model_service)


@router.put("/{service_id}", response_model=ResponseModel[ModelServiceOut], summary="更新模型服务")
def update_model_service(service_id: int, service_data: ModelServiceUpdate, db: Session = Depends(get_db),
                         user_id: int = Depends(verify_token)):
    """
    更新指定ID的模型服务配置
    """
    updated_service = model_setting_service.update_model_service(db, id=service_id, service_data=service_data)
    return ResponseModel(data=updated_service)


@router.delete("/{service_id}", response_model=ResponseModel, summary="删除模型服务")
def delete_model_service(service_id: int, db: Session = Depends(get_db), user_id: int = Depends(verify_token)):
    """
    删除指定ID的模型服务配置
    """
    model_setting_service.remove_model_service(db, id=service_id)
    return ResponseModel(data={"message": "删除成功"})


@router.post("/{service_id}/test", response_model=ResponseModel, summary="测试模型服务连接")
def test_model_service_connection(service_id: int, test_data: TestConnectionRequest,
                                  db: Session = Depends(get_db), user_id: int = Depends(verify_token)):
    """
    测试指定模型服务的连接
    """
    result = model_setting_service.test_connection(db, service_id=service_id, api_key=test_data.api_key)
    return ResponseModel(data=result)


@router.put("/{service_id}/toggle", response_model=ResponseModel[ModelServiceOut], summary="启用/禁用模型服务")
def toggle_model_service(service_id: int, toggle_data: ToggleServiceRequest,
                         db: Session = Depends(get_db), user_id: int = Depends(verify_token)):
    """
    启用或禁用指定的模型服务
    """
    updated_service = model_setting_service.toggle_service(db, service_id=service_id, enabled=toggle_data.is_enabled)
    return ResponseModel(data=updated_service)
