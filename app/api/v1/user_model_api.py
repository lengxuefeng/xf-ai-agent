from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.security import verify_token
from db import get_db
from schemas.response_model import ResponseModel
from schemas.user_model_schemas import UserModelCreate, UserModelUpdate, UserModelOut
from services.user_model_service import user_model_service

from utils.custom_logger import get_logger, LogTarget

log = get_logger(__name__)
router = APIRouter(prefix="/user_model", tags=["用户模型配置"])

"""
用户模型接口
"""

@router.get("/", response_model=ResponseModel[List[UserModelOut]], summary="获取当前用户的所有模型配置")
def get_user_models(db: Session = Depends(get_db), user_id: int = Depends(verify_token)):
    """
    根据Token获取当前用户的所有模型配置
    """
    # 从服务层调用获取用户模型的方法
    user_models = user_model_service.get_user_models(db, user_id=user_id)
    # 返回包含模型列表的响应
    return ResponseModel(data=user_models)


@router.post("/", response_model=ResponseModel[UserModelOut], summary="创建新的用户模型配置")
def create_user_model(user_model_req: UserModelCreate, db: Session = Depends(get_db), user_id: int = Depends(verify_token)):
    """
    为用户创建一个新的模型配置
    """
    # 从服务层调用创建用户模型的方法，并传入user_id，支持多个模型激活
    user_model = user_model_service.create_user_model(db, user_model=user_model_req, user_id=user_id, allow_multiple=True)
    # 返回新创建的模型配置
    return ResponseModel(data=user_model)


@router.put("/{model_id}", response_model=ResponseModel[UserModelOut], summary="更新用户模型配置")
def update_user_model(model_id: int, user_model_req: UserModelUpdate, db: Session = Depends(get_db), user_id: int = Depends(verify_token)):
    """
    更新指定ID的用户模型配置
    """
    log.info(f"收到更新模型配置请求: id={model_id}, model={user_model_req.selected_model}", target=LogTarget.LOG)
    try:
        updated_model = user_model_service.update_user_model(
            db,
            id=model_id,
            user_model=user_model_req,
            user_id=user_id,
            allow_multiple=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    log.info(f"模型配置更新完成: id={model_id}", target=LogTarget.LOG)
    # 返回更新后的模型配置
    return ResponseModel(data=updated_model)


@router.put("/{model_id}/activate", response_model=ResponseModel[UserModelOut], summary="激活用户模型配置")
def activate_user_model(model_id: int, db: Session = Depends(get_db), user_id: int = Depends(verify_token)):
    """
    激活指定ID的用户模型配置，支持多个模型同时激活
    """
    try:
        activated_model = user_model_service.activate_user_model(db, id=model_id, user_id=user_id, allow_multiple=True)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ResponseModel(data=activated_model)


@router.put("/{model_id}/deactivate", response_model=ResponseModel[UserModelOut], summary="取消激活用户模型配置")
def deactivate_user_model(model_id: int, db: Session = Depends(get_db), user_id: int = Depends(verify_token)):
    """
    取消激活指定ID的用户模型配置
    """
    try:
        deactivated_model = user_model_service.deactivate_user_model(db, id=model_id, user_id=user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ResponseModel(data=deactivated_model)


@router.get("/active", response_model=ResponseModel[UserModelOut], summary="获取当前激活的用户模型配置")
def get_active_user_model(db: Session = Depends(get_db), user_id: int = Depends(verify_token)):
    """
    获取用户当前激活的模型配置
    """
    active_model = user_model_service.get_active_user_model(db, user_id=user_id)
    return ResponseModel(data=active_model)


@router.delete("/{model_id}", response_model=ResponseModel, summary="删除用户模型配置")
def delete_user_model(model_id: int, db: Session = Depends(get_db), user_id: int = Depends(verify_token)):
    """
    删除指定ID的用户模型配置
    """
    try:
        user_model_service.remove_user_model(db, id=model_id, user_id=user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ResponseModel(data={"message": "删除成功"})
