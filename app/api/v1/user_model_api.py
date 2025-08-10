from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.security import verify_token
from db import get_db
from schemas.schemas import ResponseModel
from schemas.user_model_schemas import UserModelCreate, UserModelUpdate, UserModelOut
from services.user_model_service import user_model_service

# 创建一个APIRouter实例，用于定义用户模型相关的路由
router = APIRouter()

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
    # 从服务层调用创建用户模型的方法，并传入user_id
    user_model = user_model_service.create_user_model(db, user_model=user_model_req, user_id=user_id)
    # 返回新创建的模型配置
    return ResponseModel(data=user_model)


@router.put("/{model_id}", response_model=ResponseModel[UserModelOut], summary="更新用户模型配置")
def update_user_model(model_id: int, user_model_req: UserModelUpdate, db: Session = Depends(get_db), Authorization: str = Depends(verify_token)):
    """
    更新指定ID的用户模型配置
    """
    # 从服务层调用更新用户模型的方法
    updated_model = user_model_service.update_user_model(db, id=model_id, user_model=user_model_req)
    # 返回更新后的模型配置
    return ResponseModel(data=updated_model)


@router.delete("/{model_id}", response_model=ResponseModel, summary="删除用户模型配置")
def delete_user_model(model_id: int, db: Session = Depends(get_db), Authorization: str = Depends(verify_token)):
    """
    删除指定ID的用户模型配置
    """
    # 从服务层调用删除用户模型的方法
    user_model_service.remove_user_model(db, id=model_id)
    # 返回成功的响应
    return ResponseModel(data={"message": "删除成功"})
