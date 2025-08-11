from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.security import verify_token
from db import get_db
from schemas.chat_schemas import ResponseModel
from schemas.model_setting_schemas import ModelSettingCreate, ModelSettingUpdate, ModelSettingOut
from services.model_setting_service import model_setting_service

router = APIRouter(prefix="/user_setting", tags=["用户设置"])

"""
模型配置接口
"""

@router.get("/", response_model=ResponseModel[List[ModelSettingOut]], summary="获取当前用户的所有模型配置")
def get_model_settings(db: Session = Depends(get_db), user_id: int = Depends(verify_token)):
    """
    根据Token获取当前用户的所有模型配置
    """
    # 从服务层调用获取模型配置的方法
    model_settings = model_setting_service.get_model_settings(db, user_id=user_id)
    # 返回包含模型配置列表的响应
    return ResponseModel(data=model_settings)


@router.post("/", response_model=ResponseModel[ModelSettingOut], summary="创建新的模型配置")
def create_model_setting(model_setting_req: ModelSettingCreate, db: Session = Depends(get_db), user_id: int = Depends(verify_token)):
    """
    为用户创建一个新的模型配置
    """
    # 从服务层调用创建模型配置的方法
    model_setting = model_setting_service.create_model_setting(db, model_setting=model_setting_req, user_id=user_id)
    # 返回新创建的模型配置
    return ResponseModel(data=model_setting)


@router.put("/{setting_id}", response_model=ResponseModel[ModelSettingOut], summary="更新模型配置")
def update_model_setting(setting_id: int, model_setting_req: ModelSettingUpdate, db: Session = Depends(get_db), Authorization: str = Depends(verify_token)):
    """
    更新指定ID的模型配置
    """
    # 从服务层调用更新模型配置的方法
    updated_setting = model_setting_service.update_model_setting(db, id=setting_id, model_setting=model_setting_req)
    # 返回更新后的模型配置
    return ResponseModel(data=updated_setting)


@router.delete("/{setting_id}", response_model=ResponseModel, summary="删除模型配置")
def delete_model_setting(setting_id: int, db: Session = Depends(get_db), Authorization: str = Depends(verify_token)):
    """
    删除指定ID的模型配置
    """
    # 从服务层调用删除模型配置的方法
    model_setting_service.remove_model_setting(db, id=setting_id)
    # 返回成功的响应
    return ResponseModel(data={"message": "删除成功"})
