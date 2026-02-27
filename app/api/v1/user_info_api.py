# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.security import verify_token
from db import get_db
from schemas.response_model import ResponseModel
from schemas.user_info_schemas import UserInfoLogin, UserInfoCreate, UserInfoUpdate, UserInfoChangePassword
from services.user_info_service import user_info_service

user_router = APIRouter(prefix="/user", tags=["用户信息"])

@user_router.post("/create", response_model=ResponseModel, summary="创建用户")
def create_user_info(req: UserInfoCreate, db: Session = Depends(get_db)):
    user_info_resp = user_info_service.create_user(db, req)
    return ResponseModel.success(data=user_info_resp)

@user_router.post("/login", response_model=ResponseModel, summary="用户登录")
def login_user_info(req: UserInfoLogin, db: Session = Depends(get_db)):
    user_info_resp = user_info_service.login(db, req)
    return ResponseModel.success(data=user_info_resp)

@user_router.get("/detail", response_model=ResponseModel, summary="获取用户信息")
def get_user_info(db: Session = Depends(get_db), user_id: int = Depends(verify_token)):
    user_info_resp = user_info_service.get_user_by_id(db, user_id=user_id)
    return ResponseModel.success(data=user_info_resp)

@user_router.put("/update", response_model=ResponseModel, summary="更新用户信息")
def update_user_info(req: UserInfoUpdate, db: Session = Depends(get_db), user_id: int = Depends(verify_token)):
    # 【修复】函数名改为了 update_user_info 避免冲突，路由改成了语义更明确的 /update
    user_info_resp = user_info_service.update_user(db, user_id=user_id, user_update=req)
    return ResponseModel.success(data=user_info_resp)

@user_router.post("/change-password", response_model=ResponseModel, summary="修改密码")
def change_user_password(req: UserInfoChangePassword, db: Session = Depends(get_db), user_id: int = Depends(verify_token)):
    # 【修复】函数名改为了 change_user_password 避免冲突
    user_info_resp = user_info_service.change_password(db, user_id=user_id, user_update=req)
    return ResponseModel.success(data=user_info_resp)