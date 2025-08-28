from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.security import verify_token
from db.mysql import get_db
from schemas.response_model import ResponseModel
from schemas.user_info_schemas import UserInfoLogin, UserInfoCreate, UserInfoUpdate, UserInfoChangePassword
from services.user_info_service import user_info_service

user_router = APIRouter(prefix="/user", tags=["用户信息"])

"""
用户信息接口
"""


@user_router.post("/create", response_model=ResponseModel, summary="创建用户")
def create_user_info(req: UserInfoCreate, db: Session = Depends(get_db)):
    """
    创建用户
    """
    user_info_resp = user_info_service.create_user(db, req)
    return ResponseModel(data=user_info_resp)


@user_router.post("/login", response_model=ResponseModel, summary="用户登录")
def login_user_info(req: UserInfoLogin, db: Session = Depends(get_db)):
    """
    用户登录
    """
    user_info_resp = user_info_service.login(db, req)
    return ResponseModel.success(data=user_info_resp)


@user_router.get("/detail", response_model=ResponseModel, summary="获取用户信息")
def get_user_info(db: Session = Depends(get_db),
                  user_id: int = Depends(verify_token)):
    """
    获取用户信息
    """
    user_info_resp = user_info_service.get_user_by_id(db, user_id=user_id)
    return ResponseModel(data=user_info_resp)


@user_router.put("/{id}", response_model=ResponseModel, summary="更新用户信息")
def get_user_info(req: UserInfoUpdate, db: Session = Depends(get_db),
                  user_id: int = Depends(verify_token)):
    """
    更新用户信息
    """
    user_info_resp = user_info_service.update_user(db, user_id=user_id, user_update=req)
    return ResponseModel.success(data=user_info_resp)


@user_router.post("/change-password", response_model=ResponseModel, summary="修改密码")
def get_user_info(req: UserInfoChangePassword, db: Session = Depends(get_db),
                  user_id: int = Depends(verify_token)):
    """
    修改密码
    """
    user_info_resp = user_info_service.change_password(db, user_id=user_id, user_update=req)
    return ResponseModel.success(data=user_info_resp)


@user_router.post("/refresh-token", response_model=ResponseModel, summary="刷新token")
def refresh_token(db: Session = Depends(get_db),
                  user_id: int = Depends(verify_token)):
    """
    token 刷新
    """
    token = user_info_service.refresh_token_by_user_id(db, user_id=user_id)
    return ResponseModel.success(data=token)
