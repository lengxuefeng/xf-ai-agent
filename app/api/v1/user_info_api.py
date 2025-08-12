from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.security import verify_token
from db.mysql import get_db
from schemas.response_model import ResponseModel
from schemas.user_info_schemas import UserInfoLogin, UserInfoCreate
from services.user_info_service import user_info_service

user_router = APIRouter(prefix="/user", tags=["用户信息"])

"""
用户信息接口
"""


@user_router.post("/create", response_model=ResponseModel)
def create_user_info(user_info_req: UserInfoCreate, db: Session = Depends(get_db)):
    """
    创建用户
    """
    user_info_resp = user_info_service.create_user(db, user_info_req)
    return ResponseModel(data=user_info_resp)


@user_router.post("/login", response_model=ResponseModel)
def login_user_info(user_info_req: UserInfoLogin, db: Session = Depends(get_db)):
    """
    用户登录
    """
    user_info_resp = user_info_service.login(db, user_info_req)
    return ResponseModel.success(data=user_info_resp)


@user_router.post("/user_info", response_model=ResponseModel)
def get_user_info(db: Session = Depends(get_db),
                  user_id: int = Depends(verify_token)):
    """
    获取用户信息
    """
    user_info_resp = user_info_service.get_user_by_id(db, user_id=user_id)
    return ResponseModel(data=user_info_resp)
