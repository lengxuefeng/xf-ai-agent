from fastapi import APIRouter

from schemas.response_model import ResponseModel
from schemas.user_info_schemas import UserInfoCreate
from services.user_info_service import user_info_service

chat_history_router = APIRouter(prefix="/history", tags=["历史对话信息"])

"""
历史对话信息接口
"""


@chat_history_router.post("/create", response_model=dict)
def create_user_info(user_info_req: UserInfoCreate):
    """
    创建用户
    """
    pass


