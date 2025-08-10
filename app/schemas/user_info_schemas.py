# schemas/user_info.py
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, constr

from schemas.base import BaseSchema
from schemas.common import BaseRequestParams, PageParams


class UserInfoBase(BaseModel):
    """
    用户信息基础模型
    """
    token: str
    nick_name: str
    user_name: str
    password: str


class UserInfoLogin(BaseModel):
    """
    用户登录模型
    """
    user_name: str
    password: str


class UserInfoCreate(UserInfoBase):
    """
    创建用户模型，必填字段不能为空
    """
    nick_name: constr(min_length=1) = Field(..., description="昵称，不能为空")
    user_name: constr(min_length=1) = Field(..., description="用户名，不能为空")
    password: constr(min_length=1) = Field(..., description="密码，不能为空")
    phone: constr(min_length=1) = Field(..., description="手机号，不能为空")


class UserInfoUpdate(BaseModel):
    """
    更新用户模型
    """
    nick_name: Optional[str] = None
    user_name: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None


class UserInfoQuery(BaseModel):
    nick_name: Optional[str] = None
    user_name: Optional[str] = None
    phone: Optional[str] = None


class UserInfoResp(BaseSchema):
    nick_name: str
    user_name: str
    phone: str
    token: str
    create_time: datetime
    update_time: datetime


class UserInfoPageQuery(UserInfoQuery, BaseRequestParams, PageParams):
    pass


class UserInfoRead(BaseSchema):
    """
    用户信息模型
    """
    id: int
    user_name: str
    nick_name: str
    token: str
    create_time: datetime
    update_time: datetime
