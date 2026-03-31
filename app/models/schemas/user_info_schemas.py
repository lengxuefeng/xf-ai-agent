# schemas/user_info.py
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, constr

from models.schemas.base import BaseSchema
from models.schemas.common import BaseRequestParams, PageParams


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


class UserInfoCreate(BaseModel):
    """
    创建用户模型，必填字段不能为空
    """
    nick_name: constr(min_length=2) = Field(..., description="昵称，不能为空")
    user_name: constr(min_length=2) = Field(..., description="用户名，不能为空")
    password: constr(min_length=6) = Field(..., description="密码，不能为空")
    phone: constr(min_length=11) = Field(..., description="手机号，不能为空")


class UserInfoUpdate(BaseModel):
    """
    更新用户模型
    """
    nick_name: Optional[str] = None
    user_name: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None


class UserInfoChangePassword(BaseModel):
    """
    修改密码模型
    """
    old_password: constr(min_length=6) = Field(..., description="旧密码，不能为空")
    new_password: constr(min_length=6) = Field(..., description="新密码，不能为空")


class TokenResponse(BaseModel):
    """
    Token响应模型
    """
    access_token: str = Field(..., description="访问令牌")
    refresh_token: str = Field(..., description="刷新令牌")
    token_type: str = Field(default="bearer", description="令牌类型")
    expires_in: int = Field(..., description="令牌过期时间（秒）")


class RefreshTokenRequest(BaseModel):
    """
    刷新令牌请求模型
    """
    refresh_token: str = Field(..., description="刷新令牌")


class PasswordResetRequest(BaseModel):
    """
    密码重置请求模型
    """
    phone: str = Field(..., description="手机号")


class PasswordResetConfirm(BaseModel):
    """
    密码重置确认模型
    """
    phone: str = Field(..., description="手机号")
    token: str = Field(..., description="重置令牌")
    new_password: constr(min_length=6) = Field(..., description="新密码")

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
