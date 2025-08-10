# schemas/user_model.py
from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional

from schemas.base import BaseSchema


class UserModelBase(BaseModel):
    """
    用户模型基础模型
    """
    user_id: int
    model_setting_id: int
    model_name: str
    api_key: str
    api_url: str


class UserModelCreate(UserModelBase):
    """
    创建用户模型
    """
    pass


class UserModelUpdate(BaseModel):
    """
    更新用户模型
    """
    user_id: Optional[int] = None
    model_setting_id: Optional[int] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    api_url: Optional[str] = None


class UserModel(BaseSchema):
    """
    用户模型
    """
    id: int
    create_time: datetime
    update_time: datetime
