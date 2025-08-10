# schemas/user_model.py
from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional

from schemas.base import BaseSchema


class UserModelBase(BaseModel):
    """
    用户模型基础模型 - 不含user_id，用于创建
    """
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
    更新用户模型 - 不允许更新user_id
    """
    model_setting_id: Optional[int] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    api_url: Optional[str] = None


class UserModel(BaseSchema):
    """
    数据库模型对应的Pydantic模型
    """
    id: int
    user_id: int
    create_time: datetime
    update_time: datetime


class UserModelOut(BaseSchema):
    """
    用户模型输出模型 - 包含所有字段
    """
    id: int
    user_id: int
    create_time: datetime
    update_time: datetime
