# schemas/model_setting.py
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from schemas.base import BaseSchema


class ModelSettingBase(BaseModel):
    """
    模型设置基础模型 - 不含user_id
    """
    model_name: str
    model_type: str
    model_url: str
    model_params: str
    model_desc: str


class ModelSettingCreate(ModelSettingBase):
    """
    创建模型设置模型
    """
    pass


class ModelSettingUpdate(BaseModel):
    """
    更新模型设置模型 - 不允许更新user_id
    """
    model_name: Optional[str] = None
    model_type: Optional[str] = None
    model_url: Optional[str] = None
    model_params: Optional[str] = None
    model_desc: Optional[str] = None


class ModelSettingResp(BaseModel):
    """
    模型设置响应
    """
    id: int
    model_name: str
    model_type: str
    model_url: str
    model_params: str
    model_desc: str


class ModelSetting(ModelSettingBase):
    """
    数据库模型对应的Pydantic模型
    """
    id: int
    user_id: int
    create_time: datetime
    update_time: datetime
    model_config = ConfigDict(from_attributes=True)


class ModelSettingOut(BaseSchema):
    """
    模型设置输出 - 包含所有字段
    """
    id: int
    user_id: int
    create_time: datetime
    update_time: datetime
