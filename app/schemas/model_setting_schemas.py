# schemas/model_setting.py
from datetime import datetime
from typing import Optional, List, Dict

from pydantic import BaseModel

from schemas.base import BaseSchema


class ModelServiceBase(BaseModel):
    """
    模型服务基础模型
    """
    service_name: str
    service_type: str
    service_url: str
    api_key_template: str
    icon: str = "FiCpu"
    models: Dict[str, List[str]] = {
        "chat": [],
        "embedding": [],
        "vision": [],
        "other": []
    }
    description: Optional[str] = None
    is_enabled: bool = True


class ModelServiceCreate(ModelServiceBase):
    """
    创建模型服务模型
    """
    pass


class ModelServiceUpdate(BaseModel):
    """
    更新模型服务模型
    """
    service_name: Optional[str] = None
    service_type: Optional[str] = None
    service_url: Optional[str] = None
    api_key_template: Optional[str] = None
    icon: Optional[str] = None
    models: Optional[Dict[str, List[str]]] = None
    description: Optional[str] = None
    is_system_default: Optional[bool] = None
    is_enabled: Optional[bool] = None


class ModelServiceOut(BaseSchema):
    """
    系统模型服务输出模型
    """
    id: int
    service_name: str
    service_type: str
    service_url: str
    api_key_template: str
    icon: str
    models: Dict[str, List[str]]
    description: Optional[str]
    is_system_default: bool
    is_enabled: bool
    create_time: datetime
    update_time: datetime


class TestConnectionRequest(BaseModel):
    """
    测试连接请求模型
    """
    api_key: str


class ToggleServiceRequest(BaseModel):
    """
    启用/禁用服务请求模型
    """
    is_enabled: bool
