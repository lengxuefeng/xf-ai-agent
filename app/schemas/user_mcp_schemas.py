# schemas/user_mcp.py
from datetime import datetime
from typing import Optional, Dict, Any

from pydantic import BaseModel, ConfigDict

from schemas.base import BaseSchema


class UserMCPBase(BaseModel):
    """
    用户MCP设置基础模型
    """
    user_id: int
    mcp_setting_json: Dict[str, Any]


class UserMCPCreate(UserMCPBase):
    """
    创建用户MCP设置模型
    """
    pass


class UserMCPUpdate(BaseModel):
    """
    更新用户MCP设置模型
    """
    user_id: Optional[int] = None
    mcp_setting_json: Optional[Dict[str, Any]] = None


class UserMCP(BaseSchema):
    """
    用户MCP设置模型
    """
    id: int
    create_time: datetime
    update_time: datetime
