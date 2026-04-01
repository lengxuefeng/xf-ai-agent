# -*- coding: utf-8 -*-
from datetime import datetime
import json
from typing import Literal, Optional

from pydantic import BaseModel, field_validator, model_validator

from models.schemas.base import BaseSchema


TransportLiteral = Literal["sse", "stdio"]


class _UserMCPPayloadMixin(BaseModel):
    transport: TransportLiteral = "sse"
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[list[str]] = None
    is_active: bool = True

    @field_validator("url", "command", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("args", mode="before")
    @classmethod
    def normalize_args(cls, value):
        if value in (None, "", []):
            return None
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                return [part for part in candidate.split() if part]
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
            return None
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return value

    @model_validator(mode="after")
    def validate_transport_fields(self):
        if self.transport == "sse" and not self.url:
            raise ValueError("SSE 模式下 url 不能为空")
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio 模式下 command 不能为空")

        if self.transport == "sse":
            self.command = None
            self.args = None
        else:
            self.url = None

        return self


class UserMCPBase(_UserMCPPayloadMixin):
    """用户 MCP 配置基础模型。"""

    name: str

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("name 不能为空")
        return normalized


class UserMCPCreate(UserMCPBase):
    """创建用户 MCP 配置。"""


class UserMCPUpdate(BaseModel):
    """更新用户 MCP 配置。"""

    name: Optional[str] = None
    transport: Optional[TransportLiteral] = None
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[list[str]] = None
    is_active: Optional[bool] = None

    @field_validator("name", "url", "command", mode="before")
    @classmethod
    def normalize_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("args", mode="before")
    @classmethod
    def normalize_args(cls, value):
        return _UserMCPPayloadMixin.normalize_args(value)


class UserMCPConnectionTestRequest(_UserMCPPayloadMixin):
    """MCP 连通性测试请求。"""

    name: Optional[str] = None


class UserMCP(BaseSchema):
    """数据库模型对应的 Pydantic 模型。"""

    id: int
    user_id: int
    name: str
    transport: TransportLiteral
    url: Optional[str]
    command: Optional[str]
    args: Optional[list[str]]
    is_active: bool
    create_time: datetime
    update_time: datetime


class UserMCPOut(UserMCP):
    """用户 MCP 输出模型。"""
