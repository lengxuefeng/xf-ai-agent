# app/schemas/common.py
from pydantic import BaseModel, Field, constr


class BaseRequestParams(BaseModel):
    token: constr(min_length=1) = Field(..., description="访问令牌，必填")


class PageParams(BaseModel):
    page: int = Field(1, ge=1, description="页码，从1开始")
    size: int = Field(10, ge=1, le=100, description="每页数量，最大100")
