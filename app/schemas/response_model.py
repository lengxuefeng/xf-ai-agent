from typing import Any, Optional, TypeVar, Generic

from pydantic import BaseModel, Field

# 定义泛型类型变量
T = TypeVar('T')

"""
统一返回对象
"""


class ResponseModel(BaseModel, Generic[T]):
    """
    统一返回对象
    """
    code: int = Field(default=200, description="状态码")
    message: str = Field(default="success", description="响应信息")
    data: Optional[T] = Field(default=None, description="数据内容")

    @classmethod
    def success(cls, data: Optional[T] = None, message: str = "success"):
        return cls(code=200, message=message, data=data)

    @classmethod
    def fail(cls, code: int, message: str):
        return cls(code=code, message=message, data=None)
