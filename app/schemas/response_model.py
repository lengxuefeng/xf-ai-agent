from typing import Any, Optional

from pydantic import BaseModel, Field

"""
统一返回对象
"""


class ResponseModel(BaseModel):
    """
    统一返回对象
    """
    code: int = Field(default=200, description="状态码")
    message: str = Field(default="success", description="响应信息")
    data: Optional[Any] = Field(default=None, description="数据内容")

    @classmethod
    def success(cls, data: Any = None, message: str = "success"):
        return cls(code=200, message=message, data=data)

    @classmethod
    def fail(cls, code: int, message: str):
        return cls(code=code, message=message, data=None)
