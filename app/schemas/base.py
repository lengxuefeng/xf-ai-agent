# app/schemas/base.py

import datetime

from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,  # 允许从 ORM 对象读取属性
        json_encoders={datetime.datetime: lambda v: v.strftime('%Y-%m-%d %H:%M:%S')}  # 日期格式化
    )
