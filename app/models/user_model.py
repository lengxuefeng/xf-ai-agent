# -*- coding: utf-8 -*-
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import String, DateTime, func, BigInteger, Boolean, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db import Base
from models.model_setting import ModelSetting

"""
【学习笔记】用户模型设置表 (SQLAlchemy 2.0 风格)
优势：使用 Mapped[type] 进行强类型声明，完全消灭 IDE 里的类型警告，
      并在运行时和 Pydantic BaseModel 完美契合。
"""


class UserModel(Base):
    __tablename__ = 't_user_model'  # 指定数据库表名

    # --- 核心主键与关联 ID ---
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, doc='主键')
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, doc='用户ID')
    model_setting_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey('t_model_setting.id'), nullable=False, doc='关联的系统模型配置ID'
    )

    # --- 业务必填字段 ---
    service_name: Mapped[str] = mapped_column(String(100), nullable=False, doc='用户自定义服务名称')
    selected_model: Mapped[str] = mapped_column(String(100), nullable=False, doc='用户选择的具体模型名称')
    api_key: Mapped[str] = mapped_column(String(500), nullable=False, doc='用户的API密钥')

    # --- 业务可选字段 (使用 Optional 标记) ---
    api_url: Mapped[Optional[str]] = mapped_column(String(500), doc='用户自定义API地址（可选，覆盖系统默认）')
    custom_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, doc='用户自定义配置，JSON格式')

    # --- 状态与时间标记 ---
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, doc='是否为当前激活的模型配置')

    # func.now() 由数据库服务器生成时间，绝对精准
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), doc='创建时间')
    update_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), doc='更新时间'
    )

    # --- 关系映射 (Relationships) ---
    # 注意："ModelSetting" 使用字符串是为了防止两个 Model 文件循环导入 (Circular Import)
    model_setting: Mapped["ModelSetting"] = relationship("ModelSetting", back_populates="user_models")

    def __repr__(self) -> str:
        """
        优化打印输出，在调试或看日志时更直观。
        (不打印 api_key 和 custom_config，防止敏感信息泄露或日志过长)
        """
        return (f"<UserModel("
                f"id={self.id}, "
                f"user_id={self.user_id}, "
                f"model_setting_id={self.model_setting_id}, "
                f"service_name='{self.service_name}', "
                f"selected_model='{self.selected_model}', "
                f"is_active={self.is_active})>")