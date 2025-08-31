from sqlalchemy import Column, String, DateTime, func, BigInteger, Boolean, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship

from db.mysql import Base

"""
用户模型设置表
"""


class UserModel(Base):
    __tablename__ = 't_user_model'  # 指定数据库表名

    id = Column(BigInteger, primary_key=True, autoincrement=True, doc='主键')
    user_id = Column(BigInteger, nullable=False, doc='用户ID')
    model_setting_id = Column(BigInteger, ForeignKey('t_model_setting.id'), nullable=False, doc='关联的系统模型配置ID')
    service_name = Column(String(100), nullable=False, doc='用户自定义服务名称')
    selected_model = Column(String(100), nullable=False, doc='用户选择的具体模型名称')
    api_key = Column(String(500), nullable=False, doc='用户的API密钥')
    api_url = Column(String(500), doc='用户自定义API地址（可选，覆盖系统默认）')
    custom_config = Column(JSON, doc='用户自定义配置，JSON格式')
    is_active = Column(Boolean, default=False, doc='是否为当前激活的模型配置')
    create_time = Column(DateTime, server_default=func.now(), doc='创建时间')
    update_time = Column(DateTime, server_default=func.now(), onupdate=func.now(), doc='更新时间')
    
    # 关联系统模型配置
    model_setting = relationship("ModelSetting", back_populates="user_models")

    def __repr__(self):
        return (f"<UserModel("
                f"id={self.id}, "
                f"user_id={self.user_id}, "
                f"model_setting_id={self.model_setting_id}, "
                f"service_name={self.service_name}, "
                f"selected_model={self.selected_model}, "
                f"api_key={self.api_key[:10]}..., "
                f"api_url={self.api_url}, "
                f"is_active={self.is_active}, "
                f"create_time={self.create_time}, "
                f"update_time={self.update_time})"
                f">")
