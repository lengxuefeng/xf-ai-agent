from sqlalchemy import Column, String, DateTime, func, BigInteger

from db import Base

"""
模型配置表
"""


class ModelSetting(Base):
    __tablename__ = 't_model_setting'  # 指定数据库表名

    id = Column(BigInteger, primary_key=True, autoincrement=True, doc='主键')
    user_id = Column(BigInteger, nullable=False, doc='用户ID')
    model_name = Column(String(50), nullable=False, doc='模型名称')
    model_type = Column(String(50), nullable=False, doc='模型类型')
    model_url = Column(String(255), nullable=False, doc='模型地址')
    model_params = Column(String(255), nullable=False, doc='模型参数')
    model_desc = Column(String(255), nullable=False, doc='模型描述')
    create_time = Column(DateTime, server_default=func.now(), doc='创建时间')
    update_time = Column(DateTime, server_default=func.now(), onupdate=func.now(), doc='更新时间')

    def __repr__(self):
        return (f"<UserModel("
                f"id={self.id}, "
                f"user_id={self.user_id}, "
                f"model_name={self.model_name}, "
                f"model_type={self.model_type}, "
                f"model_url={self.model_url}, "
                f"model_params={self.model_params}, "
                f"model_desc={self.model_desc}, "
                f"create_time={self.create_time}, "
                f"update_time={self.update_time})"
                f">")
