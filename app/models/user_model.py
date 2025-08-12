from sqlalchemy import Column, String, DateTime, func, BigInteger

from db.mysql import Base

"""
用户模型设置表
"""


class UserModel(Base):
    __tablename__ = 't_user_model'  # 指定数据库表名

    id = Column(BigInteger, primary_key=True, autoincrement=True, doc='主键')
    user_id = Column(BigInteger, nullable=False, doc='用户ID')
    model_setting_id = Column(BigInteger, nullable=False, doc='模型配置ID')
    model_name = Column(String(100), nullable=False, doc='模型名称，多个逗号分隔')
    api_key = Column(String(255), nullable=False, doc='api密钥')
    api_url = Column(String(255), nullable=False, doc='api地址')
    create_time = Column(DateTime, server_default=func.now(), doc='创建时间')
    update_time = Column(DateTime, server_default=func.now(), onupdate=func.now(), doc='更新时间')

    def __repr__(self):
        return (f"<UserModel("
                f"id={self.id}, "
                f"user_id={self.user_id}, "
                f"model_setting_id={self.model_setting_id}, "
                f"model_name={self.model_name}, "
                f"api_key={self.api_key}, "
                f"api_url={self.api_url}, "
                f"create_time={self.create_time}, "
                f"update_time={self.update_time})"
                f">")
