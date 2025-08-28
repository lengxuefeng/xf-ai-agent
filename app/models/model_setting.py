from typing import List, Optional
from sqlalchemy import Column, String, DateTime, func, BigInteger, Boolean, Text, JSON
from sqlalchemy.types import TypeDecorator
from sqlalchemy.dialects.mysql import JSON as MYSQL_JSON
import json

from db.mysql import Base

"""
模型配置表
"""


class JSONType(TypeDecorator):
    """
    自定义JSON类型，用于处理MySQL JSON字段的序列化和反序列化
    """
    impl = JSON
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value) if isinstance(value, str) else value
        return value


class ModelSetting(Base):
    __tablename__ = 't_model_setting'  # 指定数据库表名

    id = Column(BigInteger, primary_key=True, autoincrement=True, doc='主键ID，自增')
    user_id = Column(BigInteger, nullable=False, doc='用户ID，关联用户表')
    service_name = Column(String(100), nullable=False, doc='服务名称，如OpenAI、Google Gemini等')
    service_type = Column(String(50), nullable=False, doc='服务类型标识，如openai、gemini等，用于程序识别')
    service_url = Column(String(500), nullable=False, doc='API服务基础地址，如https://api.openai.com/v1')
    api_key_template = Column(String(100), doc='API密钥格式模板，如sk-xxx、AIxxx等，用于提示用户')
    icon = Column(String(50), default='FiCpu', doc='服务图标名称，使用Feather Icons图标库')
    models = Column(JSONType, nullable=False, doc='支持的模型列表，JSON格式存储，如["gpt-4","gpt-3.5-turbo"]')
    description = Column(Text, doc='服务描述信息，可选字段')
    is_enabled = Column(Boolean, default=True, doc='是否启用该服务，true为启用，false为禁用')
    create_time = Column(DateTime, server_default=func.now(), doc='创建时间，自动记录')
    update_time = Column(DateTime, server_default=func.now(), onupdate=func.now(), doc='更新时间，自动更新')

    def __repr__(self):
        return (f"<ModelSetting("
                f"id={self.id}, "
                f"user_id={self.user_id}, "
                f"service_name={self.service_name}, "
                f"service_type={self.service_type}, "
                f"service_url={self.service_url}, "
                f"api_key_template={self.api_key_template}, "
                f"icon={self.icon}, "
                f"models={self.models}, "
                f"description={self.description}, "
                f"is_enabled={self.is_enabled}, "
                f"create_time={self.create_time}, "
                f"update_time={self.update_time})"
                f">")
