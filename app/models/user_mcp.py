from sqlalchemy import Column, String, DateTime, func, BigInteger

from db import Base

"""
用户MCP设置表
"""


class UserMCP(Base):
    __tablename__ = 't_user_mcp'

    id = Column(BigInteger, primary_key=True, autoincrement=True, doc='主键')
    user_id = Column(BigInteger, nullable=False, doc='用户ID')
    mcp_setting_json = Column(String(255), nullable=False, doc='MCP配置JSON')
    create_time = Column(DateTime, server_default=func.now(), doc='创建时间')
    update_time = Column(DateTime, server_default=func.now(), onupdate=func.now(), doc='更新时间')

    def __repr__(self):
        return (f"<UserModel("
                f"id={self.id}, "
                f"user_id={self.user_id}, "
                f"mcp_setting_json={self.mcp_setting_json}, "
                f"create_time={self.create_time}, "
                f"update_time={self.update_time})"
                f">")
