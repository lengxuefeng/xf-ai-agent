from sqlalchemy import Column, String, DateTime, func, BigInteger

from db.mysql import Base

# 定义基础类


"""
用户信息表
"""


class UserInfo(Base):
    __tablename__ = 't_user_info'  # 指定数据库表名

    id = Column(BigInteger, primary_key=True, autoincrement=True, doc='主键')
    token = Column(String(255), nullable=True)
    nick_name = Column(String(50), nullable=False, doc='昵称')
    user_name = Column(String(50), nullable=False, doc='用户名')
    phone = Column(String(50), nullable=False, doc='手机号')
    password = Column(String(255), nullable=False, doc='密码')
    create_time = Column(DateTime, server_default=func.now(), doc='创建时间')
    update_time = Column(DateTime, server_default=func.now(), onupdate=func.now(), doc='更新时间')

    def __repr__(self):
        return f"<UserInfo(id={self.id}, token={self.token}, nick_name={self.nick_name})>"
