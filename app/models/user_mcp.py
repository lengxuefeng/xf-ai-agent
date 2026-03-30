# -*- coding: utf-8 -*-
"""
用户MCP（Model Context Protocol）配置模型。

MCP是一种用于连接AI模型和外部数据源的协议。
用户可以配置自己的MCP服务器，让AI模型访问自定义的数据源。

设计要点：
1. 灵活的配置：支持多种MCP服务器配置
2. 用户隔离：每个用户有独立的MCP配置
3. JSON格式：配置以JSON字符串形式存储
4. 时间戳：记录创建和更新时间

使用场景：
- 用户配置自定义的数据库连接
- 用户配置API网关
- 用户配置其他外部数据源

配置格式示例：
{
    "servers": [
        {
            "name": "my_database",
            "type": "postgres",
            "connection_string": "postgresql://..."
        }
    ]
}
"""
from sqlalchemy import Column, String, DateTime, func, BigInteger

from db import Base


class UserMCP(Base):
    """
    用户MCP设置表。

    核心职责：
    1. 存储用户的MCP服务器配置
    2. 支持多种类型的MCP服务器
    3. 提供灵活的JSON配置格式

    设计理由：
    1. MCP是连接AI模型和外部数据源的标准协议
    2. 用户可能需要访问自定义的数据源
    3. 集中管理配置，方便维护

    字段说明：
    - id: 主键，自增
    - user_id: 用户ID
    - mcp_setting_json: MCP配置JSON字符串
    - create_time: 创建时间
    - update_time: 更新时间

    MCP配置结构：
    {
        "servers": [
            {
                "name": "server_name",
                "type": "postgres/redis/...",
                "host": "localhost",
                "port": 5432,
                "database": "dbname",
                "username": "user",
                "password": "pass"
            }
        ]
    }
    """
    __tablename__ = 't_user_mcp'

    # 主键
    id = Column(BigInteger, primary_key=True, autoincrement=True, doc='主键')

    # 用户标识
    user_id = Column(BigInteger, nullable=False, doc='用户ID')

    # MCP配置
    # 使用JSON字符串存储配置信息
    mcp_setting_json = Column(String(255), nullable=False, doc='MCP配置JSON')

    # 时间戳
    create_time = Column(DateTime, server_default=func.now(), doc='创建时间')
    update_time = Column(DateTime, server_default=func.now(), onupdate=func.now(), doc='更新时间')

    def __repr__(self):
        """对象字符串表示，用于调试。"""
        return (f"<UserModel("
                f"id={self.id}, "
                f"user_id={self.user_id}, "
                f"mcp_setting_json={self.mcp_setting_json}, "
                f"create_time={self.create_time}, "
                f"update_time={self.update_time})"
                f">")
