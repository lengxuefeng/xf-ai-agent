# -*- coding: utf-8 -*-
"""
用户 MCP 服务端配置模型。

该表负责持久化用户自定义的 MCP Server 连接信息，供运行期动态加载工具时查询。
当前仅支持两类传输方式：
1. `sse`: 通过 URL 连接远端 MCP Server
2. `stdio`: 通过本地命令拉起 MCP Server
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import List, Optional

from sqlalchemy import BigInteger, Boolean, CheckConstraint, DateTime, String, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from db import Base


class MCPTransport(StrEnum):
    """MCP 传输协议枚举。"""

    SSE = "sse"
    STDIO = "stdio"


class UserMCP(Base):
    """用户 MCP 服务端配置。"""

    __tablename__ = "t_user_mcp"
    __table_args__ = (
        CheckConstraint("transport IN ('sse', 'stdio')", name="ck_t_user_mcp_transport"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, doc="主键")
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True, doc="用户 ID")

    name: Mapped[str] = mapped_column(String(100), nullable=False, doc="MCP 服务名称")
    transport: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default=text("'sse'"),
        doc="传输协议：sse / stdio",
    )

    url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True, doc="SSE 模式下的服务地址")
    command: Mapped[Optional[str]] = mapped_column(String(500), nullable=True, doc="stdio 模式下的启动命令")
    args: Mapped[Optional[List[str]]] = mapped_column(JSONB, nullable=True, doc="stdio 模式下的命令参数")

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
        doc="是否启用",
    )

    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), doc="创建时间")
    update_time: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        doc="更新时间",
    )

    def __repr__(self) -> str:
        return (
            f"<UserMCP(id={self.id}, user_id={self.user_id}, name='{self.name}', "
            f"transport='{self.transport}', is_active={self.is_active})>"
        )
