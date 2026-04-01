# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from sqlalchemy.orm import Session

from db.crud import user_mcp_db
from models.schemas.user_mcp_schemas import (
    UserMCPConnectionTestRequest,
    UserMCPCreate,
    UserMCPUpdate,
)
from common.utils.custom_logger import get_logger

log = get_logger(__name__)


class UserMCPService:
    """用户 MCP 配置服务层。"""

    def get_user_mcps(self, db: Session, user_id: int, active_only: bool = False):
        if active_only:
            return user_mcp_db.get_active_by_user_id(db, user_id=user_id)
        return user_mcp_db.get_by_user_id(db, user_id=user_id)

    def get_user_mcp_by_id(self, db: Session, mcp_id: int, user_id: Optional[int] = None):
        user_mcp = user_mcp_db.get(db, mcp_id)
        if not user_mcp:
            return None
        if user_id is not None and user_mcp.user_id != user_id:
            log.warning(f"越权访问拦截：用户 {user_id} 试图访问 MCP 配置 {mcp_id}")
            return None
        return user_mcp

    def create_user_mcp(self, db: Session, user_mcp: UserMCPCreate, user_id: int):
        create_data = self._normalize_create_payload(user_mcp).model_dump()
        create_data["user_id"] = user_id
        created_mcp = user_mcp_db.create(db, obj_in=create_data)
        log.info(f"用户 {user_id} 创建 MCP 配置成功: mcp_id={created_mcp.id}")
        return created_mcp

    def update_user_mcp(self, db: Session, mcp_id: int, user_mcp: UserMCPUpdate, user_id: int):
        existing_mcp = self.get_user_mcp_by_id(db, mcp_id, user_id)
        if not existing_mcp:
            raise ValueError(f"MCP 配置不存在或无权操作: ID={mcp_id}")

        update_data = user_mcp.model_dump(exclude_unset=True)
        merged_payload = {
            "name": existing_mcp.name,
            "transport": existing_mcp.transport,
            "url": existing_mcp.url,
            "command": existing_mcp.command,
            "args": existing_mcp.args,
            "is_active": existing_mcp.is_active,
        }
        merged_payload.update(update_data)
        normalized_payload = self._normalize_create_payload(UserMCPCreate.model_validate(merged_payload)).model_dump()
        updated_mcp = user_mcp_db.update(db, db_obj=existing_mcp, obj_in=normalized_payload)
        log.info(f"用户 {user_id} 更新 MCP 配置成功: mcp_id={mcp_id}")
        return updated_mcp

    def remove_user_mcp(self, db: Session, mcp_id: int, user_id: int):
        existing_mcp = self.get_user_mcp_by_id(db, mcp_id, user_id)
        if not existing_mcp:
            raise ValueError(f"MCP 配置不存在或无权操作: ID={mcp_id}")
        removed_mcp = user_mcp_db.remove(db, id=mcp_id)
        log.info(f"用户 {user_id} 删除 MCP 配置成功: mcp_id={mcp_id}")
        return removed_mcp

    async def ping_mcp_server(self, mcp_config: UserMCPConnectionTestRequest | UserMCPCreate) -> bool:
        normalized = self._normalize_connection_payload(mcp_config)
        try:
            if normalized.transport == "sse":
                async with sse_client(normalized.url, timeout=5) as streams:
                    read_stream, write_stream = streams
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        return True

            server_params = StdioServerParameters(
                command=normalized.command,
                args=normalized.args or [],
            )
            async with stdio_client(server_params) as streams:
                read_stream, write_stream = streams
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    return True
        except Exception as exc:
            log.warning(f"MCP 服务连通性测试失败: transport={normalized.transport}, error={exc}")
            raise RuntimeError(f"MCP 服务连接失败: {exc}") from exc

    @staticmethod
    def _normalize_create_payload(mcp_config: UserMCPCreate) -> UserMCPCreate:
        return UserMCPCreate.model_validate(mcp_config.model_dump())

    @staticmethod
    def _normalize_connection_payload(
        mcp_config: UserMCPConnectionTestRequest | UserMCPCreate,
    ) -> UserMCPConnectionTestRequest:
        payload = mcp_config.model_dump()
        return UserMCPConnectionTestRequest.model_validate(payload)


user_mcp_service = UserMCPService()
