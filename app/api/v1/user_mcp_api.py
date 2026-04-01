# -*- coding: utf-8 -*-
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from common.core.security import verify_token
from db import get_db
from models.schemas.response_model import ResponseModel
from models.schemas.user_mcp_schemas import (
    UserMCPConnectionTestRequest,
    UserMCPCreate,
    UserMCPOut,
    UserMCPUpdate,
)
from services.user_mcp_service import user_mcp_service

router = APIRouter(prefix="/mcp", tags=["MCP Servers"])


@router.get("/", response_model=ResponseModel[List[UserMCPOut]], summary="获取当前用户的 MCP 配置列表")
def get_user_mcps(
    db: Session = Depends(get_db),
    user_id: int = Depends(verify_token),
):
    user_mcps = user_mcp_service.get_user_mcps(db, user_id=user_id)
    return ResponseModel.success(data=user_mcps)


@router.post("/", response_model=ResponseModel[UserMCPOut], summary="创建新的 MCP 配置")
def create_user_mcp(
    user_mcp_req: UserMCPCreate,
    db: Session = Depends(get_db),
    user_id: int = Depends(verify_token),
):
    user_mcp = user_mcp_service.create_user_mcp(db, user_mcp=user_mcp_req, user_id=user_id)
    return ResponseModel.success(data=user_mcp, message="MCP 配置创建成功")


@router.post("/test-connection", response_model=ResponseModel[bool], summary="测试 MCP 服务连通性")
async def test_mcp_connection(
    mcp_config: UserMCPConnectionTestRequest,
    user_id: int = Depends(verify_token),
):
    del user_id
    try:
        result = await user_mcp_service.ping_mcp_server(mcp_config)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ResponseModel.success(data=result, message="MCP 服务连接成功")


@router.put("/{mcp_id}", response_model=ResponseModel[UserMCPOut], summary="更新 MCP 配置")
def update_user_mcp(
    mcp_id: int,
    user_mcp_req: UserMCPUpdate,
    db: Session = Depends(get_db),
    user_id: int = Depends(verify_token),
):
    try:
        updated_mcp = user_mcp_service.update_user_mcp(db, mcp_id=mcp_id, user_mcp=user_mcp_req, user_id=user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ResponseModel.success(data=updated_mcp, message="MCP 配置更新成功")


@router.delete("/{mcp_id}", response_model=ResponseModel[dict], summary="删除 MCP 配置")
def delete_user_mcp(
    mcp_id: int,
    db: Session = Depends(get_db),
    user_id: int = Depends(verify_token),
):
    try:
        user_mcp_service.remove_user_mcp(db, mcp_id=mcp_id, user_id=user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ResponseModel.success(data={"message": "删除成功"})
