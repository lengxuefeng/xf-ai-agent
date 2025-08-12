from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.security import verify_token
from db.mysql import get_db
from schemas.response_model import ResponseModel
from schemas.user_mcp_schemas import UserMCPCreate, UserMCPUpdate, UserMCPOut
from services.user_mcp_service import user_mcp_service

router = APIRouter(prefix="/user_mcp", tags=["用户MCP配置"])

"""
用户MCP配置接口
"""

@router.get("/", response_model=ResponseModel[List[UserMCPOut]], summary="获取当前用户的所有MCP配置")
def get_user_mcps(db: Session = Depends(get_db), user_id: int = Depends(verify_token)):
    """
    根据Token获取当前用户的所有MCP配置
    """
    # 从服务层调用获取用户MCP配置的方法
    user_mcps = user_mcp_service.get_user_mcps(db, user_id=user_id)
    # 返回包含MCP配置列表的响应
    return ResponseModel(data=user_mcps)


@router.post("/", response_model=ResponseModel[UserMCPOut], summary="创建新的用户MCP配置")
def create_user_mcp(user_mcp_req: UserMCPCreate, db: Session = Depends(get_db), user_id: int = Depends(verify_token)):
    """
    为用户创建一个新的MCP配置
    """
    # 从服务层调用创建用户MCP配置的方法
    user_mcp = user_mcp_service.create_user_mcp(db, user_mcp=user_mcp_req, user_id=user_id)
    # 返回新创建的MCP配置
    return ResponseModel(data=user_mcp)


@router.put("/{mcp_id}", response_model=ResponseModel[UserMCPOut], summary="更新用户MCP配置")
def update_user_mcp(mcp_id: int, user_mcp_req: UserMCPUpdate, db: Session = Depends(get_db), Authorization: str = Depends(verify_token)):
    """
    更新指定ID的用户MCP配置
    """
    # 从服务层调用更新用户MCP配置的方法
    updated_mcp = user_mcp_service.update_user_mcp(db, id=mcp_id, user_mcp=user_mcp_req)
    # 返回更新后的MCP配置
    return ResponseModel(data=updated_mcp)


@router.delete("/{mcp_id}", response_model=ResponseModel, summary="删除用户MCP配置")
def delete_user_mcp(mcp_id: int, db: Session = Depends(get_db), Authorization: str = Depends(verify_token)):
    """
    删除指定ID的用户MCP配置
    """
    # 从服务层调用删除用户MCP配置的方法
    user_mcp_service.remove_user_mcp(db, id=mcp_id)
    # 返回成功的响应
    return ResponseModel(data={"message": "删除成功"})
