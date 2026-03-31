# -*- coding: utf-8 -*-
"""页面终端相关接口模型。"""
from typing import Optional

from pydantic import Field

from models.schemas.base import BaseSchema


class TerminalCommandStartRequest(BaseSchema):
    """启动页面终端命令的请求体。"""

    session_id: str = Field(..., description="聊天会话 ID")
    workspace_root: str = Field(..., description="用户指定的受控工作目录")
    command_text: str = Field(..., description="要执行的命令文本")
    cwd: Optional[str] = Field(default=None, description="相对工作目录的子路径，可选")


class TerminalWorkspaceBindRequest(BaseSchema):
    """绑定聊天会话工作目录的请求体。"""

    session_id: str = Field(..., description="聊天会话 ID")
    workspace_root: str = Field(..., description="用户指定的工作目录")


class TerminalStopRequest(BaseSchema):
    """停止页面终端命令的请求体。"""

    session_id: str = Field(..., description="聊天会话 ID")


class TerminalFileSaveRequest(BaseSchema):
    """保存工作区文件的请求体。"""

    session_id: str = Field(..., description="聊天会话 ID")
    path: str = Field(..., description="相对工作目录的文件路径")
    content: str = Field(default="", description="要写回文件的内容")


class TerminalFileCreateRequest(BaseSchema):
    """创建工作区文件的请求体。"""

    session_id: str = Field(..., description="聊天会话 ID")
    path: str = Field(..., description="相对工作目录的新文件路径")
    content: str = Field(default="", description="新文件初始内容")


class TerminalDirectoryCreateRequest(BaseSchema):
    """创建工作区目录的请求体。"""

    session_id: str = Field(..., description="聊天会话 ID")
    path: str = Field(..., description="相对工作目录的新目录路径")


class TerminalEntryRenameRequest(BaseSchema):
    """重命名或移动工作区条目的请求体。"""

    session_id: str = Field(..., description="聊天会话 ID")
    path: str = Field(..., description="当前条目的相对路径")
    new_path: str = Field(..., description="新的相对路径")


class TerminalEntryDeleteRequest(BaseSchema):
    """删除工作区条目的请求体。"""

    session_id: str = Field(..., description="聊天会话 ID")
    path: str = Field(..., description="要删除的相对路径")
