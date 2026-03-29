# -*- coding: utf-8 -*-
"""页面终端接口。

提供受控命令执行能力，所有命令都必须在用户绑定的工作目录内运行。
"""
from fastapi import APIRouter, Depends, HTTPException

from core.security import verify_token
from runtime.exec.command_session_service import command_session_service
from runtime.workspace.manager import workspace_manager
from schemas.terminal_schemas import (
    TerminalDirectoryCreateRequest,
    TerminalCommandStartRequest,
    TerminalEntryDeleteRequest,
    TerminalEntryRenameRequest,
    TerminalFileCreateRequest,
    TerminalFileSaveRequest,
    TerminalStopRequest,
    TerminalWorkspaceBindRequest,
)

terminal_router = APIRouter(prefix="/terminal", tags=["页面终端"])


@terminal_router.post("/workspace/bind", summary="绑定聊天会话工作目录")
async def bind_workspace(
    req: TerminalWorkspaceBindRequest,
    _user_id: int = Depends(verify_token),
):
    """绑定并校验某个聊天会话的工作目录。"""
    try:
        return {"code": 200, "message": "工作目录绑定成功", "data": workspace_manager.bind_external_workspace(req.session_id, req.workspace_root)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@terminal_router.get("/workspace/{session_id}", summary="获取聊天会话当前工作目录")
async def get_workspace(
    session_id: str,
    _user_id: int = Depends(verify_token),
):
    """返回页面终端当前绑定的工作目录信息。"""
    return {"code": 200, "message": "获取成功", "data": workspace_manager.describe_external_workspace(session_id)}


@terminal_router.get("/fs/tree/{session_id}", summary="列出工作目录内容")
async def list_workspace_tree(
    session_id: str,
    path: str = "",
    _user_id: int = Depends(verify_token),
):
    """返回已绑定工作目录下某个目录的子项列表。"""
    try:
        return {"code": 200, "message": "获取成功", "data": workspace_manager.list_workspace_directory(session_id, path)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@terminal_router.get("/fs/file/{session_id}", summary="读取工作目录文件")
async def read_workspace_file(
    session_id: str,
    path: str,
    _user_id: int = Depends(verify_token),
):
    """读取工作目录内的文本文件内容。"""
    try:
        return {"code": 200, "message": "读取成功", "data": workspace_manager.read_workspace_file(session_id, path)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@terminal_router.put("/fs/file", summary="保存工作目录文件")
async def save_workspace_file(
    req: TerminalFileSaveRequest,
    _user_id: int = Depends(verify_token),
):
    """保存工作目录内的文本文件内容。"""
    try:
        return {"code": 200, "message": "保存成功", "data": workspace_manager.save_workspace_file(req.session_id, req.path, req.content)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@terminal_router.post("/fs/file", summary="创建工作目录文件")
async def create_workspace_file(
    req: TerminalFileCreateRequest,
    _user_id: int = Depends(verify_token),
):
    """在工作目录内创建文本文件。"""
    try:
        return {"code": 200, "message": "创建成功", "data": workspace_manager.create_workspace_file(req.session_id, req.path, req.content)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@terminal_router.post("/fs/directory", summary="创建工作目录目录")
async def create_workspace_directory(
    req: TerminalDirectoryCreateRequest,
    _user_id: int = Depends(verify_token),
):
    """在工作目录内创建目录。"""
    try:
        return {"code": 200, "message": "创建成功", "data": workspace_manager.create_workspace_directory(req.session_id, req.path)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@terminal_router.put("/fs/entry/rename", summary="重命名工作目录条目")
async def rename_workspace_entry(
    req: TerminalEntryRenameRequest,
    _user_id: int = Depends(verify_token),
):
    """在工作目录内重命名或移动文件、目录。"""
    try:
        return {
            "code": 200,
            "message": "重命名成功",
            "data": workspace_manager.rename_workspace_entry(req.session_id, req.path, req.new_path),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@terminal_router.post("/fs/entry/delete", summary="删除工作目录条目")
async def delete_workspace_entry(
    req: TerminalEntryDeleteRequest,
    _user_id: int = Depends(verify_token),
):
    """在工作目录内删除文件或目录。"""
    try:
        return {
            "code": 200,
            "message": "删除成功",
            "data": workspace_manager.delete_workspace_entry(req.session_id, req.path),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@terminal_router.post("/session/start", summary="启动页面终端命令")
async def start_command(
    req: TerminalCommandStartRequest,
    _user_id: int = Depends(verify_token),
):
    """启动一条受控命令执行。"""
    try:
        workspace_manager.bind_external_workspace(req.session_id, req.workspace_root)
        snapshot = command_session_service.start_command(
            session_id=req.session_id,
            workspace_root=req.workspace_root,
            command_text=req.command_text,
            cwd=req.cwd,
        )
        return {"code": 200, "message": "命令已启动", "data": snapshot}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@terminal_router.get("/session/{command_id}", summary="查询命令执行状态")
async def get_command_session(
    command_id: str,
    _user_id: int = Depends(verify_token),
):
    """返回单条命令执行状态。"""
    session = command_session_service.get_session(command_id)
    if not session:
        raise HTTPException(status_code=404, detail="命令会话不存在")
    return {"code": 200, "message": "获取成功", "data": session}


@terminal_router.get("/session/latest/{session_id}", summary="获取会话最近一次命令状态")
async def get_latest_command_session(
    session_id: str,
    _user_id: int = Depends(verify_token),
):
    """返回某个聊天会话最近一次命令执行状态。"""
    session = command_session_service.get_latest_session(session_id)
    return {"code": 200, "message": "获取成功", "data": session}


@terminal_router.post("/session/{command_id}/stop", summary="停止命令执行")
async def stop_command_session(
    command_id: str,
    req: TerminalStopRequest,
    _user_id: int = Depends(verify_token),
):
    """停止正在运行的命令。"""
    session = command_session_service.stop_session(command_id)
    if not session:
        raise HTTPException(status_code=404, detail="命令会话不存在")
    return {"code": 200, "message": "停止信号已发送", "data": session}
