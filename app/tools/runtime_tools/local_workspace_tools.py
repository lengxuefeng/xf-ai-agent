# -*- coding: utf-8 -*-
from __future__ import annotations

from langchain_core.tools import tool

from common.utils.code_tools import execute_python_code_tool  # noqa: F401
from harness.exec.policy import ExecPolicy
from harness.exec.runner import exec_runner
from harness.workspace.manager import workspace_manager
from models.schemas.tool_input_schemas import (
    WorkspaceCreateDirectoryToolInput,
    WorkspaceDeleteEntryToolInput,
    WorkspaceListDirectoryToolInput,
    WorkspaceMoveEntryToolInput,
    WorkspaceReadFileToolInput,
    WorkspaceRunCommandToolInput,
    WorkspaceWriteFileToolInput,
)
from tools.runtime_tools.tool_registry import runtime_tool_registry


_command_policy = ExecPolicy()


@tool(
    "workspace_list_directory",
    args_schema=WorkspaceListDirectoryToolInput,
    description="列出当前工作目录下指定目录的文件与子目录。",
)
def workspace_list_directory(workspace_root: str, path: str = "") -> dict:
    return workspace_manager.list_workspace_directory_by_root(workspace_root, path)


@tool(
    "workspace_read_file",
    args_schema=WorkspaceReadFileToolInput,
    description="读取当前工作目录中的文本文件内容。",
)
def workspace_read_file(workspace_root: str, path: str) -> dict:
    return workspace_manager.read_workspace_file_by_root(workspace_root, path)


@tool(
    "workspace_write_file",
    args_schema=WorkspaceWriteFileToolInput,
    description="写入当前工作目录中的文本文件；若文件不存在会自动创建。",
)
def workspace_write_file(workspace_root: str, path: str, content: str = "") -> dict:
    return workspace_manager.write_workspace_file_by_root(workspace_root, path, content)


@tool(
    "workspace_create_directory",
    args_schema=WorkspaceCreateDirectoryToolInput,
    description="在当前工作目录中创建新目录。",
)
def workspace_create_directory(workspace_root: str, path: str) -> dict:
    return workspace_manager.create_workspace_directory_by_root(workspace_root, path)


@tool(
    "workspace_move_entry",
    args_schema=WorkspaceMoveEntryToolInput,
    description="在当前工作目录中重命名或移动文件、目录。",
)
def workspace_move_entry(workspace_root: str, path: str, new_path: str) -> dict:
    return workspace_manager.move_workspace_entry_by_root(workspace_root, path, new_path)


@tool(
    "workspace_delete_entry",
    args_schema=WorkspaceDeleteEntryToolInput,
    description="删除当前工作目录中的文件或目录。",
)
def workspace_delete_entry(workspace_root: str, path: str) -> dict:
    return workspace_manager.delete_workspace_entry_by_root(workspace_root, path)


@tool(
    "workspace_run_command",
    args_schema=WorkspaceRunCommandToolInput,
    description="在当前工作目录中执行一条受控命令，并返回标准输出、错误输出和退出码。",
)
def workspace_run_command(
    workspace_root: str,
    command: str,
    cwd: str | None = None,
    timeout_seconds: float | None = None,
) -> dict:
    argv = _command_policy.parse_command_text(command)
    return exec_runner.run_command(
        argv,
        cwd=cwd,
        workspace_root=workspace_root,
        timeout_seconds=timeout_seconds,
    ).to_dict()


runtime_tool_registry.register_native_tool(
    workspace_list_directory,
    category="workspace",
    source="runtime.workspace",
    description="列出当前工作目录下的目录内容。",
    local_only=True,
)
runtime_tool_registry.register_native_tool(
    workspace_read_file,
    category="workspace",
    source="runtime.workspace",
    description="读取当前工作目录中的文本文件。",
    local_only=True,
)
runtime_tool_registry.register_native_tool(
    workspace_write_file,
    category="workspace",
    source="runtime.workspace",
    description="写入当前工作目录中的文本文件。",
    local_only=True,
)
runtime_tool_registry.register_native_tool(
    workspace_create_directory,
    category="workspace",
    source="runtime.workspace",
    description="创建当前工作目录中的新目录。",
    local_only=True,
)
runtime_tool_registry.register_native_tool(
    workspace_move_entry,
    category="workspace",
    source="runtime.workspace",
    description="移动或重命名当前工作目录中的条目。",
    local_only=True,
)
runtime_tool_registry.register_native_tool(
    workspace_delete_entry,
    category="workspace",
    source="runtime.workspace",
    description="删除当前工作目录中的条目。",
    local_only=True,
)
runtime_tool_registry.register_native_tool(
    workspace_run_command,
    category="exec",
    source="runtime.exec",
    requires_approval=True,
    description="在当前工作目录中执行受控命令。",
    local_only=True,
)
