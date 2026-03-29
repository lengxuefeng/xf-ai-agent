# -*- coding: utf-8 -*-
"""受控命令会话服务。

本服务用于在页面中提供类似终端的体验，但底层仍然是后端启动的受控子进程：
1. 只允许在用户指定且合法的工作目录中运行命令；
2. 只允许策略白名单中的可执行程序；
3. 通过内存态保存最近输出，供前端轮询展示。
"""
from __future__ import annotations

import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from runtime.exec.policy import ExecPolicy
from runtime.workspace.path_guard import workspace_path_guard


def _utc_now() -> float:
    """返回浮点时间戳，便于前端展示运行状态。"""
    return time.time()


@dataclass(slots=True)
class CommandSession:
    """单条命令执行会话。"""

    command_id: str
    session_id: str
    command_text: str
    argv: List[str]
    workspace_root: str
    cwd: str
    status: str = "pending"
    output: str = ""
    exit_code: Optional[int] = None
    error: str = ""
    created_at: float = field(default_factory=_utc_now)
    started_at: float = 0.0
    finished_at: float = 0.0
    pid: Optional[int] = None

    def append_output(self, chunk: str) -> None:
        """向会话输出缓冲区追加内容，并截断到可控大小。"""
        text = str(chunk or "")
        if not text:
            return
        self.output = (self.output + text)[-60000:]

    def to_dict(self) -> Dict[str, object]:
        """序列化当前命令会话状态。"""
        return {
            "command_id": self.command_id,
            "session_id": self.session_id,
            "command_text": self.command_text,
            "argv": list(self.argv),
            "workspace_root": self.workspace_root,
            "cwd": self.cwd,
            "status": self.status,
            "output": self.output,
            "exit_code": self.exit_code,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "pid": self.pid,
        }


class CommandSessionService:
    """统一管理页面终端的命令执行生命周期。"""

    def __init__(self, policy: ExecPolicy | None = None) -> None:
        """初始化命令会话服务。"""
        self._policy = policy or ExecPolicy()
        self._lock = threading.RLock()
        self._sessions: Dict[str, CommandSession] = {}
        self._latest_by_chat: Dict[str, str] = {}
        self._processes: Dict[str, subprocess.Popen[str]] = {}

    def start_command(
        self,
        *,
        session_id: str,
        workspace_root: str,
        command_text: str,
        cwd: str | None = None,
    ) -> Dict[str, object]:
        """启动一条新的受控命令。"""
        argv = self._policy.parse_command_text(command_text)
        resolved_root = workspace_path_guard.resolve_workspace_root(workspace_root)
        resolved_cwd = workspace_path_guard.resolve_execution_cwd(resolved_root, cwd)

        if not self._policy.allows(argv):
            raise ValueError(self._policy.explain_rejection(argv))

        command_id = uuid.uuid4().hex
        record = CommandSession(
            command_id=command_id,
            session_id=session_id,
            command_text=command_text,
            argv=list(argv),
            workspace_root=str(resolved_root),
            cwd=str(resolved_cwd),
        )
        with self._lock:
            self._sessions[command_id] = record
            self._latest_by_chat[session_id] = command_id

        worker = threading.Thread(
            target=self._run_process,
            args=(record.command_id,),
            daemon=True,
            name=f"terminal-{command_id[:8]}",
        )
        worker.start()
        return record.to_dict()

    def get_session(self, command_id: str) -> Optional[Dict[str, object]]:
        """读取单条命令会话快照。"""
        with self._lock:
            session = self._sessions.get(command_id)
            return session.to_dict() if session else None

    def get_latest_session(self, session_id: str) -> Optional[Dict[str, object]]:
        """返回某个聊天会话最近一次命令执行状态。"""
        with self._lock:
            latest_id = self._latest_by_chat.get(session_id)
            if not latest_id:
                return None
            session = self._sessions.get(latest_id)
            return session.to_dict() if session else None

    def stop_session(self, command_id: str) -> Optional[Dict[str, object]]:
        """停止正在运行的命令。"""
        with self._lock:
            process = self._processes.get(command_id)
            session = self._sessions.get(command_id)

        if session is None:
            return None

        if process and process.poll() is None:
            process.terminate()
            session.status = "stopping"
        return session.to_dict()

    def stats(self) -> Dict[str, object]:
        """返回命令会话服务的基础状态。"""
        with self._lock:
            active = sum(1 for process in self._processes.values() if process.poll() is None)
            total = len(self._sessions)
        return {
            "total_sessions": total,
            "active_sessions": active,
            "policy": self._policy.describe(),
            "path_guard": workspace_path_guard.describe(),
        }

    def _run_process(self, command_id: str) -> None:
        """后台线程实际运行命令，并持续收集输出。"""
        with self._lock:
            session = self._sessions.get(command_id)
        if session is None:
            return

        try:
            session.status = "running"
            session.started_at = _utc_now()
            process = subprocess.Popen(
                session.argv,
                cwd=session.cwd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            session.pid = process.pid
            with self._lock:
                self._processes[command_id] = process

            if process.stdout is not None:
                try:
                    for line in process.stdout:
                        session.append_output(line)
                finally:
                    process.stdout.close()

            exit_code = process.wait(timeout=self._policy.timeout_seconds)
            session.exit_code = exit_code
            session.status = "completed" if exit_code == 0 else "failed"
        except subprocess.TimeoutExpired:
            session.error = "命令执行超时，已被系统终止。"
            session.status = "failed"
            session.exit_code = 124
            process = self._processes.get(command_id)
            if process and process.poll() is None:
                process.kill()
        except Exception as exc:  # noqa: BLE001
            session.error = str(exc)
            session.status = "failed"
            session.exit_code = 1
        finally:
            session.finished_at = _utc_now()
            with self._lock:
                process = self._processes.pop(command_id, None)
            if process and process.poll() is None:
                try:
                    process.kill()
                except Exception:
                    pass


command_session_service = CommandSessionService()
