# -*- coding: utf-8 -*-
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

from runtime.exec.policy import ExecPolicy
from runtime.exec.result import ExecResult
from runtime.workspace.path_guard import workspace_path_guard


class ExecRunner:
    """统一命令与 Python 代码执行器。"""

    def __init__(self, policy: ExecPolicy | None = None) -> None:
        self.policy = policy or ExecPolicy()

    def run_command(
        self,
        command: Iterable[str],
        *,
        cwd: Optional[str] = None,
        workspace_root: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> ExecResult:
        """执行一条受控命令，并附带工作目录边界校验。"""
        argv = [str(part) for part in command]
        if not self.policy.allows(argv):
            return ExecResult(
                success=False,
                command=argv,
                exit_code=126,
                stderr="命令未被当前执行策略允许。",
                meta={"reason": "policy_denied"},
            )

        try:
            resolved_root = None
            resolved_cwd = None
            if workspace_root:
                resolved_root = workspace_path_guard.resolve_workspace_root(workspace_root)
                resolved_cwd = workspace_path_guard.resolve_execution_cwd(resolved_root, cwd)
            elif cwd:
                resolved_cwd = Path(cwd).expanduser().resolve(strict=True)
        except Exception as exc:  # noqa: BLE001
            return ExecResult(
                success=False,
                command=argv,
                exit_code=125,
                stderr=str(exc),
                meta={"reason": "workspace_guard"},
            )

        start = time.time()
        try:
            completed = subprocess.run(
                argv,
                cwd=str(resolved_cwd) if resolved_cwd else cwd,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout_seconds or self.policy.timeout_seconds,
                check=False,
            )
            duration_ms = int((time.time() - start) * 1000)
            return ExecResult(
                success=completed.returncode == 0,
                command=argv,
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration_ms=duration_ms,
                meta={
                    "cwd": str(resolved_cwd) if resolved_cwd else (cwd or ""),
                    "workspace_root": str(resolved_root) if workspace_root else "",
                },
            )
        except subprocess.TimeoutExpired as exc:
            duration_ms = int((time.time() - start) * 1000)
            return ExecResult(
                success=False,
                command=argv,
                exit_code=124,
                stdout=str(exc.stdout or ""),
                stderr=str(exc.stderr or ""),
                duration_ms=duration_ms,
                timed_out=True,
                meta={
                    "reason": "timeout",
                    "cwd": str(resolved_cwd) if resolved_cwd else (cwd or ""),
                    "workspace_root": str(resolved_root) if workspace_root else "",
                },
            )

    def run_python_code(
        self,
        code: str,
        *,
        cwd: Optional[str] = None,
        workspace_root: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> ExecResult:
        """将 Python 代码写入临时文件并在受控目录内运行。"""
        with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as handle:
            handle.write(str(code or ""))
            temp_path = Path(handle.name)
        try:
            return self.run_command(
                [sys.executable, str(temp_path)],
                cwd=cwd,
                workspace_root=workspace_root,
                timeout_seconds=timeout_seconds,
            )
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


exec_runner = ExecRunner()
