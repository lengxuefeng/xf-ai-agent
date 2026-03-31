# -*- coding: utf-8 -*-
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

from harness.exec.policy import ExecPolicy
from harness.exec.result import ExecResult
from harness.workspace.path_guard import workspace_path_guard
import json


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

    def run_python_with_auto_fix(
        self,
        code: str,
        llm: Any,
        *,
        cwd: Optional[str] = None,
        workspace_root: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: int = 2,
        require_json_output: bool = False,
    ) -> ExecResult:
        """执行 Python代码。若遇到运行错误或所需的 JSON 解析失败，则将报错抛回 LLM 自动修复。"""
        current_code = code
        last_result = None

        for attempt in range(max_retries + 1):
            result = self.run_python_code(
                current_code,
                cwd=cwd,
                workspace_root=workspace_root,
                timeout_seconds=timeout_seconds,
            )
            last_result = result
            error_to_fix = None

            if result.success:
                if require_json_output:
                    try:
                        json.loads(result.stdout)
                        return result
                    except json.JSONDecodeError as exc:
                        error_to_fix = f"输出结果并非有效 JSON 格式，解析报错: {exc}\n实际输出:\n{result.stdout}"
                else:
                    return result
            else:
                error_to_fix = result.stderr or result.stdout or "Unknown execution error"

            if attempt >= max_retries or not error_to_fix:
                break

            # 交给 LLM 修复
            try:
                from langchain_core.prompts import ChatPromptTemplate
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "你是一名资深 Python 工程师，负责审查与修正自动生成的出错代码。请输出修复后的完整代码，务必不要解释、也不要包含除 Markdown ```python 外的任何文字。"),
                    ("human", f"原代码：\n```python\n{{code}}\n```\n\n执行/解析报错：\n{{error}}\n\n请修改上述代码解决该报错，并重新输出完整的代码。"),
                ])
                chain = prompt | llm
                response = chain.invoke({"code": current_code, "error": error_to_fix})
                content = str(getattr(response, "content", "")).strip()
                
                if "```python" in content:
                    content = content.split("```python")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                if content:
                    current_code = content
                else:
                    break
            except Exception:
                break

        return last_result


exec_runner = ExecRunner()
