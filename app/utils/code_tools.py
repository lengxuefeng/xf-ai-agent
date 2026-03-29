"""代码执行工具统一封装。"""
from langchain_core.tools import tool

from runtime.exec.runner import exec_runner
from runtime.tools.tool_executor import tool_executor


def _execute_python_handler(
    code: str,
    cwd: str | None = None,
    workspace_root: str | None = None,
    timeout_seconds: float | None = None,
):
    """统一 Python 代码执行处理器。"""
    return exec_runner.run_python_code(
        code,
        cwd=cwd,
        workspace_root=workspace_root,
        timeout_seconds=timeout_seconds,
    ).to_dict()


tool_executor.register_handler("execute_python_code", _execute_python_handler)

def execute_python_code(
    code: str,
    *,
    cwd: str | None = None,
    workspace_root: str | None = None,
) -> str:
    """
    执行 Python 代码并返回标准输出。
    注意：此工具在本地环境中运行，仅用于演示或受信任的内部使用。
    """
    execution = tool_executor.execute(
        "execute_python_code",
        code=code,
        cwd=cwd,
        workspace_root=workspace_root,
    )
    if not execution.get("ok"):
        return f"代码执行出错: {execution.get('error', '未知错误')}"

    result = execution.get("result") or {}
    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    exit_code = int(result.get("exit_code") or 0)

    if result.get("success"):
        return stdout or "代码执行成功，但没有标准输出。"

    message = stderr or stdout or "代码执行失败，未返回更多信息。"
    return f"代码执行出错(exit_code={exit_code}): {message}"


execute_python_code_tool = tool("execute_python_code")(execute_python_code)
