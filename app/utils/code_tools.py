import sys
import io
import traceback
from langchain_core.tools import tool

@tool
def execute_python_code(code: str) -> str:
    """
    执行 Python 代码并返回标准输出。
    注意：此工具在本地环境中运行，仅用于演示或受信任的内部使用。
    """
    # 捕获 stdout
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    
    # 简单的命名空间
    local_scope = {}
    
    try:
        # 尝试执行代码
        exec(code, {}, local_scope)
        sys.stdout = old_stdout
        return redirected_output.getvalue()
    except Exception as e:
        sys.stdout = old_stdout
        return f"代码执行出错: {str(e)}\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout
