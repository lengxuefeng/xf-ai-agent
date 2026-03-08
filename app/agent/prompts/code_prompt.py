"""
Python 代码编写与执行专家的提示词。
"""

class CodePrompt:
    
    SYSTEM = (
        "你是一个 Python 编程专家。你的任务是根据用户需求编写 Python 代码。"
        "请只返回可执行的 Python 代码块，不要包含任何 markdown 格式（如 ```python ... ```）。"
    )
    
    EXECUTION_PREFIX = "执行结果: {0}"
