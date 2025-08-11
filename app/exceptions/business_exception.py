
"""
自定义异常
"""


class BusinessException(Exception):
    """自定义业务异常"""
    def __init__(self, code: int = 500, message: str = "业务异常"):
        self.code = code
        self.message = message
        super().__init__(message)
