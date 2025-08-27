
"""
自定义异常
"""


class BusinessException(Exception):
    """自定义业务异常"""
    def __init__(self, code: int = 500, message: str = "业务异常"):
        self.code = code
        self.message = message
        super().__init__(message)


class SecurityException(BusinessException):
    """
    安全相关异常
    """
    def __init__(self, message: str = "安全验证失败", code: int = 401):
        super().__init__(code, message)


class AuthenticationException(SecurityException):
    """
    认证失败异常
    """
    def __init__(self, message: str = "认证失败"):
        super().__init__(message, 401)


class AuthorizationException(SecurityException):
    """
    授权失败异常
    """
    def __init__(self, message: str = "授权失败"):
        super().__init__(message, 403)


class TokenExpiredException(AuthenticationException):
    """
    令牌过期异常
    """
    def __init__(self, message: str = "令牌已过期"):
        super().__init__(message)


class InvalidTokenException(AuthenticationException):
    """
    无效令牌异常
    """
    def __init__(self, message: str = "无效的令牌"):
        super().__init__(message)


class PasswordValidationException(BusinessException):
    """
    密码验证异常
    """
    def __init__(self, message: str = "密码验证失败"):
        super().__init__(400, message)
