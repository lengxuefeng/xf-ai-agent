# -*- coding: utf-8 -*-
import logging
from typing import Any

from fastapi import HTTPException

logger = logging.getLogger(__name__)


class BusinessExceptionHandler:
    """
    业务异常处理器，统一处理各种业务异常和错误转换
    """

    @staticmethod
    def handle_validation_error(error: ValueError) -> HTTPException:
        """
        处理输入验证错误
        
        Args:
            error: 验证错误对象
            
        Returns:
            HTTPException: 格式化的HTTP异常
        """
        logger.warning(f"输入验证错误: {str(error)}")
        return HTTPException(status_code=400, detail=str(error))

    @staticmethod
    def handle_business_error(error: Exception, context: str = "") -> HTTPException:
        """
        处理业务逻辑错误
        
        Args:
            error: 业务错误对象
            context: 错误上下文信息
            
        Returns:
            HTTPException: 格式化的HTTP异常
        """
        error_message = f"{context}: {str(error)}" if context else str(error)
        logger.error(f"业务逻辑错误: {error_message}")
        return HTTPException(status_code=500, detail=error_message)

    @staticmethod
    def handle_authentication_error(error: Exception) -> HTTPException:
        """
        处理认证相关错误
        
        Args:
            error: 认证错误对象
            
        Returns:
            HTTPException: 格式化的HTTP异常
        """
        logger.warning(f"认证错误: {str(error)}")
        return HTTPException(status_code=401, detail=f"认证失败: {str(error)}")

    @staticmethod
    def handle_authorization_error(error: Exception) -> HTTPException:
        """
        处理授权相关错误
        
        Args:
            error: 授权错误对象
            
        Returns:
            HTTPException: 格式化的HTTP异常
        """
        logger.warning(f"授权错误: {str(error)}")
        return HTTPException(status_code=403, detail=f"权限不足: {str(error)}")

    @staticmethod
    def handle_resource_not_found_error(resource_type: str, resource_id: Any = None) -> HTTPException:
        """
        处理资源未找到错误
        
        Args:
            resource_type: 资源类型
            resource_id: 资源ID
            
        Returns:
            HTTPException: 格式化的HTTP异常
        """
        message = f"{resource_type}未找到"
        if resource_id:
            message += f" (ID: {resource_id})"

        logger.warning(f"资源未找到: {message}")
        return HTTPException(status_code=404, detail=message)

    @staticmethod
    def handle_rate_limit_error(error: Exception) -> HTTPException:
        """
        处理频率限制错误
        
        Args:
            error: 频率限制错误对象
            
        Returns:
            HTTPException: 格式化的HTTP异常
        """
        logger.warning(f"频率限制错误: {str(error)}")
        return HTTPException(status_code=429, detail=f"请求过于频繁: {str(error)}")

    @staticmethod
    def safe_execute(func, *args, context: str = "", **kwargs) -> Any:
        """
        安全执行函数，自动处理异常
        
        Args:
            func: 要执行的函数
            *args: 函数位置参数
            context: 错误上下文
            **kwargs: 函数关键字参数
            
        Returns:
            Any: 函数执行结果
            
        Raises:
            HTTPException: 转换后的HTTP异常
        """
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            raise BusinessExceptionHandler.handle_validation_error(e)
        except Exception as e:
            raise BusinessExceptionHandler.handle_business_error(e, context)


# 创建全局实例
exception_handler = BusinessExceptionHandler()
