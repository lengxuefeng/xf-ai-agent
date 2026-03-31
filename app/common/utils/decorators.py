"""
装饰器工具模块
"""
import functools
import logging
from typing import Callable, Any

import streamlit as st

logger = logging.getLogger(__name__)


class _CallableDecoratorWrapper:
    def __init__(self, func: Callable):
        self._func = func
        functools.update_wrapper(self, func)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return functools.partial(self.__call__, instance)


class _ErrorHandledCallable(_CallableDecoratorWrapper):
    def __init__(self, func: Callable, show_error: bool):
        super().__init__(func)
        self._show_error = show_error

    def __call__(self, *args, **kwargs) -> Any:
        try:
            return self._func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{self._func.__name__} 执行失败: {str(e)}")
            if self._show_error:
                st.error(f"操作失败: {str(e)}")
            raise


class _LoggedCallable(_CallableDecoratorWrapper):
    def __call__(self, *args, **kwargs) -> Any:
        logger.info(f"开始执行 {self._func.__name__}")
        try:
            result = self._func(*args, **kwargs)
            logger.info(f"{self._func.__name__} 执行成功")
            return result
        except Exception as e:
            logger.error(f"{self._func.__name__} 执行失败: {str(e)}")
            raise


def _build_error_handler_wrapper(func: Callable, *, show_error: bool) -> Callable:
    return _ErrorHandledCallable(func, show_error)


def error_handler(show_error: bool = True) -> Callable:
    """
    统一错误处理装饰器
    Args:
        show_error:是否在UI中显示错误信息
    Returns:
        装饰器函数
    """
    return functools.partial(_build_error_handler_wrapper, show_error=show_error)


def log_execution(func: Callable) -> Callable:
    """
    记录函数执行的装饰器
    Args:
        func:被装饰的函数
    Returns:
        装饰器函数
    """
    return _LoggedCallable(func)
