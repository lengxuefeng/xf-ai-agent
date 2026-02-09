# -*- coding: utf-8 -*-
"""
自定义日志处理器，将 logging 输出转换为 SSE 事件
支持灵活配置是否将日志输出到思考过程
"""
import json
import logging
from typing import Optional, Callable, List
from queue import Queue
from threading import Thread


class SSELoggingHandler(logging.Handler):
    """
    SSE 日志处理器，将日志输出转换为 SSE 事件并放入队列
    """

    # 需要捕获的日志名称模式
    CAPTURE_PATTERNS = [
        'agent',           # agent 模块
        'schemas',         # schemas 模块
        'utils',           # utils 模块
        'httpx',           # HTTP 请求
    ]

    # 日志级别到图标和类型的映射
    LOG_LEVEL_CONFIG = {
        logging.DEBUG: ('🔍', 'info'),
        logging.INFO: ('ℹ️', 'info'),
        logging.WARNING: ('⚠️', 'warning'),
        logging.ERROR: ('❌', 'error'),
        logging.CRITICAL: ('🔴', 'error'),
    }

    # 日志消息的简化映射（去除冗余前缀）
    MESSAGE_SIMPLIFICATIONS = {
        'MIDDLEWARE: 正在从目录初始化技能': '🔧 初始化技能',
        'SKILL_SUCCESS: 已加载技能': '✅ 加载技能',
        'TOOL_USE: Agent 正在加载技能': '🛠️ 加载技能',
        '正在尝试读取文件': '📄 读取文件',
        'HTTP Request:': '🌐 HTTP请求',
        'POST': '',
    }

    def __init__(
        self,
        output_queue: Optional['Queue[str]'] = None,
        level=logging.INFO,
        default_show_in_thinking: bool = True
    ):
        """
        初始化 SSE 日志处理器

        Args:
            output_queue: 输出队列，SSE 格式的字符串会放入此队列
            level: 日志级别
            default_show_in_thinking: 默认是否输出到思考过程（可通过日志 extra 字段覆盖）
        """
        super().__init__(level=level)
        self.output_queue = output_queue
        self.default_show_in_thinking = default_show_in_thinking
        self.filters = []

        # 添加过滤器，只捕获特定模式的日志
        def log_filter(record: logging.LogRecord) -> bool:
            # 检查日志名称是否匹配
            for pattern in self.CAPTURE_PATTERNS:
                if pattern in record.name.lower():
                    return True
            return False

        self.addFilter(log_filter)

    def emit(self, record: logging.LogRecord) -> None:
        """
        发送日志（转换为 SSE 格式）

        Args:
            record: 日志记录
        """
        try:
            if self.output_queue is None:
                return

            # 检查是否应该输出到思考过程
            # 优先使用日志记录中的 extra 字段，否则使用默认配置
            show_in_thinking = getattr(
                record,
                'show_in_thinking',
                self.default_show_in_thinking
            )

            # 如果不输出到思考过程，直接返回
            if not show_in_thinking:
                return

            # 格式化日志消息
            message = self.format(record)

            # 简化消息
            for old, new in self.MESSAGE_SIMPLIFICATIONS.items():
                if old in message:
                    message = message.replace(old, new)
                    break

            # 获取日志级别对应的图标和类型
            icon, log_type = self.LOG_LEVEL_CONFIG.get(
                record.levelno,
                ('📋', 'info')
            )

            # 从日志名称中提取简短的 logger 名称
            logger_name = record.name.split('.')[-1] if '.' in record.name else record.name

            # 构建 SSE 数据
            sse_data = {
                "type": "log",
                "log_type": log_type,
                "logger": logger_name,
                "message": f"{icon} {message}"
            }

            # 放入队列
            self.output_queue.put(f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n")

        except Exception:
            # 防止日志处理器本身出错
            self.handleError(record)


def create_sse_handler(
    output_queue: Optional['Queue[str]'] = None,
    default_show_in_thinking: bool = True
) -> SSELoggingHandler:
    """
    创建并配置 SSE 日志处理器

    Args:
        output_queue: 输出队列
        default_show_in_thinking: 默认是否输出到思考过程

    Returns:
        SSELoggingHandler: 配置好的日志处理器
    """
    handler = SSELoggingHandler(
        output_queue=output_queue,
        level=logging.INFO,
        default_show_in_thinking=default_show_in_thinking
    )

    # 设置格式化器
    formatter = logging.Formatter(
        fmt='%(message)s',  # 只保留消息，不包含时间、logger 名称等
        datefmt=None
    )
    handler.setFormatter(formatter)

    return handler


def log_with_config(
    logger: logging.Logger,
    level: int,
    msg: str,
    show_in_thinking: bool = True,
    *args,
    **kwargs
) -> None:
    """
    带配置的日志函数，可以控制是否输出到思考过程

    Args:
        logger: 日志记录器
        level: 日志级别（logging.INFO, logging.DEBUG 等）
        msg: 日志消息
        show_in_thinking: 是否输出到思考过程
        *args, **kwargs: 传递给 logger.log 的其他参数
    """
    # 创建 LogRecord 时添加 extra 字段
    extra = kwargs.get('extra', {})
    extra['show_in_thinking'] = show_in_thinking
    kwargs['extra'] = extra
    
    logger.log(level, msg, *args, **kwargs)


def log_debug(logger: logging.Logger, msg: str, show_in_thinking: bool = True, **kwargs) -> None:
    """DEBUG 级别日志"""
    log_with_config(logger, logging.DEBUG, msg, show_in_thinking, **kwargs)


def log_info(logger: logging.Logger, msg: str, show_in_thinking: bool = True, **kwargs) -> None:
    """INFO 级别日志"""
    log_with_config(logger, logging.INFO, msg, show_in_thinking, **kwargs)


def log_warning(logger: logging.Logger, msg: str, show_in_thinking: bool = True, **kwargs) -> None:
    """WARNING 级别日志"""
    log_with_config(logger, logging.WARNING, msg, show_in_thinking, **kwargs)


def log_error(logger: logging.Logger, msg: str, show_in_thinking: bool = True, **kwargs) -> None:
    """ERROR 级别日志"""
    log_with_config(logger, logging.ERROR, msg, show_in_thinking, **kwargs)


def log_exception(logger: logging.Logger, msg: str, show_in_thinking: bool = True, **kwargs) -> None:
    """EXCEPTION 级别日志"""
    log_with_config(logger, logging.ERROR, msg, show_in_thinking, **kwargs)


def create_sse_handler(output_queue: Optional['Queue[str]'] = None) -> SSELoggingHandler:
    """
    创建并配置 SSE 日志处理器

    Args:
        output_queue: 输出队列

    Returns:
        SSELoggingHandler: 配置好的日志处理器
    """
    handler = SSELoggingHandler(output_queue=output_queue, level=logging.INFO)

    # 设置格式化器
    formatter = logging.Formatter(
        fmt='%(message)s',  # 只保留消息，不包含时间、logger 名称等
        datefmt=None
    )
    handler.setFormatter(formatter)

    return handler
