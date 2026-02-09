# -*- coding: utf-8 -*-
"""
自定义日志配置系统
支持按日期分类、灵活控制输出目标（日志文件/思考过程）
"""
import os
import logging
import logging.handlers
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from queue import Queue
import json


class LogTarget(Enum):
    """日志输出目标"""
    LOG = "LOG"       # 只输出到日志文件
    AI = "AI"         # 只输出到思考过程
    ALL = "ALL"       # 同时输出到日志文件和思考过程


class CustomLogger:
    """自定义日志记录器"""

    # 日志根目录
    LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")

    # 日志文件名前缀
    LOG_PREFIXES = {
        logging.DEBUG: "debug",
        logging.INFO: "info",
        logging.WARNING: "warning",
        logging.ERROR: "error",
        logging.CRITICAL: "critical",
    }

    # 日志级别到图标和类型的映射
    LOG_LEVEL_CONFIG = {
        logging.DEBUG: ('🔍', 'info'),
        logging.INFO: ('ℹ️', 'info'),
        logging.WARNING: ('⚠️', 'warning'),
        logging.ERROR: ('❌', 'error'),
        logging.CRITICAL: ('🔴', 'error'),
    }

    # 单例实例
    _instances: Dict[str, 'CustomLogger'] = {}

    # 全局 SSE 回调函数列表
    _global_sse_callbacks: List[callable] = []

    @classmethod
    def add_global_sse_callback(cls, callback: callable):
        """添加全局 SSE 回调函数"""
        if callback not in cls._global_sse_callbacks:
            cls._global_sse_callbacks.append(callback)
            print(f"[CustomLogger] 已添加全局回调，当前回调数: {len(cls._global_sse_callbacks)}")

    @classmethod
    def remove_global_sse_callback(cls, callback: callable):
        """移除全局 SSE 回调函数"""
        if callback in cls._global_sse_callbacks:
            cls._global_sse_callbacks.remove(callback)
            print(f"[CustomLogger] 已移除全局回调，当前回调数: {len(cls._global_sse_callbacks)}")

    @classmethod
    def clear_global_sse_callbacks(cls):
        """清空所有全局 SSE 回调函数"""
        cls._global_sse_callbacks.clear()
        print(f"[CustomLogger] 已清空所有全局回调")

    def __init__(self, name: str, log_dir: Optional[str] = None):
        """
        初始化自定义日志记录器

        Args:
            name: 日志记录器名称
            log_dir: 自定义日志目录（可选）
        """
        self.name = name
        self.log_dir = log_dir or self.LOG_DIR
        self.logger = logging.getLogger(f"custom.{name}")
        self.logger.setLevel(logging.DEBUG)

        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)

        # SSE 输出队列（用于输出到思考过程）
        self.sse_queue: Optional[Queue] = None

        # SSE 输出回调函数（用于输出到思考过程）
        self.sse_callback = None

        # 配置日志处理器
        self._setup_handlers()

    def _setup_handlers(self):
        """配置日志处理器"""
        # 清除已存在的处理器
        self.logger.handlers.clear()

        # 控制台处理器（开发调试用）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # 文件处理器（按日期）
        for level, prefix in self.LOG_PREFIXES.items():
            # 今天的日志文件
            today = datetime.now().strftime('%Y-%m-%d')
            filename = f"{today}_{prefix}.log"
            filepath = os.path.join(self.log_dir, filename)

            # 创建按天滚动的文件处理器
            file_handler = logging.handlers.TimedRotatingFileHandler(
                filename=filepath,
                when='midnight',
                interval=1,
                backupCount=30,
                encoding='utf-8',
            )
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        # 同时创建最新的链接文件（info.log, error.log 等）
        self._create_latest_links()

    def _create_latest_links(self):
        """创建最新的日志文件链接"""
        today = datetime.now().strftime('%Y-%m-%d')

        for level, prefix in self.LOG_PREFIXES.items():
            # 原始文件
            source_file = os.path.join(self.log_dir, f"{today}_{prefix}.log")
            # 链接文件
            link_file = os.path.join(self.log_dir, f"{prefix}.log")

            # 如果源文件存在，创建符号链接或复制
            if os.path.exists(source_file):
                # 删除旧的链接
                if os.path.exists(link_file) or os.path.islink(link_file):
                    try:
                        os.unlink(link_file)
                    except:
                        pass

                # 创建符号链接（Unix）或复制（Windows）
                try:
                    os.symlink(source_file, link_file)
                except (OSError, AttributeError):
                    # Windows 或不支持符号链接，使用复制
                    import shutil
                    shutil.copy2(source_file, link_file)

    def set_sse_queue(self, queue: Queue):
        """设置 SSE 输出队列"""
        self.sse_queue = queue

    def set_sse_callback(self, callback):
        """设置 SSE 输出回调函数"""
        self.sse_callback = callback

    def _log(
        self,
        level: int,
        msg: str,
        target: LogTarget = LogTarget.ALL,
        *args,
        **kwargs
    ):
        """
        内部日志方法

        Args:
            level: 日志级别
            msg: 日志消息
            target: 输出目标（LOG/AI/ALL）
            *args, **kwargs: 其他参数
        """
        # 输出到日志文件
        if target in [LogTarget.LOG, LogTarget.ALL]:
            self.logger.log(level, msg, *args, **kwargs)

        # 输出到思考过程（AI）
        if target in [LogTarget.AI, LogTarget.ALL]:
            icon, log_type = self.LOG_LEVEL_CONFIG.get(level, ('📋', 'info'))
            logger_name = self.name.split('.')[-1] if '.' in self.name else self.name

            sse_data = {
                "type": "log",
                "log_type": log_type,
                "logger": logger_name,
                "message": f"{icon} {msg}"
            }

            sse_message = f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"

            # 使用全局回调函数输出（优先）
            for callback in self._global_sse_callbacks:
                try:
                    callback(sse_message)
                except Exception as e:
                    print(f"[Logger] 全局回调输出失败: {e}")

            # 使用实例回调函数输出
            if self.sse_callback is not None:
                try:
                    self.sse_callback(sse_message)
                except Exception as e:
                    print(f"[Logger] 实例回调输出失败: {e}")

            # 使用队列输出
            if self.sse_queue is not None:
                try:
                    self.sse_queue.put(sse_message)
                except Exception as e:
                    print(f"[Logger] 队列输出失败: {e}")

    def debug(self, msg: str, target: LogTarget = LogTarget.ALL, **kwargs):
        """DEBUG 级别日志"""
        self._log(logging.DEBUG, msg, target, **kwargs)

    def info(self, msg: str, target: LogTarget = LogTarget.ALL, **kwargs):
        """INFO 级别日志"""
        self._log(logging.INFO, msg, target, **kwargs)

    def warning(self, msg: str, target: LogTarget = LogTarget.ALL, **kwargs):
        """WARNING 级别日志"""
        self._log(logging.WARNING, msg, target, **kwargs)

    def error(self, msg: str, target: LogTarget = LogTarget.ALL, **kwargs):
        """ERROR 级别日志"""
        self._log(logging.ERROR, msg, target, **kwargs)

    def critical(self, msg: str, target: LogTarget = LogTarget.ALL, **kwargs):
        """CRITICAL 级别日志"""
        self._log(logging.CRITICAL, msg, target, **kwargs)

    def exception(self, msg: str, target: LogTarget = LogTarget.ALL, **kwargs):
        """EXCEPTION 级别日志（包含堆栈）"""
        self.logger.exception(msg)
        if target in [LogTarget.AI, LogTarget.ALL]:
            icon, log_type = self.LOG_LEVEL_CONFIG.get(logging.ERROR, ('❌', 'error'))
            logger_name = self.name.split('.')[-1] if '.' in self.name else self.name

            sse_data = {
                "type": "log",
                "log_type": log_type,
                "logger": logger_name,
                "message": f"{icon} {msg}"
            }

            sse_message = f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"

            # 使用全局回调函数输出
            for callback in self._global_sse_callbacks:
                try:
                    callback(sse_message)
                except Exception as e:
                    print(f"[Logger] 全局回调输出失败: {e}")

            # 使用队列输出
            if self.sse_queue is not None:
                self.sse_queue.put(sse_message)

    @classmethod
    def get_logger(cls, name: str, log_dir: Optional[str] = None) -> 'CustomLogger':
        """
        获取日志记录器实例（单例模式）

        Args:
            name: 日志记录器名称
            log_dir: 自定义日志目录（可选）

        Returns:
            CustomLogger: 日志记录器实例

        示例:
            log = get_logger("supervisor")
            log.info("模型加载成功", target=LogTarget.ALL)
            log.debug("调试信息", target=LogTarget.LOG)
        """
        if name not in cls._instances:
            cls._instances[name] = CustomLogger(name, log_dir)
        return cls._instances[name]


# 创建快捷函数
def get_logger(name: str, log_dir: Optional[str] = None) -> CustomLogger:
    """
    获取日志记录器

    Args:
        name: 日志记录器名称
        log_dir: 自定义日志目录（可选）

    Returns:
        CustomLogger: 日志记录器实例

    示例:
        log = get_logger("supervisor")
        log.info("模型加载成功", target=LogTarget.ALL)
        log.debug("调试信息", target=LogTarget.LOG)
    """
    return CustomLogger.get_logger(name, log_dir)
