import logging
import os
import sys
from logging.config import dictConfig
from logging.handlers import RotatingFileHandler

# 确保日志目录存在
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # 不要禁用 uvicorn 自带的 logger
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
        "error_file": {
            "level": "ERROR",
            "class": "logging.FileHandler",
            "filename": os.path.join(LOG_DIR, "error.log"),
            "encoding": "utf-8",
            "formatter": "default",
        },
    },
    "root": {  # 所有 logger 都会继承 root
        "level": "DEBUG",
        "handlers": ["console", "error_file"],
    },
    "loggers": {
        "uvicorn.error": {
            "handlers": ["console", "error_file"],
            "level": "ERROR",
            "propagate": True,
        },
        "uvicorn.access": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

dictConfig(LOGGING_CONFIG)


def setup_logger():
    """
    设置应用程序的根日志记录器。
    应在应用程序启动时调用一次。
    """
    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建文件处理器 - 正常日志 (INFO及以上)
    app_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, 'app.log'),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(formatter)

    # 创建文件处理器 - 错误日志 (WARNING及以上)
    error_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, 'error.log'),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(formatter)

    # 创建控制台处理器 - 开发环境使用
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 添加处理器到根日志记录器
    root_logger.addHandler(app_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    # 设置第三方库的日志级别
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('sqlalchemy').setLevel(logging.WARNING)

    # 确保 langchain 相关库的警告和错误都能被捕获
    logging.getLogger('langchain_google_genai').setLevel(logging.WARNING)
    logging.getLogger('langchain').setLevel(logging.WARNING)
    logging.getLogger('langchain_core').setLevel(logging.WARNING)
    logging.getLogger('langchain_community').setLevel(logging.WARNING)
    logging.getLogger('langchain_openai').setLevel(logging.WARNING)

    return root_logger
