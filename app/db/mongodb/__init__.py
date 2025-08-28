# MongoDB 数据库模块
# 提供 MongoDB 相关的数据访问对象

from .chat_history_db import chat_history_db
from .mongodb import mongodb_db

__all__ = ['chat_history_db', 'mongodb_db']
