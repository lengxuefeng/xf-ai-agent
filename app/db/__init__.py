# app/db/__init__.py
# 统一导出所有数据库访问对象

from .mysql.user_info_db import user_info_db
from .mysql.user_model_db import user_model_db  
from .mysql.user_mcp_db import user_mcp_db
from .mysql.model_setting_db import model_setting_db

# 导出get_db函数
from .mysql import get_db

__all__ = [
    'user_info_db',
    'user_model_db', 
    'user_mcp_db',
    'model_setting_db',
    'get_db'
]