# -*- coding: utf-8 -*-
"""
数据库模型（Models）统一导出模块。

本模块统一导出所有的数据库模型类，方便其他模块导入使用。
使用统一的导入口可以避免循环导入问题，也方便管理依赖关系。

模型分类：
1. 用户相关模型：
   - UserInfo: 用户基础信息
   - UserModel: 用户模型配置
   - ModelSetting: 模型设置
   - UserMCP: 用户MCP配置

2. 对话相关模型：
   - ChatSessionModel: 聊天会话
   - ChatMessageModel: 聊天消息
   - SessionStateModel: 会话状态

3. 审批相关模型：
   - InterruptApprovalModel: 中断审批记录

设计要点：
1. 所有模型使用SQLAlchemy ORM定义
2. 支持PostgreSQL数据库
3. 使用自动时间戳和软删除
4. 模型之间通过外键关联

使用方式：
from models import UserInfo, UserModel, ChatMessageModel
"""
from models.chat_history import ChatMessageModel, ChatSessionModel
from models.interrupt_approval import InterruptApprovalModel
from models.model_setting import ModelSetting
from models.session_state import SessionStateModel
from models.user_info import UserInfo
from models.user_mcp import UserMCP
from models.user_model import UserModel

__all__ = [
    "UserInfo",
    "ModelSetting",
    "UserModel",
    "UserMCP",
    "ChatSessionModel",
    "ChatMessageModel",
    "SessionStateModel",
    "InterruptApprovalModel",
]
