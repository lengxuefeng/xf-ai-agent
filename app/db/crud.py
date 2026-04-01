# -*- coding: utf-8 -*-
# app/db/crud.py

from db.base_crud import CRUDBase
from sqlalchemy.orm import Session

# --- 导入 Models ---
from models.user_info import UserInfo
from models.user_model import UserModel
from models.model_setting import ModelSetting
from models.user_mcp import UserMCP
from models.user_skill import UserSkill
from models.chat_history import ChatSessionModel, ChatMessageModel
from models.session_state import SessionStateModel

# --- 导入 Schemas ---
from models.schemas.user_info_schemas import UserInfoCreate, UserInfoUpdate
from models.schemas.user_model_schemas import UserModelCreate, UserModelUpdate
from models.schemas.model_setting_schemas import ModelServiceCreate, ModelServiceUpdate
from models.schemas.user_mcp_schemas import UserMCPCreate, UserMCPUpdate
from models.schemas.user_skill_schemas import UserSkillCreate, UserSkillUpdate
from models.schemas.chat_history_schemas import ChatSessionCreate, ChatSessionUpdate, ChatMessageCreate, ChatMessageUpdate
from models.schemas.session_state_schemas import SessionStateCreate, SessionStateUpdate

"""
【大一统 CRUD 注册中心】
这里不仅实例化所有的增删改查对象，还包含了各个表专属的自定义查询逻辑。
"""


class CRUDUserInfo(CRUDBase[UserInfo, UserInfoCreate, UserInfoUpdate]):
    def get_by_username(self, db: Session, user_name: str):
        """根据用户名查询用户（登录用）"""
        return db.query(self.model).filter(self.model.user_name == user_name).first()


class CRUDUserModel(CRUDBase[UserModel, UserModelCreate, UserModelUpdate]):
    def get_by_user_id(self, db: Session, user_id: int):
        """获取用户专属的所有模型配置"""
        return db.query(self.model).filter(self.model.user_id == user_id).all()

    def get_active_by_user_id(self, db: Session, user_id: int):
        """获取用户当前激活的模型"""
        return db.query(self.model).filter(
            self.model.user_id == user_id,
            self.model.is_active == True
        ).first()

    def deactivate_all_by_user_id(self, db: Session, user_id: int):
        """批量取消用户名下全部激活模型，保证激活态切换时状态一致。"""
        return (
            db.query(self.model)
            .filter(self.model.user_id == user_id, self.model.is_active == True)
            .update({"is_active": False}, synchronize_session=False)
        )


class CRUDModelSetting(CRUDBase[ModelSetting, ModelServiceCreate, ModelServiceUpdate]):
    def get_all_enabled(self, db: Session):
        """获取所有已启用的系统模型配置"""
        return db.query(self.model).filter(self.model.is_enabled == True).all()


class CRUDUserMCP(CRUDBase[UserMCP, UserMCPCreate, UserMCPUpdate]):
    def get_by_user_id(self, db: Session, user_id: int):
        """获取用户的 MCP 配置"""
        return db.query(self.model).filter(self.model.user_id == user_id).all()

    def get_active_by_user_id(self, db: Session, user_id: int):
        """获取用户当前启用的 MCP 配置。"""
        return db.query(self.model).filter(
            self.model.user_id == user_id,
            self.model.is_active == True,
        ).all()


class CRUDUserSkill(CRUDBase[UserSkill, UserSkillCreate, UserSkillUpdate]):
    def get_by_user_id(self, db: Session, user_id: int):
        """获取用户的 Skill 配置。"""
        return db.query(self.model).filter(self.model.user_id == user_id).all()

    def get_active_by_user_id(self, db: Session, user_id: int):
        """获取用户当前启用的 Skill 配置。"""
        return db.query(self.model).filter(
            self.model.user_id == user_id,
            self.model.is_active == True,
        ).all()


class CRUDChatSession(CRUDBase[ChatSessionModel, ChatSessionCreate, ChatSessionUpdate]):
    def get_by_user_id(self, db: Session, user_id: int, skip: int = 0, limit: int = 20):
        """分页获取用户的未删除会话列表，按时间倒序"""
        return db.query(self.model).filter(
            self.model.user_id == user_id,
            self.model.is_deleted == False
        ).order_by(self.model.create_time.desc()).offset(skip).limit(limit).all()

    def count_by_user_id(self, db: Session, user_id: int) -> int:
        """统计用户当前未删除的会话总数，供分页接口返回总量。"""
        return db.query(self.model).filter(
            self.model.user_id == user_id,
            self.model.is_deleted == False
        ).count()

    def get_by_session_id(self, db: Session, session_id: str):
        """根据会话 ID 查找"""
        return db.query(self.model).filter(self.model.session_id == session_id).first()


class CRUDChatMessage(CRUDBase[ChatMessageModel, ChatMessageCreate, ChatMessageUpdate]):
    def get_by_session_id(
            self,
            db: Session,
            session_id: str,
            skip: int = 0,
            limit: int = 50,
            order_desc: bool = False
    ):
        """分页获取某个会话的未删除聊天记录，支持按时间正序/倒序。"""
        order_clause = self.model.create_time.desc() if order_desc else self.model.create_time.asc()
        return db.query(self.model).filter(
            self.model.session_id == session_id,
            self.model.is_deleted == False
        ).order_by(order_clause).offset(skip).limit(limit).all()

    def count_by_session_id(self, db: Session, session_id: str) -> int:
        """统计某个会话下未删除消息总数。"""
        return db.query(self.model).filter(
            self.model.session_id == session_id,
            self.model.is_deleted == False
        ).count()


class CRUDSessionState(CRUDBase[SessionStateModel, SessionStateCreate, SessionStateUpdate]):
    """会话状态 CRUD，负责按 session_id 读取与更新结构化上下文。"""

    def get_by_session_id(self, db: Session, session_id: str):
        """根据会话 ID 获取未删除的会话状态记录。"""
        return (
            db.query(self.model)
            .filter(self.model.session_id == session_id, self.model.is_deleted == False)
            .first()
        )


# ==========================================
# 统一导出所有表的单例操作对象
# ==========================================
user_info_db = CRUDUserInfo(UserInfo)
user_model_db = CRUDUserModel(UserModel)
model_setting_db = CRUDModelSetting(ModelSetting)
user_mcp_db = CRUDUserMCP(UserMCP)
user_skill_db = CRUDUserSkill(UserSkill)

chat_session_db = CRUDChatSession(ChatSessionModel)
chat_message_db = CRUDChatMessage(ChatMessageModel)
session_state_db = CRUDSessionState(SessionStateModel)
