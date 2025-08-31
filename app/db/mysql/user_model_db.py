# app/db/user_model_db.py

from sqlalchemy.orm import Session

from models.user_model import UserModel
from .base_crud import CRUDBase


class UserModelDB(CRUDBase[UserModel, UserModel, UserModel]):
    def get_by_user_id(self, db: Session, *, user_id: int) -> list[type[UserModel]]:
        """获取用户的所有模型配置"""
        return db.query(self.model).filter(self.model.user_id == user_id).all()
    
    def get_active_by_user_id(self, db: Session, *, user_id: int) -> UserModel:
        """获取用户当前激活的模型配置"""
        return db.query(self.model).filter(
            self.model.user_id == user_id,
            self.model.is_active == True
        ).first()
    
    def deactivate_all_by_user_id(self, db: Session, *, user_id: int):
        """将用户的所有模型配置设为非激活状态"""
        db.query(self.model).filter(self.model.user_id == user_id).update(
            {"is_active": False}
        )
        db.commit()


# 创建实例
user_model_db = UserModelDB(UserModel)
