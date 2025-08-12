# app/db/user_info_db.py

from sqlalchemy.orm import Session

from models.user_info import UserInfo
from schemas.user_info_schemas import UserInfoCreate, UserInfoUpdate
from .base_crud import CRUDBase


class UserInfoDB(CRUDBase[UserInfo, UserInfoCreate, UserInfoUpdate]):
    """
    用户信息数据库操作类
    """

    def get_by_token(self, db: Session, *, token: str) -> UserInfo | None:
        return db.query(self.model).filter(self.model.token == token).first()

    def get_by_phone(self, db: Session, *, phone: str) -> UserInfo | None:
        return db.query(self.model).filter(self.model.phone == phone).first()


# 创建一个实例，方便在 service 层中导入和使用
user_info_db = UserInfoDB(UserInfo)
