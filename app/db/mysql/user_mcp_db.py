# app/db/mysql/user_mcp_db.py

from sqlalchemy.orm import Session

from models.user_mcp import UserMCP
from .base_crud import CRUDBase


class UserMCPDB(CRUDBase[UserMCP, UserMCP, UserMCP]):
    def get_by_user_id(self, db: Session, *, user_id: int) -> list[type[UserMCP]]:
        return db.query(self.model).filter(self.model.user_id == user_id).all()


# 创建实例
user_mcp_db = UserMCPDB(UserMCP)