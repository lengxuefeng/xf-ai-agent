# app/services/user_mcp_service.py
from sqlalchemy.orm import Session
from db.crud import user_mcp_db
from models.schemas.user_mcp_schemas import UserMCPCreate, UserMCPUpdate


class UserMCPService:
    def get_user_mcps(self, db: Session, user_id: int):
        return user_mcp_db.get_by_user_id(db, user_id=user_id)

    def create_user_mcp(self, db: Session, user_mcp: UserMCPCreate, user_id: int):
        create_data = user_mcp.model_dump()
        create_data['user_id'] = user_id
        return user_mcp_db.create(db, obj_in=create_data)

    def update_user_mcp(self, db: Session, id: int, user_mcp: UserMCPUpdate):
        return user_mcp_db.update(db, db_obj=user_mcp_db.get(db, id), obj_in=user_mcp)

    def remove_user_mcp(self, db: Session, id: int):
        return user_mcp_db.remove(db, id=id)


user_mcp_service = UserMCPService()
