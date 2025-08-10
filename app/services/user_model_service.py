# app/services/user_model_service.py
from sqlalchemy.orm import Session
from db.user_model_db import user_model_db
from schemas.user_model_schemas import UserModelCreate, UserModelUpdate


class UserModelService:
    def get_user_models(self, db: Session, user_id: int):
        return user_model_db.get_by_user_id(db, user_id=user_id)

    def create_user_model(self, db: Session, user_model: UserModelCreate, user_id: int):
        # 将 user_id 添加到要创建的数据中
        create_data = user_model.model_dump()
        create_data['user_id'] = user_id
        return user_model_db.create(db, obj_in=create_data)

    def update_user_model(self, db: Session, id: int, user_model: UserModelUpdate):
        return user_model_db.update(db, db_obj=user_model_db.get(db, id), obj_in=user_model)

    def remove_user_model(self, db: Session, id: int):
        return user_model_db.remove(db, id=id)


user_model_service = UserModelService()
