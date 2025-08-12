# app/services/model_setting_service.py
from sqlalchemy.orm import Session

from db.mysql.model_setting_db import model_setting_db
from schemas.model_setting_schemas import ModelSettingCreate, ModelSettingUpdate


class ModelSettingService:
    def get_model_settings(self, db: Session, user_id: int):
        return model_setting_db.get_by_user_id(db, user_id=user_id)

    def create_model_setting(self, db: Session, model_setting: ModelSettingCreate, user_id: int):
        create_data = model_setting.model_dump()
        create_data['user_id'] = user_id
        return model_setting_db.create(db, obj_in=create_data)

    def update_model_setting(self, db: Session, id: int, model_setting: ModelSettingUpdate):
        return model_setting_db.update(db, db_obj=model_setting_db.get(db, id), obj_in=model_setting)

    def remove_model_setting(self, db: Session, id: int):
        return model_setting_db.remove(db, id=id)


model_setting_service = ModelSettingService()
