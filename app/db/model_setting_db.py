from models.model_setting import ModelSetting
from .base_crud import CRUDBase
from sqlalchemy.orm import Session


class ModelSettingDB(CRUDBase[ModelSetting, ModelSetting, ModelSetting]):
    def get_by_user_id(self, db: Session, *, user_id: int) -> list[type[ModelSetting]]:
        return db.query(self.model).filter(self.model.user_id == user_id).all()


model_setting_db = ModelSettingDB(ModelSetting)
