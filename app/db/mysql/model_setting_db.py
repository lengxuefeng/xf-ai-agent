from models.model_setting import ModelSetting
from .base_crud import CRUDBase
from sqlalchemy.orm import Session


class ModelSettingDB(CRUDBase[ModelSetting, ModelSetting, ModelSetting]):
    def get_all_enabled(self, db: Session) -> list[type[ModelSetting]]:
        """获取所有启用的系统模型服务"""
        return db.query(self.model).filter(self.model.is_enabled == True).all()

    def get_by_service_type(self, db: Session, *, service_type: str) -> ModelSetting:
        """根据服务类型获取模型配置"""
        return db.query(self.model).filter(self.model.service_type == service_type).first()

    def get_system_default(self, db: Session) -> ModelSetting:
        """获取系统默认模型服务配置"""
        return db.query(self.model).filter(
            self.model.is_system_default == True,
            self.model.is_enabled == True
        ).first()


model_setting_db = ModelSettingDB(ModelSetting)
