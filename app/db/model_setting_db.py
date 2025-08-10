from sqlalchemy.orm import Session

from schemas.model_setting_schemas import ModelSettingBase

"""
模型配置数据库操作
"""


class ModelSettingDB:
    @staticmethod
    def create(db: Session, model_create: ModelSettingBase) -> ModelSettingBase:
        """创建模型配置"""
        db.add(model_create)
        db.commit()
        db.refresh(model_create)
        return model_create

    @staticmethod
    def get_by_id(db: Session, setting_id: int) -> ModelSettingBase:
        """通过ID查询模型配置"""
        return db.query(ModelSettingBase).filter(ModelSettingBase.id == setting_id).first()

    @staticmethod
    def get_by_user_id(db: Session, user_id: int) -> list[ModelSettingBase]:
        """查询指定用户的所有模型配置"""
        return db.query(ModelSettingBase).filter(ModelSettingBase.user_id == user_id).all()

    @staticmethod
    def update(db: Session, setting_id: int, update_data: dict) -> ModelSettingBase:
        """更新模型配置"""
        db.query(ModelSettingBase).filter(ModelSettingBase.id == setting_id).update(update_data)
        db.commit()
        return db.query(ModelSettingBase).filter(ModelSettingBase.id == setting_id).first()

    @staticmethod
    def delete(db: Session, setting_id: int) -> bool:
        """删除模型配置"""
        rows = db.query(ModelSettingBase).filter(ModelSettingBase.id == setting_id).delete()
        db.commit()
        return rows > 0  # 返回是否删除成功
