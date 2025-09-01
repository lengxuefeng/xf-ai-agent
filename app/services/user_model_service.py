# app/services/user_model_service.py
from sqlalchemy.orm import Session
from db.mysql.user_model_db import user_model_db
from schemas.user_model_schemas import UserModelCreate, UserModelUpdate


class UserModelService:
    def get_user_models(self, db: Session, user_id: int):
        """获取用户的所有模型配置"""
        return user_model_db.get_by_user_id(db, user_id=user_id)

    def get_active_user_model(self, db: Session, user_id: int):
        """获取用户当前激活的模型配置"""
        return user_model_db.get_active_by_user_id(db, user_id=user_id)

    def create_user_model(self, db: Session, user_model: UserModelCreate, user_id: int, allow_multiple: bool = False):
        """创建用户模型配置"""
        create_data = user_model.model_dump()
        create_data['user_id'] = user_id

        # 如果设置为激活状态，需要先将其他配置设为非激活（除非允许多个激活）
        if create_data.get('is_active', False) and not allow_multiple:
            user_model_db.deactivate_all_by_user_id(db, user_id=user_id)

        return user_model_db.create(db, obj_in=create_data)

    def update_user_model(self, db: Session, id: int, user_model: UserModelUpdate, allow_multiple: bool = False):
        """更新用户模型配置"""
        update_data = user_model.model_dump(exclude_unset=True)

        # 如果要激活当前配置，需要先将其他配置设为非激活（除非允许多个激活）
        if update_data.get('is_active', False) and not allow_multiple:
            existing_model = user_model_db.get(db, id)
            if existing_model:
                user_model_db.deactivate_all_by_user_id(db, user_id=existing_model.user_id)

        return user_model_db.update(db, db_obj=user_model_db.get(db, id), obj_in=update_data)

    def activate_user_model(self, db: Session, id: int, user_id: int, allow_multiple: bool = False):
        """激活指定的用户模型配置"""
        if not allow_multiple:
            # 如果不允许多个激活，先将用户的所有配置设为非激活
            user_model_db.deactivate_all_by_user_id(db, user_id=user_id)

        # 激活指定配置
        return user_model_db.update(db, db_obj=user_model_db.get(db, id), obj_in={"is_active": True})

    def deactivate_user_model(self, db: Session, id: int, user_id: int):
        """取消激活指定的用户模型配置"""
        return user_model_db.update(db, db_obj=user_model_db.get(db, id), obj_in={"is_active": False})

    def remove_user_model(self, db: Session, id: int):
        """删除用户模型配置"""
        return user_model_db.remove(db, id=id)


user_model_service = UserModelService()
