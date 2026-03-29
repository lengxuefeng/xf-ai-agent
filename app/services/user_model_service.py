# -*- coding: utf-8 -*-
from typing import Optional

from sqlalchemy.orm import Session

from db.crud import user_model_db
from schemas.user_model_schemas import UserModelCreate, UserModelUpdate
from utils.custom_logger import get_logger

log = get_logger(__name__)


class UserModelService:
    """用户大模型配置服务层"""

    def get_user_models(self, db: Session, user_id: int):
        """获取用户的所有模型配置"""
        return user_model_db.get_by_user_id(db, user_id=user_id)

    def get_active_user_model(self, db: Session, user_id: int):
        """获取用户当前激活的模型配置"""
        return user_model_db.get_active_by_user_id(db, user_id=user_id)

    def get_user_model_by_id(self, db: Session, model_id: int, user_id: Optional[int] = None):
        """
        【安全优化】获取用户模型配置，严格验证数据归属权
        """
        user_model = user_model_db.get(db, model_id)
        if not user_model:
            return None

        # 越权访问拦截：存在 user_id 参数，且该模型不属于此人
        if user_id is not None and user_model.user_id != user_id:
            log.warning(f"越权访问拦截：用户 {user_id} 试图访问模型配置 {model_id}")
            return None

        return user_model

    def create_user_model(self, db: Session, user_model: UserModelCreate, user_id: int, allow_multiple: bool = False):
        """创建用户模型配置"""
        create_data = user_model.model_dump()
        create_data['user_id'] = user_id

        # 唯一激活态控制
        if create_data.get('is_active') and not allow_multiple:
            user_model_db.deactivate_all_by_user_id(db, user_id=user_id)

        new_model = user_model_db.create(db, obj_in=create_data)
        log.info(f"用户 {user_id} 创建了新模型配置: ID={new_model.id}")
        return new_model

    def update_user_model(
        self,
        db: Session,
        id: int,
        user_model: UserModelUpdate,
        user_id: Optional[int] = None,
        allow_multiple: bool = False,
    ):
        """
        更新用户模型配置，并在需要时校验归属权。

        这里显式接收 user_id，是为了让 API 层不会把“当前登录用户”信息丢在半路，
        避免别人拿到配置 ID 后直接越权修改。
        """
        existing_model = self.get_user_model_by_id(db, id, user_id)
        if not existing_model:
            raise ValueError(f"要更新的模型配置不存在或无权操作: ID={id}")

        # exclude_unset=True 完美保留，只更新前端传了的字段
        update_data = user_model.model_dump(exclude_unset=True)

        # 唯一激活态控制
        if update_data.get('is_active', False) and not allow_multiple:
            user_model_db.deactivate_all_by_user_id(db, user_id=existing_model.user_id)

        updated_model = user_model_db.update(db, db_obj=existing_model, obj_in=update_data)
        log.info(f"模型配置 ID={id} 更新成功")
        return updated_model

    def activate_user_model(self, db: Session, id: int, user_id: int, allow_multiple: bool = False):
        """
        【安全优化】激活指定的用户模型配置（附加归属权校验）
        """
        existing_model = self.get_user_model_by_id(db, id, user_id)
        if not existing_model:
            raise ValueError(f"模型配置不存在或无权操作: ID={id}")

        if not allow_multiple:
            user_model_db.deactivate_all_by_user_id(db, user_id=user_id)

        activated_model = user_model_db.update(db, db_obj=existing_model, obj_in={'is_active': True})
        log.info(f"用户 {user_id} 成功激活了模型配置 ID={id}")
        return activated_model

    def deactivate_user_model(self, db: Session, id: int, user_id: int):
        """取消激活指定模型配置，避免前端点了停用却仍然保持激活态。"""
        existing_model = self.get_user_model_by_id(db, id, user_id)
        if not existing_model:
            raise ValueError(f"模型配置不存在或无权操作: ID={id}")
        deactivated_model = user_model_db.update(db, db_obj=existing_model, obj_in={"is_active": False})
        log.info(f"用户 {user_id} 取消激活模型配置 ID={id}")
        return deactivated_model

    def remove_user_model(self, db: Session, id: int, user_id: int):
        """删除用户模型配置前先校验归属权，避免直接按主键物理删除越权数据。"""
        existing_model = self.get_user_model_by_id(db, id, user_id)
        if not existing_model:
            raise ValueError(f"模型配置不存在或无权操作: ID={id}")
        removed_model = user_model_db.remove(db, id=id)
        log.info(f"用户 {user_id} 删除模型配置 ID={id}")
        return removed_model


# 导出全局单例
user_model_service = UserModelService()
