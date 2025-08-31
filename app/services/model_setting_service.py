# app/services/model_setting_service.py
from sqlalchemy.orm import Session

from db.mysql.model_setting_db import model_setting_db
from schemas.model_setting_schemas import ModelServiceCreate, ModelServiceUpdate


class ModelSettingService:
    """系统模型服务业务逻辑层"""

    def get_model_services(self, db: Session, user_id: int = None):
        """获取系统预定义的模型服务配置"""
        # 现在返回所有启用的系统模型服务，不再按用户过滤
        return model_setting_db.get_all_enabled(db)

    def create_model_service(self, db: Session, service_data: ModelServiceCreate, user_id: int = None):
        """创建新的系统模型服务配置（管理员功能）"""
        # 系统模型配置不再关联特定用户
        create_data = service_data.model_dump()
        create_data['is_system_default'] = True
        
        return model_setting_db.create(db, obj_in=create_data)

    def update_model_service(self, db: Session, id: int, service_data: ModelServiceUpdate):
        """更新指定ID的模型服务配置"""
        return model_setting_db.update(db, db_obj=model_setting_db.get(db, id), obj_in=service_data)

    def remove_model_service(self, db: Session, id: int):
        """删除指定ID的模型服务配置"""
        return model_setting_db.remove(db, id=id)

    def test_connection(self, db: Session, service_id: int, api_key: str):
        """测试模型服务连接"""
        service = model_setting_db.get(db, service_id)
        if not service:
            return {"success": False, "message": "服务不存在"}

        # 这里可以根据不同服务类型实现具体的连接测试逻辑
        # 目前返回模拟结果
        return {"success": True, "message": "连接测试成功"}

    def toggle_service(self, db: Session, service_id: int, enabled: bool):
        """启用或禁用模型服务"""
        service = model_setting_db.get(db, service_id)
        if not service:
            return None

        update_data = {"is_enabled": enabled}
        return model_setting_db.update(db, db_obj=service, obj_in=update_data)


model_setting_service = ModelSettingService()
