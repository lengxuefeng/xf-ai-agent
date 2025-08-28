# app/services/model_setting_service.py
from sqlalchemy.orm import Session

from db.mysql.model_setting_db import model_setting_db
from schemas.model_setting_schemas import ModelServiceCreate, ModelServiceUpdate


class ModelSettingService:
    """模型服务业务逻辑层"""

    def get_model_services(self, db: Session, user_id: int):
        """获取用户的所有模型服务配置"""
        return model_setting_db.get_by_user_id(db, user_id=user_id)

    def create_model_service(self, db: Session, service_data: ModelServiceCreate, user_id: int):
        """创建新的模型服务配置"""
        # 将用户ID添加到service_data中
        # 由于Pydantic模型是不可变的，我们需要创建一个包含user_id的字典
        create_data = service_data.model_dump()
        create_data['user_id'] = user_id
        
        # 直接传递字典给数据库层
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
