# app/services/model_setting_service.py
from sqlalchemy.orm import Session

from config.constants.model_category_keywords import (
    CLAUDE_VISION_MARKERS,
    DISTILBERT_EMBED_MARKERS,
    EMBEDDING_EXACT_NAMES,
    MODEL_CATEGORY_KEYWORDS,
    ModelCategory,
)
from db.crud import model_setting_db
from models.schemas.model_setting_schemas import ModelServiceCreate, ModelServiceUpdate


class ModelSettingService:
    """系统模型服务业务逻辑层"""

    @staticmethod
    def _contains_any(text: str, keywords) -> bool:
        """判断文本是否命中任意关键词。"""
        return any(keyword in text for keyword in keywords)

    def get_model_services(self, db: Session, user_id: int = None):
        """获取系统预定义的模型服务配置"""
        # 现在返回所有启用的系统模型服务，不再按用户过滤
        services = model_setting_db.get_all_enabled(db)
        # 确保返回的模型数据格式正确
        for service in services:
            if isinstance(service.models, list):
                # 如果是旧格式的列表，转换为新的分类格式
                service.models = self._categorize_models(service.models)
            elif isinstance(service.models, dict):
                # 如果已经是字典格式，直接使用
                pass
            else:
                # 如果是其他格式或None，设置为默认分类格式
                service.models = {
                    "chat": [],
                    "embedding": [],
                    "vision": [],
                    "other": []
                }
        return services

    def create_model_service(self, db: Session, service_data: ModelServiceCreate, user_id: int = None):
        """创建新的系统模型服务配置（管理员功能）"""
        # 系统模型配置不再关联特定用户
        create_data = service_data.model_dump()
        create_data['is_system_default'] = True
        
        return model_setting_db.create(db, obj_in=create_data)

    def update_model_service(self, db: Session, id: int, service_data: ModelServiceUpdate):
        """更新指定ID的模型服务配置"""
        existing_service = model_setting_db.get(db, id)
        if not existing_service:
            return None
        
        # 如果更新的模型数据是列表格式，转换为分类格式
        if service_data.models and isinstance(service_data.models, list):
            service_data.models = self._categorize_models(service_data.models)
        
        return model_setting_db.update(db, db_obj=existing_service, obj_in=service_data)

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
    
    def _categorize_models(self, models):
        """将模型列表按类型分类"""
        if isinstance(models, dict):
            return models  # 已经是分类格式
            
        categories = {
            "chat": [],
            "embedding": [],
            "vision": [],
            "other": []
        }
        
        if not isinstance(models, list):
            return categories
        
        for model in models:
            model_lower = model.lower().strip()
            
            # 嵌入模型关键词 - 优先级最高
            if (
                self._contains_any(model_lower, MODEL_CATEGORY_KEYWORDS[ModelCategory.EMBEDDING])
                or all(k in model_lower for k in DISTILBERT_EMBED_MARKERS)
                or model_lower in EMBEDDING_EXACT_NAMES
            ):
                categories['embedding'].append(model)
            # 视觉模型关键词 - 第二优先级  
            elif (
                self._contains_any(model_lower, MODEL_CATEGORY_KEYWORDS[ModelCategory.VISION])
                or all(k in model_lower for k in CLAUDE_VISION_MARKERS)
            ):
                categories['vision'].append(model)
            # 对话模型关键词 - 第三优先级
            elif self._contains_any(model_lower, MODEL_CATEGORY_KEYWORDS[ModelCategory.CHAT]):
                categories['chat'].append(model)
            else:
                categories['other'].append(model)
        
        return categories


model_setting_service = ModelSettingService()
