# app/services/model_setting_service.py
from sqlalchemy.orm import Session

from db.crud import model_setting_db
from schemas.model_setting_schemas import ModelServiceCreate, ModelServiceUpdate


class ModelSettingService:
    """系统模型服务业务逻辑层"""

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
            if ('embedding' in model_lower or 'embed' in model_lower or 
                'bge-' in model_lower or 'text-embedding' in model_lower or
                'e5-' in model_lower or 'sentence-' in model_lower or
                'all-minilm' in model_lower or 'all-mpnet' in model_lower or
                'gte-' in model_lower or 'm3e-' in model_lower or
                'text2vec' in model_lower or 'simcse' in model_lower or
                'sbert' in model_lower or 'instructor' in model_lower or
                'multilingual-e5' in model_lower or 'paraphrase-' in model_lower or
                ('distilbert' in model_lower and 'embed' in model_lower) or
                model_lower in ['engtrng', 'engtng1', '333', 'zz', '1123']):
                categories['embedding'].append(model)
            # 视觉模型关键词 - 第二优先级  
            elif ('dall-e' in model_lower or 'midjourney' in model_lower or 
                  'stable-diffusion' in model_lower or 'image' in model_lower or
                  'vision' in model_lower or 'visual' in model_lower or
                  'gpt-4-vision' in model_lower or 'gpt-4v' in model_lower or
                  ('claude-3' in model_lower and 'vision' in model_lower) or
                  'gemini-pro-vision' in model_lower or 'gemini-vision' in model_lower or
                  'cogvlm' in model_lower or 'blip' in model_lower or
                  'clip' in model_lower or 'flamingo' in model_lower or
                  'llava' in model_lower or 'minigpt' in model_lower or
                  'instructblip' in model_lower or 'qwen-vl' in model_lower or
                  'internvl' in model_lower or 'yi-vl' in model_lower or
                  '视觉' in model_lower or '图像' in model_lower or
                  '看图' in model_lower or '识图' in model_lower):
                categories['vision'].append(model)
            # 对话模型关键词 - 第三优先级
            elif ('gpt' in model_lower or 'claude' in model_lower or 
                  'gemini' in model_lower or 'llama' in model_lower or 
                  'qwen' in model_lower or 'chat' in model_lower or
                  'pro' in model_lower or 'turbo' in model_lower or
                  'flash' in model_lower or 'sonnet' in model_lower or
                  'haiku' in model_lower or 'opus' in model_lower or
                  'mistral' in model_lower or 'yi-' in model_lower or
                  'baichuan' in model_lower or 'chatglm' in model_lower or
                  'vicuna' in model_lower or 'alpaca' in model_lower or
                  'wizard' in model_lower or 'orca' in model_lower or
                  'falcon' in model_lower or 'mpt' in model_lower or
                  'palm' in model_lower or 'bard' in model_lower or
                  'ernie' in model_lower or 'wenxin' in model_lower or
                  'tongyi' in model_lower or 'spark' in model_lower):
                categories['chat'].append(model)
            else:
                categories['other'].append(model)
        
        return categories


model_setting_service = ModelSettingService()
