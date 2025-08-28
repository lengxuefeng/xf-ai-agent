# crud_base.py
from typing import Type, TypeVar, Generic, Optional, List, Any, Union

from pydantic import BaseModel
from sqlalchemy.orm import Session, Query

from db.mysql import Base

# -------------------------------
# 泛型类型
# -------------------------------
ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)

# -------------------------------
# 通用 CRUD 封装
# -------------------------------
class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    通用 CRUD 封装（不管理事务，由 get_db 上下文管理器统一管理）
    """

    def __init__(self, model: Type[ModelType]):
        self.model = model

    def query(self, db: Session) -> Query:
        """返回 SQLAlchemy Query 对象，可链式调用"""
        return db.query(self.model)

    def get(self, db: Session, id: Any) -> Optional[ModelType]:
        return db.query(self.model).filter(self.model.id == id).first()

    def get_multi(self, db: Session, skip: int = 0, limit: int = 10) -> List[ModelType]:
        return db.query(self.model).offset(skip).limit(limit).all()

    def create(self, db: Session, obj_in: Union[CreateSchemaType, dict]) -> ModelType:
        """创建对象，不提交事务，由外部上下文管理"""
        if isinstance(obj_in, dict):
            obj_data = obj_in
        else:
            obj_data = obj_in.model_dump()
        db_obj = self.model(**obj_data)
        db.add(db_obj)
        db.flush()  # 刷新对象
        return db_obj

    def update(self, db: Session, db_obj: ModelType, obj_in: Union[UpdateSchemaType, dict]) -> ModelType:
        """更新对象，不提交事务，由外部上下文管理"""
        update_data = obj_in if isinstance(obj_in, dict) else obj_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_obj, field, value)
        db.add(db_obj)
        db.flush()
        return db_obj

    def remove(self, db: Session, id: Any) -> Optional[ModelType]:
        """删除对象，不提交事务，由外部上下文管理"""
        obj = db.query(self.model).get(id)
        if obj:
            db.delete(obj)
            db.flush()
        return obj

