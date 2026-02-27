# -*- coding: utf-8 -*-
# app/db/base_crud.py
from typing import Type, TypeVar, Generic, Optional, List, Any, Union, Dict

from pydantic import BaseModel
from sqlalchemy.orm import Session, Query

# 从我们刚刚重构的 PgSQL 引擎根目录导入 Base
from db import Base

# -------------------------------
# 泛型类型约束
# -------------------------------
ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


# -------------------------------
# 通用 CRUD 封装
# -------------------------------
class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    通用 CRUD 基类
    职责：提供单表的标准增删改查操作。
    """

    def __init__(self, model: Type[ModelType]):
        self.model = model

    def query(self, db: Session) -> Query:
        return db.query(self.model)

    def get(self, db: Session, id: Any) -> Optional[ModelType]:
        # 使用 SQLAlchemy 2.0 推荐的 db.get()，利用一级缓存提升性能
        return db.get(self.model, id)

    def get_multi(self, db: Session, skip: int = 0, limit: int = 10) -> List[ModelType]:
        return db.query(self.model).offset(skip).limit(limit).all()

    def create(self, db: Session, obj_in: Union[CreateSchemaType, Dict[str, Any]]) -> ModelType:
        obj_in_data = obj_in if isinstance(obj_in, dict) else obj_in.model_dump()
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        db.commit()  # 显式提交
        db.refresh(db_obj)
        return db_obj

    def update(self, db: Session, db_obj: ModelType, obj_in: Union[UpdateSchemaType, Dict[str, Any]]) -> ModelType:
        obj_data = {column.name: getattr(db_obj, column.name) for column in db_obj.__table__.columns}

        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.model_dump(exclude_unset=True)

        for field in obj_data:
            if field in update_data:
                setattr(db_obj, field, update_data[field])

        db.add(db_obj)
        db.flush()
        db.refresh(db_obj)
        return db_obj

    def remove(self, db: Session, *, id: int) -> Optional[ModelType]:
        obj = db.get(self.model, id)
        if obj:
            db.delete(obj)
            db.flush()
        return obj