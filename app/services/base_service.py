# app/services/base_service.py
from typing import Any, Generic, List, Optional, Type, TypeVar

from pydantic import BaseModel
from sqlalchemy.orm import Session, Query

from db import Base

# 泛型类型
ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class BaseService(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    通用 Service 基类，类似 MyBatis-Plus 的 ServiceImpl
    """

    def __init__(self, model: Type[ModelType]):
        self.model = model

    # ======== MyBatis-Plus 风格的链式查询入口 ========
    def query(self, db: Session) -> Query:
        """
        返回 SQLAlchemy Query 对象，可链式调用
        用法: service.query(db).filter(...).order_by(...).all()
        """
        return db.query(self.model)

    # ======== 基础 CRUD ========

    def get(self, db: Session, id: Any) -> Optional[ModelType]:
        return db.query(self.model).filter(self.model.id == id).first()

    def get_multi(self, db: Session, skip: int = 0, limit: int = 100) -> List[ModelType]:
        return db.query(self.model).offset(skip).limit(limit).all()

    def create(self, db: Session, obj_in: CreateSchemaType) -> ModelType:
        obj_in_data = obj_in.model_dump()
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(self, db: Session, db_obj: ModelType, obj_in: UpdateSchemaType | dict) -> ModelType:
        update_data = obj_in if isinstance(obj_in, dict) else obj_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_obj, field, value)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def remove(self, db: Session, id: int) -> Optional[ModelType]:
        obj = db.query(self.model).get(id)
        if obj:
            db.delete(obj)
            db.commit()
        return obj

    # ======== 分页查询（MyBatis-Plus style） ========
    def page(
            self,
            db: Session,
            page: int = 1,
            size: int = 10,
            filters: Optional[list] = None,
            order_by: Optional[list] = None
    ) -> dict:
        """
        分页查询
        filters: 过滤条件列表，例如 [User.email.like('%@gmail.com%')]
        order_by: 排序条件列表，例如 [User.id.desc()]
        """
        query = db.query(self.model)

        if filters:
            for f in filters:
                query = query.filter(f)

        total = query.count()

        if order_by:
            query = query.order_by(*order_by)

        items = query.offset((page - 1) * size).limit(size).all()

        return {
            "total": total,
            "page": page,
            "size": size,
            "items": items
        }
