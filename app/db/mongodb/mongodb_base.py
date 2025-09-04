from typing import TypeVar, Generic, Type, Any, Optional, List, Dict, Tuple

from pydantic import BaseModel
from pymongo.collection import Collection

from .mongodb import mongodb_db
from schemas.common import PageParams

# Pydantic模型类型变量
ModelType = TypeVar("ModelType", bound=BaseModel)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class MongoCRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    def __init__(self, *, collection_name: str, model: Type[ModelType]):
        """
        提供默认Create, Read, Update, Delete (CRUD) 方法的CRUD基类。

        :param collection_name: MongoDB集合（表）的名称。
        :param model: Pydantic模型类，用于数据校验和转换。
        """
        self.db = mongodb_db.get_db()
        self.collection: Collection = self.db[collection_name]
        self.model = model

    def _to_model(self, doc: Optional[Dict[str, Any]]) -> Optional[ModelType]:
        """将从MongoDB获取的文档转换为Pydantic模型实例。"""
        if doc:
            # Pydantic的 `populate_by_name=True` 配置会自动处理 `_id` 到 `id` 的映射。
            return self.model.model_validate(doc)
        return None

    def create(self, *, obj_in: CreateSchemaType) -> Optional[ModelType]:
        """
        创建一个新的文档。

        :param obj_in: Pydantic模型实例，包含要创建的数据。
        :return: 创建后的文档对应的Pydantic模型实例。
        """
        print(f"====={obj_in}")
        db_obj = obj_in.model_dump()
        result = self.collection.insert_one(db_obj)
        inserted_doc = self.collection.find_one({"_id": result.inserted_id})
        return self._to_model(inserted_doc)

    def get(self, *, query: Dict[str, Any]) -> Optional[ModelType]:
        """
        根据查询条件获取单个文档。

        :param query: MongoDB查询字典。
        :return: 匹配的文档对应的Pydantic模型实例，或None。
        """
        doc = self.collection.find_one(query)
        return self._to_model(doc)

    def get_multi(
            self, *, query: Dict[str, Any] = None, skip: int = 0, limit: int = 100, sort: Optional[Any] = None
    ) -> List[ModelType]:
        """
        获取多个文档。

        :param query: MongoDB查询字典。
        :param skip: 跳过的文档数。
        :param limit: 返回的最大文档数。
        :param sort: 排序条件。
        :return: Pydantic模型实例的列表。
        """
        cursor = self.collection.find(query or {}).skip(skip).limit(limit)
        if sort:
            cursor = cursor.sort(sort)

        docs = list(cursor)
        return [self._to_model(doc) for doc in docs if doc is not None]

    def update(self, *, query: Dict[str, Any], obj_in: UpdateSchemaType) -> Optional[ModelType]:
        """
        更新一个文档。

        :param query: MongoDB查询字典，用于定位要更新的文档。
        :param obj_in: Pydantic模型实例，包含要更新的数据。
        :return: 更新后的文档对应的Pydantic模型实例。
        """
        update_data = obj_in.model_dump(exclude_unset=True)
        if not update_data:
            return self.get(query=query)

        self.collection.update_one(query, {"$set": update_data})
        updated_doc = self.collection.find_one(query)
        return self._to_model(updated_doc)

    def delete(self, *, query: Dict[str, Any]) -> bool:
        """
        删除一个文档。

        :param query: MongoDB查询字典，用于定位要删除的文档。
        :return: 如果成功删除了一个文档，返回True，否则返回False。
        """
        result = self.collection.delete_one(query)
        return result.deleted_count > 0

    def count(self, *, query: Dict[str, Any] = None) -> int:
        """
        统计符合条件的文档数量。

        :param query: MongoDB查询字典。
        :return: 文档数量。
        """
        return self.collection.count_documents(query or {})

    def paginate(
            self,
            *,
            query: Dict[str, Any] = None,
            page: int = 1,
            size: int = 10,
            sort: Optional[Any] = None
    ) -> Tuple[List[ModelType], int, int, int]:
        """
        分页查询文档。

        :param query: MongoDB查询字典。
        :param page: 页码（从1开始）。
        :param size: 每页大小。
        :param sort: 排序条件。
        :return: 元组 (数据列表, 总记录数, 当前页, 总页数)。
        """
        # 计算跳过的记录数
        skip = (page - 1) * size

        # 执行查询
        cursor = self.collection.find(query or {}).skip(skip).limit(size)
        if sort:
            cursor = cursor.sort(sort)

        docs = list(cursor)
        items = [self._to_model(doc) for doc in docs if doc is not None]

        # 计算总记录数和总页数
        total = self.count(query=query)
        pages = (total + size - 1) // size

        return items, total, page, pages
