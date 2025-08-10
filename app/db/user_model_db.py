# # app/db/user_model_db.py
#
# from sqlalchemy.orm import Session
#
# from models.user_model import UserModel
# from .base_crud import CRUDBase
#
#
# class UserModelDB(CRUDBase[UserModel, UserModel, UserModel]):
#     def get_by_user_id(self, db: Session, *, user_id: int) -> list[type[UserModel]]:
#         return db.query(self.model).filter(self.model.user_id == user_id).all()
#
#
# # 创建实例
# user_model_db = UserModelDB(UserModel)
