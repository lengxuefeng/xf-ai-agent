# app/services/service_user.py (修改后)
import uuid

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from db.user_info_db import user_info_db
from models.user_info import UserInfo
from schemas.user_info_schemas import UserInfoBase, UserInfoRead, UserInfoCreate, UserInfoUpdate, UserInfoResp
from services.base_service import BaseService
from utils.pwd_utils import encryption_utils


class UserInfoService(BaseService[UserInfo, UserInfoCreate, UserInfoUpdate]):
    def __init__(self):
        super().__init__(UserInfo)

    def list_gmail_users(self, db: Session, page: int = 1, size: int = 5):
        return self.page(
            db=db,
            page=page,
            size=size,
            order_by=[UserInfo.id.desc()]
        )

    def create_user(self, db: Session, user_create: UserInfoCreate) -> UserInfoRead:
        # 业务逻辑：给用户生成token
        user_create.token = encryption_utils.verify_token(user_create.token)
        user_create.password = encryption_utils.encrypt_password(user_create.password)
        # 调用基类create方法保存数据
        user = self.create(db, user_create)
        # 返回验证后的 Pydantic 读模型
        return UserInfoRead.model_validate(user)

    def login(self, db: Session, user: UserInfoBase) -> UserInfoRead:
        # 业务逻辑：根据用户名和密码查询用户

        db_user = user_info_db.get(db, user_name=user.user_name, password=user.password)
        # 业务逻辑：如果用户不存在或密码错误，抛出异常
        if db_user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在或密码错误")
        # 业务逻辑：如果用户存在，返回用户信息
        return UserInfoRead.model_validate(db_user)

    def get_user_by_token(self, db: Session, token: str) -> UserInfoResp:
        # 业务逻辑：根据token查询用户
        db_user = user_info_db.get_by_token(db, token=token)
        if db_user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="token不存在")
        # 业务逻辑：如果token存在，返回用户信息
        return UserInfoResp.model_validate(db_user)


user_info_service = UserInfoService()
