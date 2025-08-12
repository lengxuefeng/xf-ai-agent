# app/services/service_user.py (修改后)

from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from starlette.status import HTTP_401_UNAUTHORIZED

from db.mysql.user_info_db import user_info_db
from exceptions.business_exception import BusinessException
from models.user_info import UserInfo
from schemas.user_info_schemas import UserInfoBase, UserInfoCreate, UserInfoUpdate, UserInfoResp
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

    def create_user(self, db: Session, user_create: UserInfoCreate) -> UserInfoResp:
        # 1. 检查用户是否已存在
        db_user = self.query(db).filter(self.model.user_name == user_create.user_name).first()
        if db_user:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户名已存在")

        # 2. 加密密码
        hashed_password = encryption_utils.encrypt_password(user_create.password)

        # 3. 创建ORM对象，此时不含token
        new_user = self.model(
            password=hashed_password,
            user_name=user_create.user_name,
            nick_name=user_create.nick_name,
            phone=user_create.phone
        )

        # 4. 添加到会话并flush，以获取数据库生成的ID
        db.add(new_user)
        db.flush()

        # 5. 使用新ID生成Token
        token = encryption_utils.token_encode(user_id=new_user.id)

        # 6. 将Token赋给ORM对象，然后提交整个事务
        new_user.token = token
        db.commit()
        db.refresh(new_user)

        # 7. 返回Pydantic模型
        return UserInfoResp.model_validate(new_user)

    def login(self, db: Session, user: UserInfoBase) -> UserInfoResp:
        # 业务逻辑：根据用户名和密码查询用户
        db_user = user_info_db.query(db).filter(self.model.user_name == user.user_name).first()
        # 业务逻辑：如果用户不存在或密码错误，抛出异常
        if db_user is None:
            raise BusinessException(code=HTTP_401_UNAUTHORIZED, message="用户不存在")
        hashed_password = encryption_utils.verify_password(user.password, db_user.password)
        if not hashed_password:
            raise BusinessException(code=HTTP_401_UNAUTHORIZED, message="密码错误")
        # 业务逻辑：如果用户存在，返回用户信息
        return UserInfoResp.model_validate(db_user)

    def get_user_by_id(self, db: Session, user_id: int) -> UserInfoResp:
        # 业务逻辑：根据id查询用户
        db_user = user_info_db.get(db, id=user_id)
        if db_user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
        # 业务逻辑：如果用户存在，返回用户信息
        return UserInfoResp.model_validate(db_user)


user_info_service = UserInfoService()
