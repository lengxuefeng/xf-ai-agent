from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from starlette.status import HTTP_401_UNAUTHORIZED

from db.crud import user_info_db
from exceptions.business_exception import BusinessException
from models.user_info import UserInfo
from schemas.user_info_schemas import (
    UserInfoBase, UserInfoCreate, UserInfoUpdate, UserInfoResp, 
    UserInfoChangePassword, TokenResponse
)
from services.base_service import BaseService
from utils.pwd_utils import encryption_utils, PasswordStrengthError
from utils.config import settings


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

        # 2. 加密密码（包含密码强度验证）
        try:
            hashed_password = encryption_utils.encrypt_password(user_create.password)
        except PasswordStrengthError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

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
        token = encryption_utils.create_access_token(user_id=new_user.id)

        # 6. 将Token赋给ORM对象，然后提交整个事务
        new_user.token = token
        db.commit()
        db.refresh(new_user)

        # 7. 返回Pydantic模型
        return UserInfoResp.model_validate(new_user)

    def login(self, db: Session, user: UserInfoBase) -> TokenResponse:
        """
        用户登录，返回访问令牌和刷新令牌
        """
        # 业务逻辑：根据用户名和密码查询用户
        db_user = user_info_db.query(db).filter(self.model.user_name == user.user_name).first()
        # 业务逻辑：如果用户不存在或密码错误，抛出异常
        if db_user is None:
            raise BusinessException(code=HTTP_401_UNAUTHORIZED, message="用户不存在")
        
        if not encryption_utils.verify_password(user.password, db_user.password):
            raise BusinessException(code=HTTP_401_UNAUTHORIZED, message="密码错误")
        
        # 生成访问令牌和刷新令牌
        access_token = encryption_utils.create_access_token(db_user.id)
        refresh_token = encryption_utils.create_refresh_token(db_user.id)
        
        # 更新数据库中的token
        db_user.token = access_token
        db.commit()
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )

    def get_user_by_id(self, db: Session, user_id: int) -> UserInfoResp:
        # 业务逻辑：根据id查询用户
        db_user = self._check_info(db, user_id)
        # 业务逻辑：如果用户存在，返回用户信息
        return UserInfoResp.model_validate(db_user)

    def update_user(self, db: Session, user_id: int, user_update: UserInfoUpdate) -> bool:
        db_update_user = self.update(db, self._check_info(db, user_id), user_update)
        return True if db_update_user else False

    def _check_info(self, db: Session, user_id: int) -> UserInfo:
        # 业务逻辑：根据id查询用户
        db_user = user_info_db.get(db, id=user_id)
        if db_user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
        return db_user

    def change_password(self, db: Session, user_id: int, user_update: UserInfoChangePassword) -> bool:
        """
        修改用户密码
        """
        db_user = self._check_info(db, user_id)
        
        # 验证旧密码
        if not encryption_utils.verify_password(user_update.old_password, db_user.password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="旧密码错误")
        
        # 加密新密码（包含密码强度验证）
        try:
            db_user.password = encryption_utils.encrypt_password(user_update.new_password)
        except PasswordStrengthError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
        db.commit()
        return True

    def refresh_token(self, db: Session, refresh_token: str) -> TokenResponse:
        """
        使用刷新令牌获取新的访问令牌
        """
        try:
            # 解码刷新令牌获取用户ID
            payload = encryption_utils.decode_token(refresh_token, token_type="refresh")
            user_id = payload.get("user_id")
            
            if not user_id:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的刷新令牌")
            
            # 检查用户是否存在
            db_user = self._check_info(db, int(user_id))
            
            # 生成新的访问令牌
            new_access_token = encryption_utils.create_access_token(db_user.id)
            new_refresh_token = encryption_utils.create_refresh_token(db_user.id)
            
            # 更新数据库中的token
            db_user.token = new_access_token
            db.commit()
            
            return TokenResponse(
                access_token=new_access_token,
                refresh_token=new_refresh_token,
                token_type="bearer",
                expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
            )
            
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="刷新令牌已过期或无效")
    
    def refresh_token_by_user_id(self, db: Session, user_id: int) -> TokenResponse:
        """
        基于用户ID刷新token（适用于已认证的用户）
        """
        # 检查用户是否存在
        db_user = self._check_info(db, user_id)
        
        # 生成新的访问令牌和刷新令牌
        new_access_token = encryption_utils.create_access_token(db_user.id)
        new_refresh_token = encryption_utils.create_refresh_token(db_user.id)
        
        # 更新数据库中的token
        db_user.token = new_access_token
        db.commit()
        
        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    def revoke_token(self, db: Session, user_id: int) -> bool:
        """
        撤销用户令牌（登出）
        """
        db_user = self._check_info(db, user_id)
        db_user.token = None
        db.commit()
        return True
    
    def generate_password_reset_token(self, db: Session, phone: str) -> str:
        """
        生成密码重置令牌
        """
        # 检查手机号是否存在
        db_user = self.query(db).filter(self.model.phone == phone).first()
        if not db_user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="手机号不存在")
        
        # 生成重置令牌（实际中应该通过短信或邮件发送）
        reset_token = encryption_utils.generate_reset_token(phone)
        
        # 存储重置令牌（实际中应该存储在缓存中并设置过期时间）
        # 这里简化处理，直接返回令牌
        return reset_token
    
    def reset_password_with_token(self, db: Session, phone: str, token: str, new_password: str) -> bool:
        """
        使用重置令牌重置密码
        """
        # 检查手机号是否存在
        db_user = self.query(db).filter(self.model.phone == phone).first()
        if not db_user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="手机号不存在")
        
        # 验证重置令牌（实际中应该从缓存中获取并验证）
        # 这里简化处理
        expected_token = encryption_utils.generate_reset_token(phone)
        if not encryption_utils.verify_reset_token(phone, token, expected_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="重置令牌无效")
        
        # 加密新密码
        try:
            db_user.password = encryption_utils.encrypt_password(new_password)
        except PasswordStrengthError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
        db.commit()
        return True

    def get_user_by_token(self, db: Session, token: str) -> UserInfoResp:
        # 业务逻辑：根据token查询用户
        user_id = encryption_utils.token_decode(token)
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token 错误")
        db_user = self._check_info(db, user_id)
        # 业务逻辑：如果用户存在，返回用户信息
        return UserInfoResp.model_validate(db_user)


user_info_service = UserInfoService()
