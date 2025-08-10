import logging
import uuid
from datetime import datetime, timezone, timedelta

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from starlette import status

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

"""
密码加密工具类
"""

SECURITY_KEY = "xfaixfaixfaixfaiagent"
ALGORITHMS = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/members/idcode/login",
                                     auto_error=False)  # auto_error=False 自己处理身份验证失败的情况。


class EncryptionUtils:
    def __init__(self, token_length: int = 32):
        self.token_length = token_length

    def generate_token(self, phone: str) -> str:
        """
        生成密码重置令牌
        :param phone: 手机号
        :return: 密码重置令牌
        """
        token = str(uuid.uuid5(uuid.NAMESPACE_OID, phone))
        return token[:self.token_length]

    def verify_token(self, phone: str, token: str) -> bool:
        """
        验证密码重置令牌
        :param phone: 手机号
        :param token: 密码重置令牌
        :return: 是否验证通过
        """
        return self.generate_token(phone) == token[:self.token_length]

    def encrypt_password(self, password: str) -> str:
        """
        加密密码
        :param password: 原始密码
        :return: 加密后的密码
        """
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        验证密码
        :param plain_password: 输入的密码
        :param hashed_password: 数据库中的密码
        :return: 是否验证通过
        """
        return pwd_context.verify(plain_password, hashed_password)

    # token加密
    def token_encode(self, user_id: int):
        token_expires = datetime.now(timezone.utc) + timedelta(days=7)  # 设置 token 有效期为 2天
        token_data = {
            "user_id": user_id,
            "exp": token_expires.timestamp()
        }
        return jwt.encode(token_data, SECURITY_KEY, ALGORITHMS)

    # token解密
    def token_decode(self, token: str = Depends(oauth2_scheme)) -> int:
        invalid_token = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token无效")
        user_id = None
        try:
            token_data = jwt.decode(token, SECURITY_KEY, ALGORITHMS)
            if token_data:
                user_id = token_data.get('user_id', None)
        except Exception as error:
            logging.info(f"token 解密失败, token: {token}, error: {error}")
            raise invalid_token
        if user_id is None:
            raise invalid_token
        return user_id


encryption_utils = EncryptionUtils()
