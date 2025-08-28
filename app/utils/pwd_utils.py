# -*- coding: utf-8 -*-
import logging
import re
import secrets
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from starlette import status

from utils.config import settings

logger = logging.getLogger(__name__)

# 使用配置中的bcrypt rounds
# 添加更稳定的配置以避免bcrypt版本读取警告
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=settings.BCRYPT_ROUNDS,
    # 添加以下配置避免版本检查警告
    bcrypt__default_rounds=settings.BCRYPT_ROUNDS,
    bcrypt__min_rounds=4,
    bcrypt__max_rounds=31
)

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/user_info/login",
    auto_error=False
)


class PasswordStrengthError(Exception):
    """密码强度不足异常"""
    pass


class TokenError(Exception):
    """令牌相关异常"""
    pass


class EncryptionUtils:
    """
    加密工具类，提供密码加密、验证和JWT令牌处理功能
    """

    def __init__(self, token_length: int = 32):
        self.token_length = token_length
        self.jwt_secret_key = settings.JWT_SECRET_KEY
        self.jwt_algorithm = settings.JWT_ALGORITHM
        self.access_token_expire_minutes = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS

    def validate_password_strength(self, password: str) -> Tuple[bool, str]:
        """
        验证密码强度
        :param password: 密码
        :return: (是否通过, 错误信息)
        """
        if not settings.PASSWORD_STRENGTH_CHECK:
            return True, ""

        if len(password) < settings.PASSWORD_MIN_LENGTH:
            return False, f"密码长度不能少于{settings.PASSWORD_MIN_LENGTH}位"

        # 检查是否包含大小写字母、数字和特殊字符
        has_lower = bool(re.search(r'[a-z]', password))
        has_upper = bool(re.search(r'[A-Z]', password))
        has_digit = bool(re.search(r'\d', password))
        has_special = bool(re.search(r'[!@#$%^&*()_+\-=\[\]{};:\'"\\|,.<>/?]', password))

        strength_count = sum([has_lower, has_upper, has_digit, has_special])

        if strength_count < 3:
            return False, "密码必须包含大小写字母、数字和特殊字符中的至少3种"

        # 检查常用弱密码
        weak_passwords = [
            'password', '123456', 'password123', 'admin', 'qwerty',
            '12345678', '123456789', '1234567890', 'abc123'
        ]
        if password.lower() in weak_passwords:
            return False, "不能使用常见的弱密码"

        return True, ""

    def encrypt_password(self, password: str) -> str:
        """
        加密密码
        :param password: 原始密码
        :return: 加密后的密码
        :raises PasswordStrengthError: 密码强度不足时抛出
        """
        # 验证密码强度
        is_valid, error_msg = self.validate_password_strength(password)
        if not is_valid:
            raise PasswordStrengthError(error_msg)

        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        验证密码
        :param plain_password: 输入的密码
        :param hashed_password: 数据库中的密码
        :return: 是否验证通过
        """
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"密码验证失败: {e}")
            return False

    def generate_reset_token(self, phone: str) -> str:
        """
        生成密码重置令牌
        :param phone: 手机号
        :return: 密码重置令牌
        """
        # 使用更安全的随机数生成器
        token = secrets.token_urlsafe(self.token_length)
        return token

    def verify_reset_token(self, phone: str, token: str, stored_token: str) -> bool:
        """
        验证密码重置令牌
        :param phone: 手机号
        :param token: 提交的令牌
        :param stored_token: 存储的令牌
        :return: 是否验证通过
        """
        return secrets.compare_digest(token, stored_token)

    def create_access_token(self, user_id: int, extra_data: Optional[Dict[str, Any]] = None) -> str:
        """
        创建访问令牌
        :param user_id: 用户ID
        :param extra_data: 额外数据
        :return: JWT令牌
        """
        expires_delta = timedelta(minutes=self.access_token_expire_minutes)
        return self._create_token(
            data={"user_id": user_id, "type": "access", **(extra_data or {})},
            expires_delta=expires_delta
        )

    def create_refresh_token(self, user_id: int) -> str:
        """
        创建刷新令牌
        :param user_id: 用户ID
        :return: JWT刷新令牌
        """
        expires_delta = timedelta(days=self.refresh_token_expire_days)
        return self._create_token(
            data={"user_id": user_id, "type": "refresh"},
            expires_delta=expires_delta
        )

    def _create_token(self, data: Dict[str, Any], expires_delta: timedelta) -> str:
        """
        创建JWT令牌的内部方法
        :param data: 令牌数据
        :param expires_delta: 过期时间
        :return: JWT令牌
        """
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + expires_delta
        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),  # 发行时间
            "jti": secrets.token_urlsafe(16)  # JWT ID，用于唯一标识
        })

        try:
            encoded_jwt = jwt.encode(to_encode, self.jwt_secret_key, algorithm=self.jwt_algorithm)
            return encoded_jwt
        except Exception as e:
            logger.error(f"JWT令牌创建失败: {e}")
            raise TokenError("令牌创建失败")

    def decode_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """
        解码JWT令牌
        :param token: JWT令牌
        :param token_type: 令牌类型（access或refresh）
        :return: 令牌数据
        :raises TokenError: 令牌无效时抛出
        """
        print(f"token: {token}")
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret_key,
                algorithms=[self.jwt_algorithm],
                options={"verify_signature": True, "verify_exp": True}
            )

            # 验证令牌类型
            if payload.get("type") != token_type:
                raise TokenError(f"令牌类型不匹配，期望: {token_type}, 实际: {payload.get('type')}")

            return payload

        except jwt.ExpiredSignatureError:
            raise TokenError("令牌已过期")
        except jwt.InvalidTokenError as e:
            logger.warning(f"无效的JWT令牌: {e}")
            raise TokenError("无效的令牌")
        except Exception as e:
            logger.error(f"JWT令牌解码失败: {e}")
            raise TokenError("令牌解码失败")

    def get_user_id_from_token(self, token: str) -> int:
        """
        从令牌中获取用户ID
        :param token: JWT令牌
        :return: 用户ID
        :raises TokenError: 令牌无效时抛出
        """
        payload = self.decode_token(token)
        user_id = payload.get("user_id")

        if not user_id:
            raise TokenError("令牌中缺少用户ID")

        return int(user_id)

    def refresh_access_token(self, refresh_token: str) -> str:
        """
        使用刷新令牌生成新的访问令牌
        :param refresh_token: 刷新令牌
        :return: 新的访问令牌
        :raises TokenError: 刷新令牌无效时抛出
        """
        payload = self.decode_token(refresh_token, token_type="refresh")
        user_id = payload.get("user_id")

        if not user_id:
            raise TokenError("刷新令牌中缺少用户ID")

        return self.create_access_token(user_id)

    # 兼容旧的方法名
    def token_encode(self, user_id: int) -> str:
        """
        兼容旧的方法名，生成访问令牌
        :param user_id: 用户ID
        :return: JWT令牌
        """
        return self.create_access_token(user_id)

    def token_decode(self, token: str = Depends(oauth2_scheme)) -> int:
        """
        兼容旧的方法名，解码令牌获取用户ID
        :param token: JWT令牌
        :return: 用户ID
        :raises HTTPException: 令牌无效时抛出
        """
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="未提供令牌"
            )

        try:
            return self.get_user_id_from_token(token)
        except TokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e)
            )


# 创建全局实例
encryption_utils = EncryptionUtils()


# 为了向后兼容，提供一些函数式接口
def encrypt_password(password: str) -> str:
    """加密密码的函数式接口"""
    return encryption_utils.encrypt_password(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码的函数式接口"""
    return encryption_utils.verify_password(plain_password, hashed_password)


def create_access_token(user_id: int) -> str:
    """创建访问令牌的函数式接口"""
    return encryption_utils.create_access_token(user_id)


def decode_access_token(token: str) -> int:
    """解码访问令牌获取用户ID的函数式接口"""
    return encryption_utils.get_user_id_from_token(token)


if __name__ == '__main__':
    token = encryption_utils.create_access_token(user_id=1)
    print(token)
    user_id = encryption_utils.get_user_id_from_token(token)
    print(user_id)
