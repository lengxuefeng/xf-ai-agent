from typing import Optional, Dict, Any
import logging

from fastapi import Header, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from utils.pwd_utils import encryption_utils, TokenError

logger = logging.getLogger(__name__)

# 使用HTTPBearer更标准的认证方式
security = HTTPBearer(auto_error=False)


def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> int:
    """
    校验并解码Token，返回用户ID
    :param credentials: HTTP Bearer 认证凭据
    :return: user_id
    :raises HTTPException: 令牌无效时抛出
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少认证信息",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少Bearer Token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    try:
        user_id = encryption_utils.get_user_id_from_token(credentials.credentials)
        return user_id
    except TokenError as e:
        logger.warning(f"Token验证失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        logger.error(f"Token验证发生未知错误: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="认证失败",
            headers={"WWW-Authenticate": "Bearer"}
        )


def verify_refresh_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> int:
    """
    验证刷新令牌，返回用户ID
    :param credentials: HTTP Bearer 认证凭据
    :return: user_id
    :raises HTTPException: 令牌无效时抛出
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少刷新令牌",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    try:
        payload = encryption_utils.decode_token(credentials.credentials, token_type="refresh")
        user_id = payload.get("user_id")
        if not user_id:
            raise TokenError("刷新令牌中缺少用户ID")
        return int(user_id)
    except TokenError as e:
        logger.warning(f"刷新令牌验证失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )


def get_optional_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[int]:
    """
    可选的用户认证，不抛出异常
    :param credentials: HTTP Bearer 认证凭据
    :return: user_id 或 None
    """
    if not credentials or not credentials.credentials:
        return None
    
    try:
        return encryption_utils.get_user_id_from_token(credentials.credentials)
    except TokenError:
        return None
    except Exception as e:
        logger.debug(f"可选认证失败: {e}")
        return None


# 兼容旧的接口
def verify_token_header(
    auth_token: Optional[str] = Header(default=None, alias="Authorization"),
) -> int:
    """
    兼容旧版本的token验证方法
    :param auth_token: 来自请求头的Bearer Token
    :return: user_id
    """
    if not auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少Authorization头"
        )
    
    if not auth_token.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token格式错误，应为 'Bearer <token>'"
        )
    
    token = auth_token.split("Bearer ")[1]
    
    try:
        return encryption_utils.get_user_id_from_token(token)
    except TokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )

