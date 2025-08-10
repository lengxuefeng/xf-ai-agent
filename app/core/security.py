from typing import Optional

from fastapi import Header, HTTPException, status

from utils.pwd_utils import encryption_utils


def verify_token(
        auth_token: Optional[str] = Header(default=None, alias="Authorization"),
) -> str:
    """
    校验token
    :param auth_token: token
    :param phone: 手机号
    :return: token
    """
    if not auth_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token不能为空")
    valid_token = encryption_utils.token_decode(auth_token)
    if valid_token != auth_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token校验失败")
    return auth_token
