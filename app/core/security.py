from typing import Optional

from fastapi import Header, HTTPException, status

from utils.pwd_utils import encryption_utils


def verify_token(
        auth_token: Optional[str] = Header(default=None, alias="Authorization"),
) -> int:
    """
    校验并解码Token，返回用户ID
    :param auth_token: 来自请求头的Bearer Token
    :return: user_id
    """
    if not auth_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token不能为空")

    # 假设 token_decode 失败会返回 None 或 False
    user_id = encryption_utils.token_decode(auth_token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token无效")

    return user_id

