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
    # 校验 token 格式
    if not auth_token.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token格式错误")
    # 提取 token 中的用户 ID
    user_id = encryption_utils.token_decode(auth_token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token无效")

    return user_id

