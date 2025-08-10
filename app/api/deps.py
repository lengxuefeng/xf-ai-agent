
from fastapi import Depends

from app.core.security import validate_token
from app.schemas.common import BaseRequestParams, PageParams
from db import get_db


def get_base_params(params: BaseRequestParams = Depends()):
    validate_token(params.token)
    return params

def get_page_params(params: PageParams = Depends()):
    return params

def get_db_session():
    return next(get_db())
