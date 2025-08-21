from fastapi import APIRouter

from schemas.common import PageParams

chat_history_router = APIRouter(prefix="/history", tags=["历史对话信息"])

"""
历史对话信息接口
"""


@chat_history_router.post("/page", response_model=dict)
def page(page_params: PageParams):
    """
    分页查询
    """
    pass


