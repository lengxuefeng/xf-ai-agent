from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.v1.chat_api import chat_router
from api.v1.user_info_api import user_router
from app.api.v1 import chat_api as v1_endpoints

app = FastAPI(
    title="LangGraph Agent FastAPI",
    description="基于 LangChain、LangGraph 和 FastAPI 构建的智能代理后端项目。",
    version="1.0.0",
)

# 配置 CORS
# 这是一个最宽松的 CORS 配置，通常用于开发环境以排除所有 CORS 问题。
# 在生产环境中，请务必限制 allow_origins、allow_methods 和 allow_headers。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=False,  # 不允许凭证，因为允许所有来源时不能允许凭证
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)

# 包含 API 路由
app.include_router(user_router)
app.include_router(chat_router)


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "欢迎来到 LangGraph Agent FastAPI! 访问 /docs 查看 API 文档。"}
