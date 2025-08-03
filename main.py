from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import endpoints as v1_endpoints

app = FastAPI(
    title="LangGraph Agent FastAPI",
    description="基于 LangChain、LangGraph 和 FastAPI 构建的智能代理后端项目。",
    version="1.0.0",
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应限制为您的前端域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)

# 包含 API 路由
app.include_router(v1_endpoints.router, prefix="/api/v1", tags=["v1"])

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "欢迎来到 LangGraph Agent FastAPI! 访问 /docs 查看 API 文档。"}
