from fastapi import APIRouter
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import JSONResponse

from api.v1.chat_api import chat_router
from api.v1.chat_history_api import chat_history_router
from api.v1.user_info_api import user_router
from api.v1.model_setting_api import router as model_setting_router
from api.v1.user_model_api import router as user_model_router
from api.v1.user_mcp_api import router as user_mcp_router
from app.core.logger import setup_logger
from exceptions.business_exception import BusinessException
from schemas.response_model import ResponseModel

# 配置日志
setup_logger()

app = FastAPI(
    title="LangGraph Agent FastAPI",
    description="基于 LangChain、LangGraph 和 FastAPI 构建的智能代理后端项目。",
    version="1.0.0",
    docs_url="/docs",
    # swagger_ui_parameters={
    #     "swaggerJsUrl": "https://cdn.bootcdn.net/ajax/libs/swagger-ui/5.10.3/swagger-ui-bundle.min.js",
    #     "swaggerCssUrl": "https://cdn.bootcdn.net/ajax/libs/swagger-ui/5.10.3/swagger-ui.min.css",
    #     "swaggerFaviconUrl": "https://cdn.bootcdn.net/ajax/libs/swagger-ui/5.10.3/favicon-32x32.png",
    # }
)


# ===== 全局异常处理器 =====
@app.exception_handler(BusinessException)
async def business_exception_handler(request: Request, exc: BusinessException):
    return JSONResponse(
        status_code=exc.code,  # 使用业务异常的错误码作为HTTP状态码
        content=ResponseModel.fail(code=exc.code, message=exc.message).model_dump()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,  # 使用HTTP异常的状态码
        content=ResponseModel.fail(code=exc.status_code, message=exc.detail).model_dump()
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,  # 系统异常返回500状态码
        content=ResponseModel.fail(code=500, message="系统繁忙，请稍后重试").model_dump()
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

# 创建 v1 版本的 API 路由
api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(user_router)
api_v1_router.include_router(chat_router)
api_v1_router.include_router(chat_history_router)
api_v1_router.include_router(model_setting_router)
api_v1_router.include_router(user_model_router)
api_v1_router.include_router(user_mcp_router)

# 在主应用中包含 v1 路由
app.include_router(api_v1_router)

# HTML文件放在templates目录
templates = Jinja2Templates(directory="app/templates")


# 定义根路由，返回 HTML 页面
@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 聊天测试页面
@app.get("/chat", include_in_schema=False, response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})
