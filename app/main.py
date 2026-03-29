# -*- coding: utf-8 -*-
"""
FastAPI 应用主入口模块。

职责：
1. 创建 FastAPI 应用实例并配置 Swagger 文档
2. 注册全局异常处理器（业务异常、HTTP 异常、未知异常）
3. 配置 CORS 跨域中间件
4. 挂载所有 v1 版本 API 路由（用户、聊天、模型设置等）
5. 在启动时通过 SQLAlchemy 自动建表
"""
import logging
import os

from utils.tracing_guard import apply_tracing_env_guard

# tracing 全局守卫：默认关闭时，统一压制兼容 tracing 开关，避免被环境变量反向开启。
apply_tracing_env_guard()

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
from api.v1.interrupt_api import interrupt_router
from api.v1.metrics_api import metrics_router
from api.v1.health_api import health_router
from api.v1.terminal_api import terminal_router
from core.logger import setup_logger
from db import Base, engine
import models  # noqa: F401  # 确保所有 ORM 模型在 create_all 前被加载
from exceptions.business_exception import BusinessException
from schemas.response_model import ResponseModel
from core.middleware import ProcessTimeMiddleware, DynamicModelMiddleware, SSESafeGZipMiddleware

# 配置日志
setup_logger()
logger = logging.getLogger(__name__)

# 这行代码会在服务启动时，检查 PgSQL 里有没有表，没有就会自动按照你的 Model 创表！
AUTO_CREATE_TABLES = os.getenv("AUTO_CREATE_TABLES", "true").lower() == "true"
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


@app.on_event("startup")
async def startup_initialize_tables():
    """
    启动阶段初始化数据库表结构并启动 SessionPool 预热。

    设计说明：
    - 避免在模块 import 阶段就触发数据库连接，降低导入副作用；
    - 保留原有 create_all 行为，兼容当前项目运行方式；
    - 若禁用 AUTO_CREATE_TABLES，建议使用 Alembic 管理迁移。
    - SessionPool 在表结构就绪后启动，预热默认 Supervisor 图实例。
    """
    if not AUTO_CREATE_TABLES:
        logging.getLogger(__name__).info("AUTO_CREATE_TABLES=false，跳过 create_all，建议使用 Alembic 迁移。")
    else:
        try:
            Base.metadata.create_all(bind=engine)
            logging.getLogger(__name__).info("数据库表结构检查完成（create_all）。")
        except Exception as exc:
            # 启动期数据库不可用属于关键故障，直接抛出，防止服务带病启动
            logging.getLogger(__name__).exception(f"数据库初始化失败: {exc}")
            raise

    # 启动 Supervisor 图预热池（后台 daemon 线程，不阻塞服务启动）
    try:
        from services.session_pool import session_pool
        from constants.chat_service_constants import CHAT_DEFAULT_MODEL_CONFIG
        import asyncio
        # 在后台线程中执行预热（create_supervisor_graph 是同步阻塞操作）
        await asyncio.to_thread(session_pool.start, dict(CHAT_DEFAULT_MODEL_CONFIG))
        logging.getLogger(__name__).info("SessionPool 启动完成。")
    except Exception as pool_exc:
        # 预热池启动失败不阻断主服务，降级为按需编译
        logging.getLogger(__name__).warning(f"SessionPool 启动失败，已降级为按需编译: {pool_exc}")


@app.on_event("shutdown")
async def shutdown_session_pool():
    """服务关闭时停止 SessionPool 后台线程，释放资源。"""
    try:
        from services.session_pool import session_pool
        session_pool.stop()
    except Exception as exc:
        logging.getLogger(__name__).warning(f"SessionPool 停止异常: {exc}")


# ====== 辅助函数：获取请求体 ======
async def get_request_body(request: Request):
    try:
        body = await request.body()
        return body.decode("utf-8") if body else "{}"
    except Exception:
        return "{}"


# ===== 业务异常处理 =====
@app.exception_handler(BusinessException)
async def business_exception_handler(request: Request, exc: BusinessException):
    body = await get_request_body(request)
    logger.error(
        f"BusinessException | method={request.method} | url={request.url} "
        f"| code={exc.code} | message={exc.message} | body={body}",
        exc_info=True
    )
    return JSONResponse(
        status_code=exc.code,
        content=ResponseModel.fail(code=exc.code, message=exc.message).model_dump()
    )


# ===== HTTP 异常处理 (400/404/422 等) =====
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    body = await get_request_body(request)
    logger.error(
        f"HTTPException | method={request.method} | url={request.url} "
        f"| status_code={exc.status_code} | detail={exc.detail} | body={body}",
        exc_info=True
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=ResponseModel.fail(code=exc.status_code, message=exc.detail).model_dump()
    )


# ===== 全局异常处理 (500) =====
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    body = await get_request_body(request)
    logger.error(
        f"Unhandled Exception | method={request.method} | url={request.url} "
        f"| error={str(exc)} | body={body}",
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content=ResponseModel.fail(code=500, message="系统繁忙，请稍后重试").model_dump()
    )


# 注册统一性能监控与动态模型分发中间件 (注意中间件执行顺序：后注册的先运行)

# gzip 压缩：SSE 路径自动跳过，仅压缩普通响应。
app.add_middleware(SSESafeGZipMiddleware, minimum_size=500)
# 动态加载大模型配置 (比如通过 X-User-Model-Id 或 Body 中解析提取并缓存到 State 中)
app.add_middleware(DynamicModelMiddleware)
# 请求耗时与 Request_ID 计算
app.add_middleware(ProcessTimeMiddleware)

# 配置 CORS
# CORSMiddleware 必须最后注册，使其成为最外层中间件，拦截预检请求并注入跨域头
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
api_v1_router.include_router(interrupt_router)
api_v1_router.include_router(metrics_router)
api_v1_router.include_router(health_router)
api_v1_router.include_router(terminal_router)

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
