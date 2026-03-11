# XF-AI-Agent

基于 LangChain、LangGraph 和 FastAPI 构建的企业级多智能体 AI 代理系统。

## 项目简介

XF-AI-Agent 是一套完整的多智能体协作系统，采用主管-专家架构，通过智能路由将用户请求分发给最合适的领域专家处理。系统支持动态模型切换、流式响应、RAG 检索增强、人工审批等企业级功能。

### 核心特性

- **多智能体协同** - 医疗、代码、SQL、天气、搜索等 6+ 专业领域智能体
- **智能路由系统** - 三层路由架构：数据域路由 → 意图路由 → DAG 规划器
- **实时流式响应** - 基于 SSE 的真正实时输出，毫秒级延迟
- **人工审批机制** - 敏感操作支持人工审核，安全可控
- **RAG 检索增强** - 基于 PGVector 的高性能向量检索
- **多模型支持** - 统一接入 Ollama、OpenRouter、Gemini、通义千问等
- **历史压缩** - 自动压缩长对话历史，减少 Token 消耗
- **语义缓存** - 相似问题智能缓存，提升响应速度
- **配置中心** - 运行时参数集中管理，支持热更新

## 快速开始

### 环境准备

- Python 3.12+
- PostgreSQL 14+ (用于业务数据和向量检索)
- Node.js 18+ (前端开发)

### 安装后端

```bash
cd xf-ai-agent

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 复制环境变量配置
cp .env.example .env

# 编辑 .env 文件，填入必要的配置
# - 数据库连接信息
# - 大模型 API Key
# - 其他必要配置

# 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 访问服务

- **API 服务**: http://localhost:8000
- **Swagger 文档**: http://localhost:8000/docs
- **ReDoc 文档**: http://localhost:8000/redoc

## 📖 文档阅读顺序（按顺序阅读）

### 第一步：快速了解项目
📖 [完整项目说明文档.md](./完整项目说明文档.md) - **最核心的文档**，包含项目的所有内容（必读）

### 第二步：API 接口开发
📖 [API文档.md](./API文档.md) - 接口开发必备（开发人员必读）

### 第三步：深入专题文档（按需）
- [docs/函数方法与包说明表.md](./docs/函数方法与包说明表.md) - 函数方法大全（查阅使用）
- [docs/PostgreSQL数据库完整技术文档.md](./docs/PostgreSQL数据库完整技术文档.md) - 数据库运维（运维开发必读）
- [docs/database_init.sql](docs/database_init.sql) - 数据库初始化脚本（包含所有8张表）
- [docs/后端项目说明文档.md](./docs/后端项目说明文档.md) - 后端开发详解（开发人员）

## 项目结构

```
xf-ai-agent/
├── app/                           # 应用核心代码
│   ├── agent/                     # AI Agent 模块
│   │   ├── agents/                # 具体业务 Agent 实现
│   │   ├── graphs/                # 图结构定义
│   │   ├── tools/                 # 工具函数
│   │   ├── llm/                   # 大模型加载
│   │   ├── rag/                   # RAG 检索增强
│   │   ├── rules/                 # 规则引擎
│   │   ├── prompts/               # 系统提示词
│   │   ├── base.py                # Agent 基类
│   │   └── graph_runner.py        # 图执行器
│   ├── api/v1/                    # API 路由层
│   ├── core/                      # 核心中间件
│   ├── config/                    # 配置管理
│   ├── constants/                 # 常量定义
│   ├── db/                        # 数据库操作
│   ├── models/                    # 数据模型
│   ├── schemas/                   # 数据验证模型
│   ├── services/                  # 业务服务层
│   └── utils/                     # 工具函数
├── docs/                          # 专题文档
├── alembic/                       # 数据库迁移
├── .env                           # 环境变量配置
├── main.py                        # 应用启动入口
└── pyproject.toml                 # 项目依赖配置
```

## 技术栈

### 核心框架
- **FastAPI** 0.116.0+ - 现代化 Web 框架
- **LangChain** 1.2.10+ - LLM 应用框架
- **LangGraph** 0.2.50+ - 状态机与图编排引擎

### 数据库与存储
- **PostgreSQL** 14+ - 主数据库（支持向量检索）
- **PGVector** 0.3.6+ - 向量存储与检索
- **Redis** 6.4.0+ - 缓存与会话管理
- **SQLAlchemy** - ORM 框架
- **Alembic** - 数据库迁移工具

### LLM 与嵌入
- **langchain-openai** - OpenAI 兼容接口
- **langchain-google-genai** - Google Gemini
- **langchain-ollama** - 本地 Ollama 模型
- **langchain-community** - 社区模型集成

### 工具库
- **uvicorn** - ASGI 服务器
- **pydantic** - 数据验证
- **python-dotenv** - 环境变量管理
- **httpx** - HTTP 客户端
- **tavily-python** - 搜索 API 客户端

### 安全与认证
- **PyJWT** - JWT 令牌生成与验证
- **passlib** - 密码加密
- **bcrypt** - 密码哈希

## 部署说明

### 开发环境

```bash
# 使用 uvicorn 启动开发服务器
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 生产环境

```bash
# 使用 gunicorn 启动（推荐）
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -

# 或使用 uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker 部署

```bash
# 构建镜像
docker build -t xf-ai-agent .

# 运行容器
docker run -d \
  --name xf-ai-agent \
  -p 8000:8000 \
  --env-file .env \
  xf-ai-agent
```

## 主要功能

### 1. 智能对话
- 实时流式输出
- 多轮对话记忆
- 上下文自动管理

### 2. 专业智能体
- **医疗 Agent** - 健康咨询、症状分析
- **代码 Agent** - 代码生成、调试、优化
- **SQL Agent** - 数据库查询、分析
- **天气 Agent** - 天气查询、预报
- **搜索 Agent** - 联网搜索、信息检索
- **云柚 Agent** - 云柚业务系统交互

### 3. 安全机制
- JWT 认证
- 权限控制
- 人工审批
- SQL 注入防护

### 4. 性能优化
- 语义缓存
- 历史压缩
- 并发执行
- 负载均衡

## 贡献指南

欢迎贡献代码、提出建议或报告问题！

## 许可证

MIT License
