# xf-ai-agent 项目

这是一个基于 LangChain、LangGraph 和 FastAPI 构建的智能代理后端项目。它提供了一个 HTTP API 接口，用于与 LangGraph 驱动的智能代理进行交互，并支持知识库问答 (RAG)。

[langchain文档](https://python.langchain.com/docs)  
[langgraph文档](https://langchain-ai.github.io/langgraph/)


## 项目结构

```
xf-ai-agent/
├── app/                  # 应用核心逻辑
│   ├── api/              # API 路由定义
│   │   └── v1/           # API 版本化
│   │       └── endpoints.py # 包含所有 FastAPI 路由端点
│   ├── models/           # Pydantic 模型定义
│   │   └── schemas.py    # 用于 API 请求和响应的数据结构
│   ├── services/         # 业务逻辑层，包含核心功能实现
│   │   ├── agent.py      # LangGraph Agent 的核心定义 (状态、节点、边)
│   │   ├── retriever.py  # 负责 RAG 的检索器和向量库创建
│   │   └── tools.py      # 为 Agent 定义的工具 (例如，文档搜索)
│   ├── utils/            # 通用工具函数、帮助类等
│   │   ├── config.py     # 配置加载 (例如，从 .env 读取环境变量)
│   │   └── common.py     # 其他通用辅助函数
│   └── dependencies.py   # FastAPI 依赖注入函数 (例如，获取共享资源、认证)
├── data/                 # 存放 RAG 知识库源文件
│   └── knowledge.txt
├── vectorstore/          # 存放生成的向量数据库
├── .env                  # 环境变量 (例如 GOOGLE_API_KEY)
├── main.py               # FastAPI 应用入口，负责加载应用和路由
├── pyproject.toml        # 项目依赖和元数据配置
└── README.md             # 项目说明文档 (中文)
```

## 安装与运行

### 1. 克隆项目

```bash
git clone <项目仓库地址>
cd xf-ai-agent
```

### 2. 创建并激活虚拟环境 (使用 uv)

```bash
uv venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

### 3. 安装依赖 (使用 uv)

```bash
uv pip install .
```

### 4. 生成 requirements.txt (可选)

如果您需要为其他环境生成 `requirements.txt` 文件，可以使用以下命令：

```bash
uv pip freeze > requirements.txt
```

### 5. 使用 requirements.txt 安装依赖 (可选)

如果您已经有了 `requirements.txt` 文件，可以使用 `pip` 或 `uv pip` 来安装依赖：

```bash
# 使用 pip
pip install -r requirements.txt

# 或者使用 uv pip
uv pip install -r requirements.txt
```

### 6. 配置环境变量

在项目根目录下创建 `.env` 文件，并填入您的 Google API Key：

```
GOOGLE_API_KEY="YOUR_API_KEY"
```

### 5. 准备知识库数据

将您的知识文本文件放入 `data/` 目录下，例如 `data/knowledge.txt`。

### 6. 运行 FastAPI 应用

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

应用将在 `http://0.0.0.0:8000` 启动。您可以通过访问 `http://0.0.0.0:8000/docs` 查看 API 文档 (Swagger UI)。

## API 文档

访问 `http://0.0.0.0:8000/docs` 查看自动生成的 API 文档。

## 使用示例

### 聊天 API

向 `/api/v1/chat` 端点发送 POST 请求，以与智能代理进行交互。

**请求示例:**

```json
{
  "message": "你好，请问有什么可以帮助你的吗？"
}
```

**响应示例:**

```json
{
  "response": "你好！我是一个智能助手，很高兴为您服务。"
}
```

## 核心组件说明

### LangGraph Agent (`app/services/agent.py`)

定义了智能代理的状态、节点和边。它能够根据用户输入和内部逻辑进行多步骤推理和工具使用。

### 知识检索器 (`app/services/retriever.py`)

负责从 `data/` 目录加载文本数据，创建向量嵌入，并构建 FAISS 向量存储。它提供了一个检索接口，供 Agent 在需要时查询相关文档片段。

### 工具 (`app/services/tools.py`)

定义了 Agent 可以使用的外部工具，例如文档搜索工具。这些工具通过 LangChain 的 `Tool` 接口集成。

### Pydantic 模型 (`app/models/schemas.py`)

定义了 API 请求和响应的数据结构，确保数据传输的类型安全和清晰。

### 配置 (`app/utils/config.py`)

处理环境变量和应用配置的加载。

## 贡献

欢迎提交 Pull Request 或报告 Issue。

## 许可证

本项目采用 MIT 许可证。