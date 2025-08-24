# XF-AI-Agent 多智能体系统说明文档

## 📖 目录

1. [系统概述](#系统概述)
2. [架构设计](#架构设计)
3. [核心功能](#核心功能)
4. [技术栈](#技术栈)
5. [模块详解](#模块详解)
6. [API接口](#api接口)
7. [配置管理](#配置管理)
8. [部署指南](#部署指南)
9. [使用示例](#使用示例)
10. [开发指南](#开发指南)

## 🌟 系统概述

XF-AI-Agent 是一个基于 LangGraph 框架构建的多智能体协作系统，支持动态模型配置、流式响应和多种 AI 服务集成。系统采用主管-专家模式，通过智能路由将用户请求分发给最适合的专家代理处理。

### 核心特性

- 🤖 **多智能体协作**：支持医疗、代码、SQL、天气、搜索等专业领域
- 🔄 **动态模型切换**：支持 Ollama、OpenRouter、Gemini、Silicon Flow 等多种模型服务
- 📡 **流式响应**：实现真正的打字机效果，提升用户体验
- 🧠 **RAG 增强**：支持检索增强生成，提升回答准确性
- 🔧 **可扩展架构**：易于添加新的智能体和功能模块

## 🏗️ 架构设计

### 系统架构图

```mermaid
graph TB
    subgraph "前端层"
        A[Web界面] --> B[模型配置面板]
        B --> C[聊天交互界面]
    end
    
    subgraph "API层"
        D[FastAPI服务] --> E[流式聊天接口]
        E --> F[模型配置处理]
    end
    
    subgraph "核心层"
        G[GraphRunner] --> H[Supervisor主管图]
        H --> I[路由决策]
        I --> J[专家代理调度]
    end
    
    subgraph "智能体层"
        K[医疗代理] --> L[代码代理]
        L --> M[SQL代理]
        M --> N[天气代理]
        N --> O[搜索代理]
    end
    
    subgraph "模型层"
        P[统一模型加载器] --> Q[Ollama]
        P --> R[OpenRouter]
        P --> S[Gemini]
        P --> T[Silicon Flow]
    end
    
    subgraph "工具层"
        U[天气API] --> V[搜索引擎]
        V --> W[数据库连接]
        W --> X[RAG向量库]
    end
    
    A --> D
    D --> G
    G --> K
    K --> P
    P --> U
```

### 核心组件关系

1. **主管图 (Supervisor Graph)**：负责分析用户请求并路由到合适的专家代理
2. **专家代理 (Expert Agents)**：各领域的专业处理单元
3. **统一模型加载器**：根据配置动态加载不同的 AI 模型
4. **流式处理器**：实现 SSE 格式的实时响应输出

## 🎯 核心功能

### 1. 智能路由系统

- **关键词匹配**：基于预定义关键词快速路由
- **LLM 语义理解**：复杂请求的智能分析
- **多轮对话支持**：维持上下文连续性

### 2. 专家代理系统

#### 医疗代理 (Medical Agent)
- 健康咨询和症状分析
- 药物信息查询
- 自动添加免责声明

#### 代码代理 (Code Agent)
- 代码生成和优化
- Bug 修复建议
- 人工审核机制

#### SQL 代理 (SQL Agent)
- 自然语言转 SQL
- 数据库查询优化
- 执行前确认机制

#### 天气代理 (Weather Agent)
- 实时天气查询
- 多城市并发查询
- 格式化天气展示

#### 搜索代理 (Search Agent)
- 网络信息检索
- 实时资讯获取
- 智能内容整合

### 3. 模型配置系统

支持的模型服务：

| 服务类型 | 支持模型 | 用途 |
|---------|---------|------|
| Ollama | qwen3:8b, llama2 等 | 本地部署 |
| OpenRouter | GPT-4, Claude 等 | 商业模型 |
| Gemini | gemini-1.5-pro | Google AI |
| Silicon Flow | Qwen/QwQ-32B | 硅基流动 |
| ZhipuAI | GLM-4-Flash | 智谱 AI |

### 4. RAG 增强系统

- **嵌入模型支持**：BGE-M3, Nomic-Embed 等
- **相似度阈值调整**：0.1-1.0 可调
- **向量数据库集成**：支持多种向量存储

## 🛠️ 技术栈

### 后端技术

- **框架**：FastAPI + LangGraph
- **AI 集成**：LangChain, OpenAI, Ollama
- **数据存储**：MySQL + MongoDB + Redis
- **异步处理**：Python asyncio
- **流式处理**：Server-Sent Events (SSE)

### 前端技术

- **基础**：HTML5 + CSS3 + JavaScript
- **样式**：Tailwind CSS
- **图标**：Font Awesome
- **实时通信**：EventSource (SSE)

### 开发工具

- **语言**：Python 3.8+
- **包管理**：pip / conda
- **代码质量**：Black, Flake8
- **版本控制**：Git

## 📁 模块详解

### 目录结构

```
app/
├── agent/                 # 智能体核心模块
│   ├── agents/           # 各专业领域代理
│   │   ├── weather_agent.py    # 天气查询代理
│   │   ├── code_agent.py       # 代码生成代理
│   │   ├── sql_agent.py        # SQL 查询代理
│   │   ├── medical_agent.py    # 医疗咨询代理
│   │   └── search_agent.py     # 搜索代理
│   ├── graphs/           # 图结构定义
│   │   ├── supervisor.py       # 主管图
│   │   └── state.py           # 状态管理
│   ├── llm/             # 模型加载模块
│   │   ├── unified_loader.py   # 统一模型加载器
│   │   ├── ollama_model.py     # Ollama 模型
│   │   └── loader_llm_multi.py # 多模型加载器
│   ├── tools/           # 工具模块
│   │   ├── weather_tools.py    # 天气工具
│   │   ├── search_tools.py     # 搜索工具
│   │   └── sql_tools.py        # SQL 工具
│   ├── graph_runner.py   # 图运行器
│   └── agent_builder.py  # 代理构建器
├── api/v1/              # API 接口
│   ├── chat_api.py      # 聊天接口
│   └── ...              # 其他接口
├── db/                  # 数据库模块
├── services/            # 业务服务
├── utils/               # 工具类
└── main.py             # 应用入口
```

### 关键模块详解

#### 1. GraphRunner (图运行器)

负责执行主图并处理流式输出：

```python
class GraphRunner:
    def __init__(self, model_config: dict = None):
        self.model_config = model_config or {}
        self.graph = None  # 延迟初始化
    
    def stream_run(self, user_input: str, session_id: str, model_config: dict = None):
        # 动态创建或更新图
        if self.graph is None or model_config:
            self.graph = create_graph(final_config)
        
        # 流式处理
        for event in self.graph.stream(initial_state):
            yield formatted_event
```

#### 2. UnifiedModelLoader (统一模型加载器)

根据配置动态加载模型：

```python
class UnifiedModelLoader:
    SERVICE_LOADERS = {
        'ollama': 'load_ollama_model',
        'openrouter': 'load_open_router',
        'netlify-gemini': 'load_open_router',
        'silicon-flow': 'load_silicon_flow',
        'zhipu': 'load_zhipu_model',
        'tongyi': 'load_tongyi_model',
    }
    
    @classmethod
    def load_chat_model(cls, config: ModelConfig):
        # 根据服务类型加载对应模型
        service = config.model_service.lower()
        return cls._load_model_by_service(service, config)
```

#### 3. 流式处理系统

实现真正的打字机效果：

```python
def stream_text(text: str, delay: float = 0.03):
    """将文本逐字符流式输出"""
    for char in text:
        yield to_sse({"type": "stream", "content": char})
        time.sleep(delay)
```

## 🔌 API 接口

### 主要接口

#### 1. 流式聊天接口

**POST** `/api/v1/chat/stream`

**请求参数：**

```json
{
  "user_input": "杞县今天的天气怎么样？",
  "session_id": "unique_session_id",
  "model": "gemini-1.5-pro",
  "model_service": "netlify-gemini",
  "deep_thinking_mode": "auto",
  "rag_enabled": false,
  "similarity_threshold": 0.7,
  "embedding_model": "bge-m3:latest"
}
```

**响应格式：**

```
data: {"type": "thinking", "content": "正在路由到: weather_agent..."}

data: {"type": "response_start", "content": ""}

data: {"type": "stream", "content": "根"}
data: {"type": "stream", "content": "据"}
data: {"type": "stream", "content": "查"}
...

data: {"type": "response_end", "content": ""}
```

### 响应事件类型

| 事件类型 | 描述 | 示例 |
|---------|------|------|
| thinking | 系统思考过程 | "正在路由到: weather_agent..." |
| response_start | 响应开始 | "" |
| stream | 逐字符内容 | "天" |
| response_end | 响应结束 | "" |
| error | 错误信息 | "模型加载失败" |
| interrupt | 需要用户确认 | "请确认执行SQL..." |

## ⚙️ 配置管理

### 环境变量配置

```bash
# 模型 API 密钥
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_API_BASE=https://openrouter.ai/api/v1
SILICONFLOW_API_KEY=your_siliconflow_key
SILICONFLOW_API_BASE=https://api.siliconflow.cn/v1
ZHIPUAI_API_KEY=your_zhipu_key

# 天气 API
HF_API_KEY=your_hefeng_weather_key
HF_REAL_TIME_URL=https://devapi.qweather.com/v7/weather/now
HF_LOCATION_URL=https://geoapi.qweather.com/v2/city/lookup

# 数据库配置
MYSQL_URL=mysql://user:pass@localhost/db
MONGODB_URL=mongodb://localhost:27017/db
REDIS_URL=redis://localhost:6379

# 搜索配置
TAVILY_API_KEY=your_tavily_key
```

### 模型配置参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| model | string | "gemini-1.5-pro" | 模型名称 |
| model_service | string | "netlify-gemini" | 模型服务类型 |
| deep_thinking_mode | string | "auto" | 深度思考模式 |
| rag_enabled | boolean | false | 是否启用 RAG |
| similarity_threshold | float | 0.7 | 相似度阈值 |
| embedding_model | string | "bge-m3:latest" | 嵌入模型 |

## 🚀 部署指南

### 1. 环境准备

```bash
# 克隆项目
git clone <repository_url>
cd xf-ai-agent

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置文件

创建 `.env` 文件并配置必要的环境变量。

### 3. 数据库初始化

```bash
# MySQL 初始化
mysql -u root -p < mysql.sql

# MongoDB 无需初始化
# Redis 确保服务运行
redis-server
```

### 4. 启动服务

```bash
# 开发模式
python main.py

# 生产模式
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. 访问服务

- 主页：http://localhost:8000
- API 文档：http://localhost:8000/docs
- 聊天测试：http://localhost:8000/chat

## 💡 使用示例

### 1. 天气查询

```
用户：杞县今天天气怎么样？
系统：正在路由到: weather_agent...
回复：根据查询结果，杞县今天天气是阴天，温度33°C，体感温度35°C...
```

### 2. 代码生成

```
用户：写一个Python的快速排序函数
系统：正在路由到: code_agent...
回复：以下是Python快速排序的实现：

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    ...

请输入 'ok' 接受代码，或输入修改意见。
```

### 3. SQL 查询

```
用户：查询销售额最高的产品
系统：正在路由到: sql_agent...
回复：将执行以下SQL语句，请审核：

SELECT product_name, SUM(sales_amount) as total_sales
FROM products p
JOIN sales s ON p.id = s.product_id
GROUP BY product_name
ORDER BY total_sales DESC
LIMIT 1;

请输入 'ok' 确认执行。
```

## 🔧 开发指南

### 添加新的智能体

1. **创建代理类**

```python
class NewAgent:
    def __init__(self, req: AgentRequest):
        self.model = req.model
        self.graph = create_tool_agent_executor(...)
    
    def run(self, req: AgentRequest) -> Generator:
        # 实现代理逻辑
        for event in self.graph.stream(state):
            yield event
```

2. **注册到主管图**

```python
# 在 supervisor.py 中添加
agent_classes["new_agent"] = AgentInfo(
    cls=NewAgent,
    description="新代理的描述",
    keywords=["关键词1", "关键词2"]
)
```

3. **添加工具（如需要）**

```python
# 在 tools/ 目录下创建工具
@tool
def new_tool(param: str) -> str:
    """工具描述"""
    return "工具结果"
```

### 扩展模型支持

1. **添加模型加载器**

```python
def load_new_model(model_name: str):
    """加载新的模型服务"""
    return NewModelClass(model=model_name, api_key=api_key)
```

2. **注册到统一加载器**

```python
SERVICE_LOADERS['new_service'] = 'load_new_model'
```

### 自定义流式处理

```python
def custom_stream_processor(content: str):
    """自定义流式处理逻辑"""
    for chunk in process_content(content):
        yield to_sse({"type": "custom", "content": chunk})
```

## 📚 参考资料

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [LangChain 文档](https://python.langchain.com/)

## 🤝 贡献指南

欢迎贡献代码和建议！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

---

**XF-AI-Agent** - 让 AI 协作更智能，让服务更贴心！