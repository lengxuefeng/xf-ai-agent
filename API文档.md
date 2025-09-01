# XF AI Agent API 接口文档

## 概述

XF AI Agent 提供了完整的 RESTful API 接口，支持用户管理、对话处理、模型配置、文档处理等功能。所有接口均基于 FastAPI 框架构建，提供自动生成的 OpenAPI 文档。

**API 基础信息**:
- 基础 URL: `http://localhost:8000/api/v1`
- 文档地址: `http://localhost:8000/docs` (Swagger UI)
- 认证方式: JWT Bearer Token
- 内容类型: `application/json`

## 认证说明

除了用户注册和登录接口外，所有 API 都需要在请求头中携带有效的 JWT Token：

```http
Authorization: Bearer <your-jwt-token>
```

Token 获取方式：通过用户登录接口获得，有效期为 24 小时。

## 用户管理接口

### 1. 用户注册

**接口**: `POST /users/register`

**描述**: 创建新用户账户

**请求体**:
```json
{
  "username": "string",      // 用户名，3-20字符，唯一
  "email": "string",         // 邮箱地址，必须有效格式
  "password": "string",      // 密码，至少8字符
  "full_name": "string"      // 可选，用户全名
}
```

**响应**:
```json
{
  "code": 200,
  "message": "用户注册成功",
  "data": {
    "user_id": 1,
    "username": "testuser",
    "email": "test@example.com",
    "full_name": "Test User",
    "is_active": true,
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

**错误响应**:
- `400`: 参数验证失败
- `409`: 用户名或邮箱已存在

### 2. 用户登录

**接口**: `POST /users/login`

**描述**: 用户身份验证，获取访问令牌

**请求体**:
```json
{
  "username": "string",      // 用户名或邮箱
  "password": "string"       // 密码
}
```

**响应**:
```json
{
  "code": 200,
  "message": "登录成功",
  "data": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer",
    "expires_in": 86400,
    "user_info": {
      "user_id": 1,
      "username": "testuser",
      "email": "test@example.com",
      "full_name": "Test User"
    }
  }
}
```

**错误响应**:
- `401`: 用户名或密码错误
- `403`: 账户已被禁用

### 3. 获取用户信息

**接口**: `GET /users/profile`

**描述**: 获取当前用户的详细信息

**请求头**:
```http
Authorization: Bearer <token>
```

**响应**:
```json
{
  "code": 200,
  "message": "获取成功",
  "data": {
    "user_id": 1,
    "username": "testuser",
    "email": "test@example.com",
    "full_name": "Test User",
    "is_active": true,
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z"
  }
}
```

### 4. 更新用户信息

**接口**: `PUT /users/profile`

**描述**: 更新当前用户的基本信息

**请求体**:
```json
{
  "full_name": "string",     // 可选，用户全名
  "email": "string"          // 可选，新邮箱地址
}
```

**响应**:
```json
{
  "code": 200,
  "message": "更新成功",
  "data": {
    "user_id": 1,
    "username": "testuser",
    "email": "newemail@example.com",
    "full_name": "New Full Name",
    "updated_at": "2024-01-01T12:00:00Z"
  }
}
```

### 5. 修改密码

**接口**: `POST /users/change-password`

**描述**: 修改当前用户密码

**请求体**:
```json
{
  "old_password": "string",  // 当前密码
  "new_password": "string"   // 新密码，至少8字符
}
```

**响应**:
```json
{
  "code": 200,
  "message": "密码修改成功",
  "data": null
}
```

**错误响应**:
- `400`: 原密码错误
- `422`: 新密码格式不符合要求

## 对话接口

### 1. 发送消息

**接口**: `POST /chat`

**描述**: 发送消息给 AI 助手并获取回复

**请求体**:
```json
{
  "message": "string",           // 用户消息内容
  "session_id": "string",        // 可选，会话ID
  "model_config": {              // 可选，模型配置
    "model_name": "string",
    "temperature": 0.7,
    "max_tokens": 2000
  },
  "enable_rag": true,            // 可选，是否启用RAG
  "thinking_mode": "auto"        // 可选，思考模式: auto/manual/off
}
```

**响应**:
```json
{
  "code": 200,
  "message": "处理成功",
  "data": {
    "response": "AI助手的回复内容",
    "session_id": "session_123",
    "message_id": "msg_456",
    "thinking_process": "思考过程...",  // 可选
    "sources": [                        // RAG相关来源，可选
      {
        "title": "文档标题",
        "content": "相关内容片段",
        "score": 0.85
      }
    ],
    "model_info": {
      "model_name": "gpt-4",
      "tokens_used": 150
    }
  }
}
```

### 2. 流式对话

**接口**: `POST /chat/stream`

**描述**: 发送消息并以流式方式接收回复

**请求体**: 同上

**响应**: Server-Sent Events (SSE) 流

```
data: {"type": "start", "session_id": "session_123"}

data: {"type": "thinking", "content": "正在思考用户的问题..."}

data: {"type": "token", "content": "这是"}

data: {"type": "token", "content": "一个"}

data: {"type": "token", "content": "流式"}

data: {"type": "token", "content": "回复"}

data: {"type": "sources", "content": [{"title": "文档1", "score": 0.9}]}

data: {"type": "end", "message_id": "msg_456", "tokens_used": 150}
```

**事件类型说明**:
- `start`: 开始处理
- `thinking`: 思考过程
- `token`: 回复内容片段
- `sources`: RAG 检索来源
- `error`: 错误信息
- `end`: 处理完成

## 对话历史接口

### 1. 获取对话历史列表

**接口**: `GET /chat_history/`

**描述**: 获取当前用户的对话历史列表

**查询参数**:
- `page`: 页码，默认1
- `size`: 每页数量，默认20
- `search`: 搜索关键词，可选

**响应**:
```json
{
  "code": 200,
  "message": "获取成功",
  "data": {
    "items": [
      {
        "id": 1,
        "session_id": "session_123",
        "title": "关于Python编程的讨论",
        "message_count": 10,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T01:00:00Z"
      }
    ],
    "total": 50,
    "page": 1,
    "size": 20,
    "pages": 3
  }
}
```

### 2. 获取对话详情

**接口**: `GET /chat_history/{session_id}`

**描述**: 获取指定会话的完整对话记录

**响应**:
```json
{
  "code": 200,
  "message": "获取成功",
  "data": {
    "session_id": "session_123",
    "title": "关于Python编程的讨论",
    "messages": [
      {
        "id": "msg_1",
        "role": "user",
        "content": "什么是Python？",
        "timestamp": "2024-01-01T00:00:00Z"
      },
      {
        "id": "msg_2",
        "role": "assistant",
        "content": "Python是一种高级编程语言...",
        "thinking_process": "用户询问Python的基本概念...",
        "sources": [],
        "timestamp": "2024-01-01T00:01:00Z"
      }
    ],
    "model_info": {
      "model_name": "gpt-4",
      "total_tokens": 500
    },
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T01:00:00Z"
  }
}
```

### 3. 删除对话历史

**接口**: `DELETE /chat_history/{session_id}`

**描述**: 删除指定的对话记录

**响应**:
```json
{
  "code": 200,
  "message": "删除成功",
  "data": null
}
```

### 4. 更新对话标题

**接口**: `PUT /chat_history/{session_id}/title`

**描述**: 更新对话的标题

**请求体**:
```json
{
  "title": "新的对话标题"
}
```

**响应**:
```json
{
  "code": 200,
  "message": "更新成功",
  "data": {
    "session_id": "session_123",
    "title": "新的对话标题"
  }
}
```

## 模型设置接口

### 1. 获取模型服务列表

**接口**: `GET /model_setting/`

**描述**: 获取系统可用的模型服务配置

**查询参数**:
- `enabled_only`: 只返回启用的服务，默认false

**响应**:
```json
{
  "code": 200,
  "message": "获取成功",
  "data": [
    {
      "id": 1,
      "service_name": "OpenAI",
      "service_type": "openai",
      "service_url": "https://api.openai.com/v1",
      "api_key_template": "sk-...",
      "icon": "FiCpu",
      "models": {
        "chat": ["gpt-4", "gpt-3.5-turbo"],
        "embedding": ["text-embedding-ada-002"],
        "vision": ["gpt-4-vision-preview"],
        "other": []
      },
      "description": "OpenAI官方API服务",
      "is_system_default": true,
      "is_enabled": true,
      "created_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

### 2. 创建模型服务

**接口**: `POST /model_setting/`

**描述**: 创建新的模型服务配置（管理员功能）

**请求体**:
```json
{
  "service_name": "Custom Service",
  "service_type": "openai",
  "service_url": "https://api.custom.com/v1",
  "api_key_template": "custom-key-...",
  "icon": "FiCpu",
  "models": {
    "chat": ["custom-model-1"],
    "embedding": ["custom-embed-1"],
    "vision": [],
    "other": []
  },
  "description": "自定义模型服务",
  "is_enabled": true
}
```

### 3. 更新模型服务

**接口**: `PUT /model_setting/{id}`

**描述**: 更新指定的模型服务配置

**请求体**: 同创建接口

### 4. 删除模型服务

**接口**: `DELETE /model_setting/{id}`

**描述**: 删除指定的模型服务配置

## 用户模型配置接口

### 1. 获取用户模型配置

**接口**: `GET /user_model/`

**描述**: 获取当前用户的模型配置列表

**响应**:
```json
{
  "code": 200,
  "message": "获取成功",
  "data": [
    {
      "id": 1,
      "user_id": 1,
      "model_setting_id": 1,
      "service_name": "OpenAI",
      "selected_model": "gpt-4",
      "api_key": "sk-...",
      "api_url": "https://api.openai.com/v1",
      "custom_config": {
        "temperature": 0.7,
        "max_tokens": 2000
      },
      "is_active": true,
      "created_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

### 2. 创建用户模型配置

**接口**: `POST /user_model/`

**描述**: 为当前用户创建新的模型配置

**请求体**:
```json
{
  "model_setting_id": 1,        // 可选，基于系统模型服务
  "service_name": "OpenAI",
  "selected_model": "gpt-4",
  "api_key": "sk-your-key",
  "api_url": "https://api.openai.com/v1",  // 可选
  "custom_config": {            // 可选
    "temperature": 0.7,
    "max_tokens": 2000
  }
}
```

### 3. 更新用户模型配置

**接口**: `PUT /user_model/{id}`

**描述**: 更新指定的用户模型配置

**请求体**: 同创建接口

### 4. 激活模型配置

**接口**: `PUT /user_model/{id}/activate`

**描述**: 激活指定的模型配置为当前使用

**响应**:
```json
{
  "code": 200,
  "message": "激活成功",
  "data": {
    "id": 1,
    "is_active": true
  }
}
```

### 5. 停用模型配置

**接口**: `PUT /user_model/{id}/deactivate`

**描述**: 停用指定的模型配置

### 6. 删除用户模型配置

**接口**: `DELETE /user_model/{id}`

**描述**: 删除指定的用户模型配置

## 文档处理接口

### 1. 上传文档

**接口**: `POST /documents/upload`

**描述**: 上传文档文件用于RAG处理

**请求**: Multipart form data
- `files`: 文件列表，支持PDF、TXT格式
- `user_id`: 用户ID

**响应**:
```json
{
  "code": 200,
  "message": "上传成功",
  "data": {
    "uploaded_files": [
      {
        "filename": "document.pdf",
        "size": 1024000,
        "file_id": "file_123"
      }
    ],
    "total_files": 1,
    "total_size": 1024000
  }
}
```

### 2. 处理文档

**接口**: `POST /documents/process`

**描述**: 处理上传的文档，生成向量索引

**请求体**:
```json
{
  "file_ids": ["file_123", "file_456"],  // 要处理的文件ID列表
  "chunk_size": 1000,                    // 可选，文本分块大小
  "chunk_overlap": 200,                  // 可选，分块重叠大小
  "embedding_model": "text-embedding-ada-002"  // 可选，嵌入模型
}
```

**响应**:
```json
{
  "code": 200,
  "message": "处理成功",
  "data": {
    "processed_files": 2,
    "total_chunks": 150,
    "vector_store_id": "vs_789",
    "processing_time": 45.6
  }
}
```

### 3. 获取文档列表

**接口**: `GET /documents/`

**描述**: 获取用户上传的文档列表

**响应**:
```json
{
  "code": 200,
  "message": "获取成功",
  "data": [
    {
      "file_id": "file_123",
      "filename": "document.pdf",
      "size": 1024000,
      "status": "processed",
      "chunks_count": 75,
      "uploaded_at": "2024-01-01T00:00:00Z",
      "processed_at": "2024-01-01T00:05:00Z"
    }
  ]
}
```

### 4. 删除文档

**接口**: `DELETE /documents/{file_id}`

**描述**: 删除指定的文档及其向量数据

## MCP 配置接口

### 1. 获取MCP配置

**接口**: `GET /user_mcp/`

**描述**: 获取用户的MCP（Model Context Protocol）配置

**响应**:
```json
{
  "code": 200,
  "message": "获取成功",
  "data": [
    {
      "id": 1,
      "user_id": 1,
      "mcp_name": "File System",
      "mcp_type": "filesystem",
      "config": {
        "root_path": "/home/user/documents",
        "allowed_extensions": [".txt", ".md", ".py"]
      },
      "is_enabled": true,
      "created_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

### 2. 创建MCP配置

**接口**: `POST /user_mcp/`

**描述**: 创建新的MCP配置

**请求体**:
```json
{
  "mcp_name": "File System",
  "mcp_type": "filesystem",
  "config": {
    "root_path": "/home/user/documents",
    "allowed_extensions": [".txt", ".md", ".py"]
  },
  "is_enabled": true
}
```

### 3. 更新MCP配置

**接口**: `PUT /user_mcp/{id}`

**描述**: 更新指定的MCP配置

### 4. 删除MCP配置

**接口**: `DELETE /user_mcp/{id}`

**描述**: 删除指定的MCP配置

## 错误响应格式

所有接口的错误响应都遵循统一格式：

```json
{
  "code": 400,
  "message": "错误描述",
  "data": null,
  "errors": [                    // 可选，详细错误信息
    {
      "field": "username",
      "message": "用户名已存在"
    }
  ]
}
```

## 状态码说明

- `200`: 请求成功
- `201`: 创建成功
- `400`: 请求参数错误
- `401`: 未认证或认证失败
- `403`: 权限不足
- `404`: 资源不存在
- `409`: 资源冲突
- `422`: 请求参数验证失败
- `500`: 服务器内部错误

## 请求限制

- 单个请求最大大小: 10MB
- 文件上传最大大小: 10MB
- API 请求频率限制: 100次/分钟/用户
- 并发连接数限制: 10个/用户

## 使用示例

### Python 示例

```python
import requests
import json

# 基础配置
BASE_URL = "http://localhost:8000/api/v1"
headers = {"Content-Type": "application/json"}

# 用户登录
login_data = {
    "username": "testuser",
    "password": "password123"
}
response = requests.post(f"{BASE_URL}/users/login", 
                        json=login_data, headers=headers)
token = response.json()["data"]["access_token"]

# 设置认证头
auth_headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}"
}

# 发送对话消息
chat_data = {
    "message": "你好，请介绍一下Python编程语言",
    "enable_rag": True,
    "thinking_mode": "auto"
}
response = requests.post(f"{BASE_URL}/chat", 
                        json=chat_data, headers=auth_headers)
print(response.json()["data"]["response"])
```

### JavaScript 示例

```javascript
const BASE_URL = 'http://localhost:8000/api/v1';

// 用户登录
async function login(username, password) {
    const response = await fetch(`${BASE_URL}/users/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
    });
    const data = await response.json();
    return data.data.access_token;
}

// 流式对话
async function streamChat(token, message) {
    const response = await fetch(`${BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ message })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                console.log('Received:', data);
            }
        }
    }
}
```

## 版本更新

### v1.0.0 (当前版本)
- 完整的用户认证和管理接口
- 对话和流式对话接口
- 模型配置管理接口
- 文档处理和RAG接口
- MCP协议支持接口
- 对话历史管理接口

---

如需更多技术支持，请联系开发团队或查看在线文档。
