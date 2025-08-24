# API 文档

本文档详细说明了 `xf-ai-agent` 后端项目中的所有核心 API 接口。

## 认证

所有需要认证的接口均通过 `Authorization` 请求头中的 `Bearer Token` 进行验证。

---

## 用户模型接口 (`/api/v1/user_model`)

管理用户个人专属的模型服务配置。

### 1. 获取当前用户的所有模型配置

- **GET** `/api/v1/user_model/`
- **摘要**: 获取当前认证用户的所有模型配置列表。
- **成功响应 (200)**:
  ```json
  {
    "code": 200,
    "message": "请求成功",
    "data": [
      {
        "id": 1,
        "user_id": 123,
        "model_setting_id": 1,
        "model_name": "gemini-1.5-pro",
        "api_key": "sk-...",
        "api_url": "https://api.gemini.com/v1",
        "create_time": "2025-08-10T10:00:00Z",
        "update_time": "2025-08-10T10:00:00Z"
      }
    ]
  }
  ```

### 2. 创建新的用户模型配置

- **POST** `/api/v1/user_model/`
- **摘要**: 为当前认证的用户创建一个新的模型服务配置。
- **请求体**:
  ```json
  {
    "user_id": 123,
    "model_setting_id": 1,
    "model_name": "gpt-4o",
    "api_key": "sk-abc...",
    "api_url": "https://api.openai.com/v1"
  }
  ```
- **成功响应 (200)**: 返回新创建的模型配置对象。

### 3. 更新用户模型配置

- **PUT** `/api/v1/user_model/{model_id}`
- **摘要**: 更新指定ID的用户模型配置。
- **参数**:
  - `model_id` (路径参数, integer, 必需): 用户模型配置记录的唯一ID。
- **请求体**: (所有字段均为可选)
  ```json
  {
    "model_name": "gpt-4-turbo",
    "api_key": "sk-def...",
    "api_url": "https://api.openai.com/v1/chat/completions"
  }
  ```
- **成功响应 (200)**: 返回更新后的模型配置对象。

### 4. 删除用户模型配置

- **DELETE** `/api/v1/user_model/{model_id}`
- **摘要**: 删除指定ID的用户模型配置。
- **参数**:
  - `model_id` (路径参数, integer, 必需): 用户模型配置记录的唯一ID。
- **成功响应 (200)**:
  ```json
  {
    "code": 200,
    "message": "请求成功",
    "data": {
      "message": "删除成功"
    }
  }
  ```

---

## 模型市场接口 (`/api/v1/model_setting`)

管理平台提供的、可供所有用户选择的基础模型配置。

### 1. 获取当前用户的所有模型配置

- **GET** `/api/v1/model_setting/`
- **摘要**: 获取当前认证用户可用的所有模型市场配置。
- **成功响应 (200)**:
  ```json
  {
    "code": 200,
    "message": "请求成功",
    "data": [
      {
        "id": 1,
        "user_id": 123,
        "model_name": "Gemini 1.5 Pro",
        "model_type": "google",
        "model_url": "https://api.gemini.com/v1",
        "model_params": "{\"temperature\": 0.7}",
        "model_desc": "谷歌最新发布的高性能模型",
        "create_time": "2025-08-10T10:00:00Z",
        "update_time": "2025-08-10T10:00:00Z"
      }
    ]
  }
  ```

### 2. 创建新的模型配置

- **POST** `/api/v1/model_setting/`
- **摘要**: 创建一个新的模型市场配置（通常由管理员操作）。
- **请求体**:
  ```json
  {
    "user_id": 1,
    "model_name": "Claude 3 Opus",
    "model_type": "anthropic",
    "model_url": "https://api.anthropic.com/v1",
    "model_params": "{}",
    "model_desc": "Anthropic公司的旗舰模型"
  }
  ```
- **成功响应 (200)**: 返回新创建的模型市场配置对象。

### 3. 更新模型配置

- **PUT** `/api/v1/model_setting/{setting_id}`
- **摘要**: 更新指定ID的模型市场配置。
- **参数**:
  - `setting_id` (路径参数, integer, 必需): 模型市场配置的唯一ID。
- **请求体**: (所有字段均为可选)
  ```json
  {
    "model_desc": "Anthropic公司最强大的旗舰模型"
  }
  ```
- **成功响应 (200)**: 返回更新后的模型市场配置对象。

### 4. 删除模型配置

- **DELETE** `/api/v1/model_setting/{setting_id}`
- **摘要**: 删除指定ID的模型市场配置。
- **参数**:
  - `setting_id` (路径参数, integer, 必需): 模型市场配置的唯一ID。
- **成功响应 (200)**:
  ```json
  {
    "code": 200,
    "message": "请求成功",
    "data": {
      "message": "删除成功"
    }
  }
  ```

---

## 用户MCP配置接口 (`/api/v1/user_mcp`)

管理用户的多模型协作与规划（Multi-Agent Collaboration & Planning）配置。

### 1. 获取当前用户的所有MCP配置

- **GET** `/api/v1/user_mcp/`
- **摘要**: 获取当前认证用户的所有MCP配置。
- **成功响应 (200)**:
  ```json
  {
    "code": 200,
    "message": "请求成功",
    "data": [
      {
        "id": 1,
        "user_id": 123,
        "mcp_setting_json": {
          "planner_model": "gemini-1.5-pro",
          "executor_model": "gpt-4o",
          "max_iterations": 5
        },
        "create_time": "2025-08-10T10:00:00Z",
        "update_time": "2025-08-10T10:00:00Z"
      }
    ]
  }
  ```

### 2. 创建新的用户MCP配置

- **POST** `/api/v1/user_mcp/`
- **摘要**: 为当前认证的用户创建一个新的MCP配置。
- **请求体**:
  ```json
  {
    "user_id": 123,
    "mcp_setting_json": {
      "planner_model": "claude-3-opus",
      "executor_model": "gemini-1.5-flash"
    }
  }
  ```
- **成功响应 (200)**: 返回新创建的MCP配置对象。

### 3. 更新用户MCP配置

- **PUT** `/api/v1/user_mcp/{mcp_id}`
- **摘要**: 更新指定ID的用户MCP配置。
- **参数**:
  - `mcp_id` (路径参数, integer, 必需): MCP配置记录的唯一ID。
- **请求体**: (所有字段均为可选)
  ```json
  {
    "mcp_setting_json": {
      "max_iterations": 10
    }
  }
  ```
- **成功响应 (200)**: 返回更新后的MCP配置对象。

### 4. 删除用户MCP配置

- **DELETE** `/api/v1/user_mcp/{mcp_id}`
- **摘要**: 删除指定ID的用户MCP配置。
- **参数**:
  - `mcp_id` (路径参数, integer, 必需): MCP配置记录的唯一ID。
- **成功响应 (200)**:
  ```json
  {
    "code": 200,
    "message": "请求成功",
    "data": {
      "message": "删除成功"
    }
  }
  ```
