# 08 API 接口说明

## 1. 对话相关

### 1.1 流式聊天

**接口**: `POST /api/v1/chat/stream`

**主要作用**:

- 发起普通聊天
- 发起多 Agent 执行
- 发起审批恢复

**请求参数**:

```json
{
  "user_input": "用户输入内容",
  "session_id": "会话 ID",
  "workspace_root": "工作目录（可选）",
  "resume_message_id": "恢复消息 ID（可选，用于审批恢复）",
  "model_config": {
    "model_name": "模型名称（可选）",
    "model_config": "模型配置（可选）"
  },
  "session_context": {
    "context_slots": {
      "city": "城市",
      "name": "用户名"
    },
    "context_summary": "上下文摘要"
  }
}
```

**响应格式**: Server-Sent Events (SSE)

**事件类型**:

- `response_start`: 响应开始
- `thinking`: 思考提示
- `stream`: 流式内容
- `workflow`: 工作流事件
- `log`: 日志
- `interrupt`: 中断事件
- `error`: 错误
- `response_end`: 响应结束

**Workflow 事件示例**:

```json
{
  "event": "workflow",
  "data": {
    "session_id": "会话 ID",
    "run_id": "运行 ID",
    "phase": "user_message_received",
    "title": "老板下达任务",
    "summary": "用户输入的内容",
    "status": "completed",
    "role": "boss",
    "meta": {
      "input_length": 100
    }
  }
}
```

**实现位置**: `app/api/v1/chat_api.py`, `app/services/chat_service.py`

### 1.2 获取模型配置

**接口**: `GET /api/v1/user-model`

**主要作用**: 获取用户模型配置列表

**响应示例**:

```json
{
  "code": 0,
  "message": "success",
  "data": [
    {
      "id": 1,
      "user_id": 1,
      "model_name": "模型名称",
      "model_config": {
        "model_name": "模型名称",
        "temperature": 0.7
      },
      "is_default": true,
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

**实现位置**: `app/api/v1/user_model_api.py`, `app/services/user_model_service.py`

### 1.3 创建模型配置

**接口**: `POST /api/v1/user-model`

**主要作用**: 创建用户模型配置

**请求参数**:

```json
{
  "model_name": "模型名称",
  "model_config": {
    "model_name": "模型名称",
    "temperature": 0.7
  },
  "is_default": false
}
```

**实现位置**: `app/api/v1/user_model_api.py`, `app/services/user_model_service.py`

### 1.4 更新模型配置

**接口**: `PUT /api/v1/user-model/{id}`

**主要作用**: 更新用户模型配置

**请求参数**: 同创建模型配置

**实现位置**: `app/api/v1/user_model_api.py`, `app/services/user_model_service.py`

### 1.5 删除模型配置

**接口**: `DELETE /api/v1/user-model/{id}`

**主要作用**: 删除用户模型配置

**实现位置**: `app/api/v1/user_model_api.py`, `app/services/user_model_service.py`

## 2. 审批相关

### 2.1 提交审批结果

**接口**: `POST /api/v1/interrupt/approve`

**主要作用**: 提交审批结果（批准或拒绝）

**请求参数**:

```json
{
  "session_id": "会话 ID",
  "message_id": "消息 ID",
  "decision": "approve",  // 或 "reject"
  "agent_name": "Agent 名称（可选）",
  "comment": "审批意见（可选）"
}
```

**响应示例**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "status": "approved",
    "approved_at": "2024-01-01T00:00:00Z"
  }
}
```

**实现位置**: `app/api/v1/interrupt_api.py`, `app/services/interrupt_service.py`

### 2.2 查询待审批记录

**接口**: `GET /api/v1/interrupt/pending?session_id={session_id}`

**主要作用**: 查询待审批记录

**响应示例**:

```json
{
  "code": 0,
  "message": "success",
  "data": [
    {
      "id": 1,
      "session_id": "会话 ID",
      "message_id": "消息 ID",
      "agent_name": "Agent 名称",
      "status": "pending",
      "action_name": "操作名称",
      "action_args": {
        "arg1": "value1"
      },
      "created_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

**实现位置**: `app/api/v1/interrupt_api.py`, `app/services/interrupt_service.py`

### 2.3 查询审批历史

**接口**: `GET /api/v1/interrupt/history?session_id={session_id}`

**主要作用**: 查询审批历史记录

**响应示例**: 同查询待审批记录

**实现位置**: `app/api/v1/interrupt_api.py`, `app/services/interrupt_service.py`

## 3. 页面终端相关

### 3.1 绑定工作目录

**接口**: `POST /api/v1/terminal/workspace/bind`

**主要作用**: 绑定会话的工作目录

**请求参数**:

```json
{
  "session_id": "会话 ID",
  "workspace_root": "工作目录绝对路径"
}
```

**响应示例**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "session_id": "会话 ID",
    "workspace_root": "工作目录绝对路径",
    "workspace_id": "工作目录 ID",
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

**实现位置**: `app/api/v1/terminal_api.py`

**安全约束**:

- 工作目录必须在允许的根目录内
- 目录必须存在且可访问
- Path Guard 验证

### 3.2 查询当前工作目录

**接口**: `GET /api/v1/terminal/workspace/{session_id}`

**主要作用**: 查询当前会话绑定的工作目录

**响应示例**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "session_id": "会话 ID",
    "workspace_root": "工作目录绝对路径",
    "workspace_id": "工作目录 ID",
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

**实现位置**: `app/api/v1/terminal_api.py`

### 3.3 列出工作目录内容

**接口**: `GET /api/v1/terminal/fs/tree/{session_id}?path={relative_path}`

**主要作用**: 列出工作目录下的文件和目录

**请求参数**:

- `session_id`: 会话 ID
- `path`: 相对路径（可选，默认为工作目录根）

**响应示例**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "path": "相对路径",
    "entries": [
      {
        "name": "文件名或目录名",
        "type": "file",  // 或 "directory"
        "size": 1024,
        "modified_at": "2024-01-01T00:00:00Z"
      }
    ]
  }
}
```

**实现位置**: `app/api/v1/terminal_api.py`

### 3.4 读取工作目录文件

**接口**: `GET /api/v1/terminal/fs/file/{session_id}?path={relative_path}`

**主要作用**: 读取工作目录下的文件内容

**请求参数**:

- `session_id`: 会话 ID
- `path`: 相对路径

**响应示例**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "path": "相对路径",
    "content": "文件内容",
    "size": 1024,
    "modified_at": "2024-01-01T00:00:00Z"
  }
}
```

**实现位置**: `app/api/v1/terminal_api.py`

### 3.5 保存工作目录文件

**接口**: `PUT /api/v1/terminal/fs/file`

**主要作用**: 保存或更新工作目录下的文件

**请求参数**:

```json
{
  "session_id": "会话 ID",
  "path": "相对路径",
  "content": "文件内容"
}
```

**响应示例**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "path": "相对路径",
    "size": 1024,
    "modified_at": "2024-01-01T00:00:00Z"
  }
}
```

**实现位置**: `app/api/v1/terminal_api.py`

### 3.6 创建工作目录文件

**接口**: `POST /api/v1/terminal/fs/file`

**主要作用**: 在工作目录下创建新文件

**请求参数**:

```json
{
  "session_id": "会话 ID",
  "path": "相对路径",
  "content": "文件内容"
}
```

**响应示例**: 同保存工作目录文件

**实现位置**: `app/api/v1/terminal_api.py`

### 3.7 创建工作目录目录

**接口**: `POST /api/v1/terminal/fs/directory`

**主要作用**: 在工作目录下创建新目录

**请求参数**:

```json
{
  "session_id": "会话 ID",
  "path": "相对路径"
}
```

**响应示例**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "path": "相对路径",
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

**实现位置**: `app/api/v1/terminal_api.py`

### 3.8 重命名工作目录条目

**接口**: `PUT /api/v1/terminal/fs/entry/rename`

**主要作用**: 重命名工作目录下的文件或目录

**请求参数**:

```json
{
  "session_id": "会话 ID",
  "old_path": "旧相对路径",
  "new_name": "新名称"
}
```

**响应示例**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "old_path": "旧相对路径",
    "new_path": "新相对路径"
  }
}
```

**实现位置**: `app/api/v1/terminal_api.py`

### 3.9 删除工作目录条目

**接口**: `POST /api/v1/terminal/fs/entry/delete`

**主要作用**: 删除工作目录下的文件或目录

**请求参数**:

```json
{
  "session_id": "会话 ID",
  "path": "相对路径"
}
```

**响应示例**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "path": "相对路径"
  }
}
```

**实现位置**: `app/api/v1/terminal_api.py`

### 3.10 启动命令

**接口**: `POST /api/v1/terminal/session/start`

**主要作用**: 在工作目录下启动命令执行

**请求参数**:

```json
{
  "session_id": "会话 ID",
  "workspace_root": "工作目录绝对路径",
  "command_text": "命令文本",
  "cwd": "当前工作目录（相对路径，可选）"
}
```

**说明**:

- `workspace_root` 是用户绑定的工作目录
- `cwd` 是相对于 `workspace_root` 的子目录
- 命令最终运行在后端服务所在机器上，而不是浏览器中
- 命令只允许在当前工作目录及其子目录内执行

**响应示例**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "command_id": "命令 ID",
    "status": "running",
    "started_at": "2024-01-01T00:00:00Z"
  }
}
```

**实现位置**: `app/api/v1/terminal_api.py`, `app/runtime/exec/command_session_service.py`

### 3.11 查询命令状态

**接口**: `GET /api/v1/terminal/session/{command_id}`

**主要作用**: 查询命令执行状态

**响应示例**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "command_id": "命令 ID",
    "status": "completed",  // running/completed/failed/cancelled
    "exit_code": 0,
    "stdout": "标准输出",
    "stderr": "错误输出",
    "started_at": "2024-01-01T00:00:00Z",
    "completed_at": "2024-01-01T00:00:10Z"
  }
}
```

**实现位置**: `app/api/v1/terminal_api.py`, `app/runtime/exec/command_session_service.py`

### 3.12 查询会话最近命令

**接口**: `GET /api/v1/terminal/session/latest/{session_id}`

**主要作用**: 查询会话最近执行的命令

**响应示例**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "command_id": "命令 ID",
    "command_text": "命令文本",
    "status": "completed",
    "exit_code": 0,
    "started_at": "2024-01-01T00:00:00Z",
    "completed_at": "2024-01-01T00:00:10Z"
  }
}
```

**实现位置**: `app/api/v1/terminal_api.py`, `app/runtime/exec/command_session_service.py`

### 3.13 停止命令

**接口**: `POST /api/v1/terminal/session/{command_id}/stop`

**主要作用**: 停止正在运行的命令

**响应示例**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "command_id": "命令 ID",
    "status": "cancelled",
    "stopped_at": "2024-01-01T00:00:10Z"
  }
}
```

**实现位置**: `app/api/v1/terminal_api.py`, `app/runtime/exec/command_session_service.py`

## 4. 健康检查

### 4.1 Runtime 健康

**接口**: `GET /api/v1/health/runtime-harness`

**主要作用**: 查看 Runtime 子系统状态

**主要用于查看**:

- Tool Registry 状态
- Workspace 状态
- Terminal Runtime 状态
- Memory 状态
- Hooks 状态
- 其他运行时子系统状态

**响应示例**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "status": "healthy",
    "subsystems": {
      "tool_registry": {
        "status": "healthy",
        "stats": {
          "total": 10,
          "enabled": 8
        }
      },
      "workspace": {
        "status": "healthy",
        "active_sessions": 5
      },
      "terminal_runtime": {
        "status": "healthy",
        "active_commands": 2
      }
    }
  }
}
```

**实现位置**: `app/api/v1/health_api.py`

## 5. 聊天历史

### 5.1 会话列表

**接口**: `GET /api/v1/chat-history/sessions`

**主要作用**: 获取用户的聊天会话列表

**查询参数**:

- `page`: 页码（可选，默认 1）
- `page_size`: 每页大小（可选，默认 20）

**响应示例**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "total": 100,
    "page": 1,
    "page_size": 20,
    "sessions": [
      {
        "session_id": "会话 ID",
        "title": "会话标题",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z"
      }
    ]
  }
}
```

**实现位置**: `app/api/v1/chat_history_api.py`, `app/services/chat_history_service.py`

### 5.2 指定会话消息

**接口**: `GET /api/v1/chat-history/sessions/{session_id}/messages`

**主要作用**: 获取指定会话的消息历史

**查询参数**:

- `page`: 页码（可选，默认 1）
- `page_size`: 每页大小（可选，默认 20）

**响应示例**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "total": 50,
    "page": 1,
    "page_size": 20,
    "messages": [
      {
        "id": 1,
        "session_id": "会话 ID",
        "user_content": "用户内容",
        "model_content": "模型内容",
        "extra_data": {
          "thinking_trace": "思考轨迹",
          "workflow_trace": [],
          "runtime_snapshot": {},
          "runtime_artifacts": []
        },
        "created_at": "2024-01-01T00:00:00Z"
      }
    ]
  }
}
```

**实现位置**: `app/api/v1/chat_history_api.py`, `app/services/chat_history_service.py`

## 6. 用户信息

### 6.1 获取用户信息

**接口**: `GET /api/v1/user-info`

**主要作用**: 获取当前用户信息

**响应示例**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": 1,
    "user_name": "用户名",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z"
  }
}
```

**实现位置**: `app/api/v1/user_info_api.py`, `app/services/user_info_service.py`

## 7. MCP 配置

### 7.1 获取 MCP 配置列表

**接口**: `GET /api/v1/user-mcp`

**主要作用**: 获取用户的 MCP 配置列表

**响应示例**:

```json
{
  "code": 0,
  "message": "success",
  "data": [
    {
      "id": 1,
      "user_id": 1,
      "mcp_name": "MCP 名称",
      "mcp_config": {
        "endpoint": "MCP 端点",
        "token": "访问令牌"
      },
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

**实现位置**: `app/api/v1/user_mcp_api.py`, `app/services/user_mcp_service.py`

### 7.2 创建 MCP 配置

**接口**: `POST /api/v1/user-mcp`

**主要作用**: 创建用户 MCP 配置

**请求参数**:

```json
{
  "mcp_name": "MCP 名称",
  "mcp_config": {
    "endpoint": "MCP 端点",
    "token": "访问令牌"
  }
}
```

**实现位置**: `app/api/v1/user_mcp_api.py`, `app/services/user_mcp_service.py`

### 7.3 更新 MCP 配置

**接口**: `PUT /api/v1/user-mcp/{id}`

**主要作用**: 更新用户 MCP 配置

**请求参数**: 同创建 MCP 配置

**实现位置**: `app/api/v1/user_mcp_api.py`, `app/services/user_mcp_service.py`

### 7.4 删除 MCP 配置

**接口**: `DELETE /api/v1/user-mcp/{id}`

**主要作用**: 删除用户 MCP 配置

**实现位置**: `app/api/v1/user_mcp_api.py`, `app/services/user_mcp_service.py`

## 8. SSE 事件类型

### 8.1 事件列表

| 事件类型 | 说明 | 数据格式 |
|---------|------|----------|
| `response_start` | 响应开始 | 空字符串 |
| `thinking` | 思考提示 | 思考文本 |
| `stream` | 流式内容 | 内容文本 |
| `workflow` | 工作流事件 | JSON 对象 |
| `log` | 日志 | 日志文本 |
| `interrupt` | 中断事件 | JSON 对象 |
| `error` | 错误 | 错误信息 |
| `response_end` | 响应结束 | 空字符串 |

### 8.2 Workflow 事件 Phases

| Phase | 标题 | 角色 |
|-------|------|------|
| `user_message_received` | 老板下达任务 | Boss |
| `rule_intercepted` | 总管快速裁决 | Supervisor |
| `workspace_prepared` | 卷宗工位已就绪 | System |
| `memory_loaded` | 中枢调入项目与会话记忆 | System |
| `context_ready` | 上下文卷宗整理完成 | System |
| `tool_registry_ready` | 运行时工具与外部能力已装载 | System |
| `artifacts_indexed` | 运行时卷宗已入册 | System |
| `agent_started` | 司员开始执行 | Worker |
| `agent_completed` | 司员执行完成 | Worker |
| `approval_pending` | 等待老板批示 | System |
| `final_report_delivered` | 总管回禀老板 | Supervisor |

## 9. 错误码

| 错误码 | 说明 |
|--------|------|
| 0 | 成功 |
| 400 | 请求参数错误 |
| 401 | 未授权 |
| 403 | 禁止访问 |
| 404 | 资源不存在 |
| 500 | 服务器错误 |
| 1001 | 工作目录不存在 |
| 1002 | 工作目录越界 |
| 1003 | 命令执行失败 |
| 1004 | 审批记录不存在 |
| 1005 | 会话不存在 |

## 10. 阅读建议

如果你准备联调前后端，请优先理解三条主线：

1. **`chat/stream`**: 核心流式聊天接口
2. **`interrupt/approve`**: 审批恢复接口
3. **`terminal/*`**: 页面终端接口

这三条主线覆盖了整个系统的核心功能。
