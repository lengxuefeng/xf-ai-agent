# 08 API 接口说明

## 1. 对话相关

### 流式聊天

- `POST /api/v1/chat/stream`

主要作用：

- 发起普通聊天
- 发起多 Agent 执行
- 发起审批恢复

重要参数：

- `user_input`
- `session_id`
- `workspace_root`
- `resume_message_id`

## 2. 审批相关

### 提交审批结果

- `POST /api/v1/interrupt/approve`

主要参数：

- `session_id`
- `message_id`
- `decision`

## 3. 页面终端相关

### 绑定工作目录

- `POST /api/v1/terminal/workspace/bind`

### 查询当前工作目录

- `GET /api/v1/terminal/workspace/{session_id}`

### 列出工作目录内容

- `GET /api/v1/terminal/fs/tree/{session_id}?path=`

### 读取工作目录文件

- `GET /api/v1/terminal/fs/file/{session_id}?path=`

### 保存工作目录文件

- `PUT /api/v1/terminal/fs/file`

### 创建工作目录文件

- `POST /api/v1/terminal/fs/file`

### 创建工作目录目录

- `POST /api/v1/terminal/fs/directory`

### 重命名工作目录条目

- `PUT /api/v1/terminal/fs/entry/rename`

### 删除工作目录条目

- `POST /api/v1/terminal/fs/entry/delete`

### 启动命令

- `POST /api/v1/terminal/session/start`

主要参数：

- `session_id`
- `workspace_root`
- `command_text`
- `cwd`

说明：

- `workspace_root` 是用户绑定的工作目录；
- `cwd` 是相对于 `workspace_root` 的子目录；
- 命令最终运行在后端服务所在机器上，而不是浏览器中；
- 命令只允许在当前工作目录及其子目录内执行。

### 查询命令状态

- `GET /api/v1/terminal/session/{command_id}`

### 查询会话最近命令

- `GET /api/v1/terminal/session/latest/{session_id}`

### 停止命令

- `POST /api/v1/terminal/session/{command_id}/stop`

## 4. 健康检查

### Runtime 健康

- `GET /api/v1/health/runtime-harness`

主要用于查看：

- tool registry
- workspace
- terminal runtime
- 其他运行时子系统状态

## 5. 聊天历史

### 会话列表

- `GET /api/v1/chat-history/sessions`

### 指定会话消息

- `GET /api/v1/chat-history/sessions/{session_id}/messages`

## 6. 阅读建议

如果你准备联调前后端，请优先理解三条主线：

1. `chat/stream`
2. `interrupt/approve`
3. `terminal/*`
