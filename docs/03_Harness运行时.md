# 03 Harness 运行时

## 1. 什么是 Harness

在这个项目里，Harness 指的是"模型之外的一整套执行底座"。

也就是：

- 文件系统
- Bash / Python 执行
- 记忆
- Web Search / MCP / Tool Registry
- 上下文工程
- Hooks / 审批 / 观测

## 2. 七大模块

### 2.1 Runtime 核心

**位置**: `app/runtime/core/`

**核心组件**:

- **RunContext**: 单次运行的统一上下文管理
  - session_id: 会话标识
  - run_id: 运行标识
  - user_input: 用户输入
  - model_config: 模型配置
  - is_resume: 是否为恢复操作
  - meta: 扩展元数据

- **SessionManager**: 运行时会话管理和状态追踪
  - 注册运行会话
  - 更新运行状态
  - 记录工作流事件
  - 清理会话资源

- **WorkflowEventBus**: 工作流事件总线和持久化
  - 事件发布和订阅
  - 事件格式化
  - 事件持久化
  - 事件回放

- **LiveStreamBus**: 子 Agent 实时流事件总线
  - 子 Agent 实时输出流
  - 流事件分发
  - 跨会话隔离

### 2.2 文件系统 Workspace

**位置**: `app/runtime/workspace/`

**作用**:

- 为每个会话、每次运行准备工作区
- 保存中间产物
- 保存最终响应
- 记录 workflow trace

**核心组件**:

- **WorkspaceManager**: 工作区管理器
  - 工作区创建和管理
  - 工作区权限控制
  - 工作区清理

- **PathGuard**: 路径安全守卫
  - 目录边界检查
  - 路径规范化
  - 越界防护

### 2.3 Bash + Sandbox / Exec Runtime

**位置**: `app/runtime/exec/`

**作用**:

- 受控执行命令或代码
- 限制命令白名单
- 限制工作目录
- 返回 stdout / stderr / exit code

**核心组件**:

- **Runner**: 命令执行器
  - Bash 命令执行
  - Python 代码执行
  - 超时控制
  - 结果格式化

- **CommandSessionService**: 命令会话服务
  - 会话管理
  - 会话恢复
  - 输出流管理

- **Policy**: 执行策略
  - 白名单检查
  - 目录限制
  - 命令过滤

### 2.4 Memory / AGENTS.md

**位置**: `app/runtime/memory/`

**作用**:

- 注入项目记忆
- 注入协作规则
- 注入用户偏好

**核心组件**:

- **MemoryService**: 记忆服务
  - 记忆查询
  - 记忆注入
  - 记忆更新

- **AgentsMemoryLoader**: AGENTS.md 加载器
  - 文件解析
  - 内容提取
  - 结构化存储

### 2.5 Web Search + MCP + Tool Registry

**位置**: `app/runtime/tools/`

**作用**:

- 联网搜索
- MCP 接入
- 外部工具统一注册和调用

**核心组件**:

- **ToolRegistry**: 工具注册表
  - 工具注册
  - 工具查询
  - 工具统计

- **ToolExecutor**: 工具执行器
  - 工具调用
  - 结果处理
  - 错误处理

- **SearchGateway**: 搜索网关
  - 搜索能力封装
  - 搜索结果处理

- **McpGateway**: MCP 网关
  - MCP 协议支持
  - MCP 工具集成

### 2.6 Context Engineering

**位置**: `app/runtime/context/`

**作用**:

- 裁剪历史
- 注入记忆
- 注入 RAG 片段
- 控制 token 预算

**核心组件**:

- **ContextBuilder**: 上下文构建器
  - 消息构建
  - 记忆注入
  - RAG 注入

- **ContextBudget**: 上下文预算管理
  - Token 预算控制
  - 内容压缩
  - 优先级管理

- **ContextInjectors**: 上下文注入器
  - 会话上下文注入
  - 记忆片段注入
  - RAG 片段注入

### 2.7 Hooks / Approval / Observability

**位置**: `app/runtime/hooks/`

**作用**:

- 执行前审批
- 中断恢复
- 质量门禁
- 事件追踪

**核心组件**:

- **HookManager**: Hook 管理器
  - Hook 注册
  - Hook 执行
  - Hook 结果聚合

- **内置 Hooks**:
  - **ApprovalHook**: 审批 Hook
  - **QualityHook**: 质量门禁
  - **ToolGuardHook**: 工具守卫

## 3. Runtime 工作流程

```text
1. 用户请求到达
   -> GraphRunner 创建 RunContext

2. SessionManager 注册运行会话
   -> 分配 run_id
   -> 初始化运行状态

3. WorkspaceManager 准备工作区
   -> 创建会话工作目录
   -> 准备运行卷宗

4. ContextBuilder 构建上下文
   -> 裁剪历史
   -> 注入记忆
   -> 注入 RAG 片段

5. HookManager 执行前置 Hooks
   -> 审批检查
   -> 质量门禁
   -> 工具守卫

6. 图执行开始
   -> WorkflowEventBus 记录事件
   -> Runtime 能力调用

7. Runtime 能力执行
   -> Workspace 文件操作
   -> Exec Runtime 命令执行
   -> Tool Registry 工具调用

8. HookManager 执行后置 Hooks
   -> 结果审核
   -> 事件追踪

9. SessionManager 更新状态
   -> 标记完成/失败
   -> 持久化结果

10. WorkspaceManager 清理资源
    -> 保留必要的中间产物
    -> 清理临时文件
```

## 4. 为什么这层很重要

如果没有 Harness，模型即使会"推理"，也很难稳定完成真实任务。

Harness 解决的是"模型如何安全、稳定、可重复地工作"。

### 4.1 安全性

- Path Guard 防止越界执行
- 命令白名单防止恶意操作
- 审批机制防止高风险执行

### 4.2 稳定性

- 统一的错误处理
- 超时控制
- 会话管理
- 状态追踪

### 4.3 可观测性

- Workflow 事件流
- 日志流
- 状态快照
- 事件回放

### 4.4 可维护性

- 统一的接口抽象
- 模块化设计
- Hook 扩展机制
- 工具注册机制

## 5. 当前项目的落地方式

当前 Runtime 主要集中在 `app/runtime/` 下，并通过 `graph_runner` 接到 Supervisor 主流程里。

### 5.1 目录结构

```text
app/runtime/
├── core/              # 运行时核心
│   ├── run_context.py
│   ├── session_manager.py
│   ├── workflow_event_bus.py
│   └── live_stream_bus.py
├── workspace/         # 工作区管理
│   ├── manager.py
│   ├── path_guard.py
│   └── models.py
├── exec/              # 命令执行
│   ├── runner.py
│   ├── command_session_service.py
│   └── policy.py
├── memory/            # 记忆管理
│   ├── memory_service.py
│   ├── agents_memory_loader.py
│   └── memory_models.py
├── tools/             # 工具注册
│   ├── tool_registry.py
│   ├── tool_executor.py
│   ├── search_gateway.py
│   └── mcp_gateway.py
├── context/           # 上下文工程
│   ├── context_builder.py
│   ├── context_budget.py
│   └── context_injectors.py
├── hooks/             # Hooks 机制
│   ├── hook_manager.py
│   ├── builtin/
│   │   ├── approval_hook.py
│   │   ├── quality_hook.py
│   │   └── tool_guard_hook.py
│   └── base.py
└── prompts/           # 提示词管理
    └── ...
```

### 5.2 集成点

Runtime 在以下关键点集成到主流程：

1. **GraphRunner**: 使用 RunContext 和 SessionManager
2. **Agent 执行**: 使用 Workspace、Tool Registry、Context Builder
3. **工具调用**: 使用 Tool Executor
4. **命令执行**: 使用 Exec Runtime
5. **审批流程**: 使用 Hook Manager 和内置 Hooks

## 6. 目录约束执行

页面终端和执行链路的关键规则是：

1. 用户先指定工作目录
2. 后端校验该目录是否合法
3. 命令只能在该目录及其子目录执行
4. 不允许越过边界

这决定了页面终端能否真正安全上生产。

### 6.1 Path Guard 实现

- 路径规范化
- 绝对路径检查
- 符号链接处理
- 相对路径解析
- 边界验证

### 6.2 安全策略

- 白名单机制
- 目录限制
- 命令过滤
- 超时控制
- 审批检查

## 7. 性能优化

### 7.1 Session Pool 预热

- 预热 Supervisor 编译图
- 减少冷启动延迟
- 跨请求共享实例

### 7.2 异步流式处理

- 生产者-消费者模型
- 后台线程执行
- 主线程消费队列
- 非阻塞 SSE 推送

### 7.3 资源管理

- 工作区清理
- 会话回收
- 缓存管理
- 连接池

