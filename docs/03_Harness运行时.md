# 03 Harness 运行时

## 1. 什么是 Harness

在这个项目里，Harness 指的是“模型之外的一整套执行底座”。

也就是：

- 文件系统
- Bash / Python 执行
- 记忆
- Web Search / MCP / Tool Registry
- 上下文工程
- Hooks / 审批 / 观测

## 2. 六大模块

### 2.1 文件系统 Workspace

作用：

- 为每个会话、每次运行准备工作区
- 保存中间产物
- 保存最终响应
- 记录 workflow trace

### 2.2 Bash + Sandbox / Exec Runtime

作用：

- 受控执行命令或代码
- 限制命令白名单
- 限制工作目录
- 返回 stdout / stderr / exit code

### 2.3 Memory / AGENTS.md

作用：

- 注入项目记忆
- 注入协作规则
- 注入用户偏好

### 2.4 Web Search + MCP + Tool Registry

作用：

- 联网搜索
- MCP 接入
- 外部工具统一注册和调用

### 2.5 Context Engineering

作用：

- 裁剪历史
- 注入记忆
- 注入 RAG 片段
- 控制 token 预算

### 2.6 Hooks / Approval / Observability

作用：

- 执行前审批
- 中断恢复
- 质量门禁
- 事件追踪

## 3. 为什么这层很重要

如果没有 Harness，模型即使会“推理”，也很难稳定完成真实任务。

Harness 解决的是“模型如何安全、稳定、可重复地工作”。

## 4. 当前项目的落地方式

当前 Runtime 主要集中在 `app/runtime/` 下，并通过 `graph_runner` 接到 Supervisor 主流程里。

## 5. 目录约束执行

页面终端和执行链路的关键规则是：

1. 用户先指定工作目录
2. 后端校验该目录是否合法
3. 命令只能在该目录及其子目录执行
4. 不允许越过边界

这决定了页面终端能否真正安全上生产。

