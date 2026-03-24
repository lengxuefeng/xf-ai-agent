# XF-AI-Agent

基于 `FastAPI + LangGraph + 多 Agent 编排 + Harness Runtime` 的生产级智能体系统。

## 1. 项目定位

XF-AI-Agent 不是单一聊天机器人，而是一套“老板视角可观测”的多 Agent 执行系统。

你可以把它理解成两层：

1. **编排层**
   负责判断用户问题应该交给谁处理、要不要拆任务、如何汇总结果。
2. **运行时层（Harness）**
   负责让 Agent 真正具备文件系统、受控执行、记忆、搜索、上下文工程、Hooks 等能力。

当前系统已经支持：

- 多 Agent 编排
- 流式 SSE 返回
- 工作流事件回放
- RAG 检索
- 审批中断与恢复
- 页面终端
- 指定目录下的受控命令执行

## 2. 适合谁阅读

如果你想通过这个项目学习以下内容，这套文档就是按这个目标整理的：

- 如何从 0 搭建多 Agent 系统
- 如何把 Agent 做成可观测、可审批、可恢复的生产级产品
- 如何为 Agent 补齐 Harness Runtime
- 如何把后端执行流程做成前端可视化卷轴
- 如何做“页面终端 + 指定目录执行”的产品化体验

## 3. 快速开始

### 3.1 环境要求

- Python 3.12+
- PostgreSQL 14+
- Node.js 18+
- 建议使用 `uv`

### 3.2 启动后端

```bash
cd xf-ai-agent
uv sync
cp .env.example .env
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3.3 启动前端

```bash
cd ../xf-ai-agent-app
npm install
npm run dev
```

## 4. 执行模型说明

### 4.1 页面终端是不是浏览器本地执行

不是。

页面终端只是**前端展示层**，真正的命令执行发生在后端进程所在的机器上：

- 你本地启动后端时：命令在你本机执行
- 你部署到服务器后：命令在服务器执行

### 4.2 为什么不弹系统终端窗口

因为当前设计是：

- 后端维护受控命令会话
- 前端通过 API 轮询/流式获取输出
- 页面自己渲染终端样式

也就是说：

- **CLI / subprocess / PTY** 是执行机制
- **页面终端** 是产品形态

### 4.3 为什么要指定目录

为了避免执行能力无限放大，系统现在要求：

1. 用户先绑定工作目录
2. 所有命令只能在这个目录或其子目录执行
3. 不允许通过相对路径、软链接等方式越界

这套约束是整个页面终端安全模型的核心。

## 5. 推荐阅读顺序

1. [docs/00_文档索引.md](./docs/00_文档索引.md)
2. [docs/01_项目总览.md](./docs/01_项目总览.md)
3. [docs/02_系统架构.md](./docs/02_系统架构.md)
4. [docs/03_Harness运行时.md](./docs/03_Harness运行时.md)
5. [docs/04_多Agent编排.md](./docs/04_多Agent编排.md)
6. [docs/05_执行与审批链路.md](./docs/05_执行与审批链路.md)
7. [docs/06_前端流程卷轴与页面终端.md](./docs/06_前端流程卷轴与页面终端.md)
8. [docs/07_数据库与会话状态.md](./docs/07_数据库与会话状态.md)
9. [docs/08_API接口说明.md](./docs/08_API接口说明.md)
10. [docs/09_测试与回归.md](./docs/09_测试与回归.md)
11. [docs/10_开发规范与中文注释规范.md](./docs/10_开发规范与中文注释规范.md)

## 6. 当前项目结构

```text
xf-ai-agent/
├── app/
│   ├── agent/          # 多 Agent 编排层
│   ├── api/            # FastAPI 接口层
│   ├── runtime/        # Harness Runtime
│   ├── services/       # 业务服务
│   ├── models/         # ORM 模型
│   ├── schemas/        # Pydantic 模型
│   └── utils/          # 通用工具
├── docs/               # 学习与运维文档
├── tests/              # 回归测试
└── AGENTS.md           # 项目运行时记忆
```

## 7. 核心能力总览

### 编排层

- `Domain Router`
- `Intent Router`
- `Parent Planner`
- `Dispatcher / Worker / Aggregator`

### Runtime 层

- Workspace 文件系统
- Bash / Python 受控执行
- Memory / AGENTS.md
- Web Search / MCP / Tool Registry
- Context Engineering
- Hooks / Approval / Observability

### 前端层

- 对话区
- 国风流程卷轴
- 运行时看板
- 页面终端

## 8. 文档治理说明

本次文档已经重构为“单入口 + 专题文档”模式：

- `README.md` 是唯一学习入口
- `docs/00_文档索引.md` 是文档导航
- 旧的重复说明文档已经移除
- 数据库专项文档保留为独立专题
- 回归报告保留在 `docs/regression/`，但不纳入主学习路径

## 9. 相关文档

- 项目运行时记忆：[AGENTS.md](./AGENTS.md)
- 数据库专项文档：[docs/PostgreSQL数据库完整技术文档.md](./docs/PostgreSQL数据库完整技术文档.md)

