# XF AI Agent Runtime Memory

## 项目定位
- 这是一个偏业务编排型的多 Agent 系统，不只是简单聊天机器人。
- 用户希望从前端清晰看到"老板发令 -> 掌柜调度 -> 司员执行 -> 汇总回禀"的全过程。
- 核心特点：强编排 + Harness Runtime 执行底座 + 流式SSE响应

## 架构设计
- **编排层**: LangGraph 编排的多 Agent 系统，Supervisor 负责任务分发和协调
- **运行时层**: Harness Runtime 提供文件系统、受控执行、记忆、搜索、上下文工程、Hooks 等能力
- **接口层**: FastAPI + SSE 流式响应，支持前端实时接收
- **服务层**: 业务服务层，处理会话、历史记录、审批、模型配置等

## 核心模块
- `app/agent/`: 多 Agent 编排层
  - `graphs/supervisor.py`: Supervisor 编排图
  - `agents/`: 各个专业 Agent（代码、搜索、SQL、天气等）
  - `graph_runner.py`: 图执行器，负责驱动 Supervisor 并产生 SSE 流
- `app/runtime/`: Harness Runtime 运行时
  - `core/`: 运行时核心（run_context, session_manager, workflow_event_bus）
  - `workspace/`: 文件系统和工作区管理
  - `exec/`: 命令执行和受控代码运行
  - `memory/`: 记忆管理和 AGENTS.md 注入
  - `tools/`: 工具注册和执行（MCP、搜索等）
  - `hooks/`: Hook 管理器和内置 hooks（审批、质量门禁、工具守卫）
  - `context/`: 上下文构建和注入
- `app/services/`: 业务服务层
  - `chat_service.py`: 聊天服务
  - `interrupt_service.py`: 中断和审批服务
  - `chat_history_service.py`: 聊天历史管理
- `app/api/`: FastAPI 接口层

## 运行时原则
- 优先保证可解释性、可观测性和审批安全
- 工具调用、外部查询、代码执行、记忆注入都应尽量结构化
- 能直接答复的问题由掌柜直办；需要专业能力时再调度司员
- "写代码"和"执行代码"必须区分，只有明确执行意图才进入审批与执行链路
- 页面终端是受控执行入口，所有命令必须在用户绑定的工作目录内运行
- 页面终端是前端产品形态，真正执行发生在后端运行节点，而不是浏览器本地沙箱

## 技术特点
- 异步流式处理：GraphRunner 使用生产者-消费者模型，图执行在后台线程，主线程消费队列推 SSE
- 前置规则拦截：Zero-LLM 快速响应，避免不必要的图执行
- 审批中断恢复：支持 Interrupt 挂起和审批恢复
- 运行时观测：Workflow 事件流、日志流、状态快照
- Session 管理：运行时会话管理和状态追踪
- 工具注册：统一的工具注册和执行机制

## 前端风格
- 国风简约
- 宣纸、卷轴、印章、墨色层次
- 不要堆砌科技风默认样式

## 质量要求
- 尽量保留流式体验
- 流程事件要能持久化并回放
- 高风险执行必须支持审批
- 前端需要同时支持"老板模式"和"工位模式"
