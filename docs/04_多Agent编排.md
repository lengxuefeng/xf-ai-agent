# 04 多 Agent 编排

## 1. 为什么要多 Agent

多 Agent 的好处不是"看起来高级"，而是为了把职责拆清楚。

例如：

- 普通聊天归 `ChatAgent`
- 代码问题归 `code_agent`
- 搜索问题归 `search_agent`
- SQL 查询归 `sql_agent`
- 天气查询归 `weather_agent`
- 医疗问题归 `medical_agent`
- 云游问题归 `yunyou_agent`

## 2. 编排核心架构

### 2.1 Supervisor 编排图

**位置**: `app/agent/graphs/supervisor.py`

Supervisor 是整个多 Agent 系统的核心编排器，负责：

- 接收用户请求
- 判断问题类型和复杂度
- 决定调用哪个专业 Agent
- 协调多 Agent 协作
- 汇总结果并回禀

### 2.2 专业 Agent

**位置**: `app/agent/agents/`

当前系统支持的专业 Agent：

- **CodeAgent**: 代码生成和执行
- **SearchAgent**: 网络搜索
- **SQLAgent**: 数据库查询
- **WeatherAgent**: 天气查询
- **MedicalAgent**: 医疗咨询
- **YunyouAgent**: 云游相关业务

### 2.3 GraphRunner 执行器

**位置**: `app/agent/graph_runner.py`

GraphRunner 是图执行器，负责：

- 驱动 Supervisor 图执行
- 使用生产者-消费者异步模型
- 产生 SSE 流式响应
- 支持前置规则拦截
- 支持审批中断和恢复
- 管理运行时上下文

### 2.4 前置规则拦截

**位置**: `app/agent/rules/`

前置规则拦截引擎（Zero-LLM）负责：

- 快速响应简单问题
- 避免不必要的图执行
- 支持多规则复合拦截
- 意图覆盖率防爆盾

### 2.5 Agent 注册

**位置**: `app/agent/registry.py`

Agent 注册机制负责：

- 统一的 Agent 注册
- Agent 能力描述
- Agent 生命周期管理

## 3. 执行流程

### 3.1 掌柜直办

适用于：

- 简单问答
- 普通咨询
- 不需要工具或外部系统的场景

**执行路径**:

```text
用户请求
-> GraphRunner
-> 前置规则拦截（可选）
-> Supervisor 判断（简单直答）
-> ChatAgent 直接回复
-> SSE 流式返回
```

### 3.2 司员执行

适用于：

- 需要专业工具
- 需要联网
- 需要代码执行
- 需要多步骤完成

**执行路径**:

```text
用户请求
-> GraphRunner
-> Supervisor 判断（需要专业能力）
-> 调度到对应的专业 Agent
-> Agent 调用 Runtime 能力
-> 执行任务
-> 返回结果
-> Supervisor 汇总
-> SSE 流式返回
```

### 3.3 多 Agent 协作

适用于：

- 复杂任务需要多个 Agent 协作
- 需要多步骤完成
- 需要结果汇总

**执行路径**:

```text
用户请求
-> GraphRunner
-> Supervisor 拆分任务
-> 调度到多个专业 Agent
-> Agent 并行或串行执行
-> Supervisor 汇总结果
-> SSE 流式返回
```

## 4. 技术实现

### 4.1 LangGraph 编排

- 使用 LangGraph 构建编排图
- 支持 StateGraph 和 MessageGraph
- 支持条件边和循环边
- 支持子图嵌套

### 4.2 流式执行

- GraphRunner 使用生产者-消费者模型
- 图执行在后台线程
- 主线程消费队列推 SSE
- 全程不阻塞事件循环

### 4.3 中断机制

- 支持 LangGraph Interrupt
- 支持审批挂起
- 支持状态恢复
- 完整的审批工作流

### 4.4 Session 管理

- LangGraph Checkpointer
- 会话状态持久化
- 跨请求状态恢复
- Session Pool 预热机制

## 5. 与 Runtime 的集成

### 5.1 RunContext 集成

- GraphRunner 创建 RunContext
- Agent 访问 RunContext
- Runtime 使用 RunContext

### 5.2 工具集成

- Agent 使用 Runtime Tool Registry
- 统一的工具调用接口
- 工具执行结果处理

### 5.3 记忆集成

- Agent 访问 Runtime Memory
- 记忆注入到 Agent 上下文
- 记忆更新和持久化

### 5.4 Hooks 集成

- Agent 触发 Runtime Hooks
- 审批和质量检查
- 事件追踪

## 6. 前端展示

### 6.1 流程卷轴

前端流程卷轴展示：

- 老板发令
- 掌柜分派
- 司员执行
- 总管回禀

### 6.2 事件映射

Workflow 事件映射到前端展示：

- `user_message_received`: 老板下达任务
- `rule_intercepted`: 总管快速裁决
- `agent_started`: 司员开始执行
- `agent_completed`: 司员执行完成
- `final_report_delivered`: 总管回禀老板

## 7. 当前项目的优势

这套编排非常适合产品形态，因为：

1. **清晰的职责分工**: 每个 Agent 专注自己的领域
2. **可扩展性强**: 容易添加新的专业 Agent
3. **可观测性好**: 流程卷轴清晰展示执行过程
4. **审批安全**: 高风险操作支持审批
5. **流式体验**: 实时流式返回结果
6. **性能优化**: 前置规则拦截、Session Pool 预热

