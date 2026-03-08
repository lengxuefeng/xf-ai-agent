# 云柚 Agent (YunyouAgent) 生产级业务流程说明

本文档详细描述了 `xf-ai-agent` 后端系统中，基于 LangGraph 重构后的 `yunyou_agent` 子图的工作流程。该流程集成了自动路由、状态管理、以及针对敏感操作的“人机回环”（Human-in-the-Loop）审批机制，达到了生产环境的可用标准。

## 1. 系统架构概述

系统采用 **Supervisor-Worker (多智能体)** 架构：

*   **父图 (Supervisor)**: 充当“项目经理”，负责解析用户意图，将任务分发给最合适的子智能体（如 `yunyou_agent`, `code_agent` 等）。
*   **子图 (Subagent)**: 充当“领域专家”，拥有独立的 StateGraph、工具集和内部审批逻辑。

## 2. 核心前提：生产环境配置

在生产环境中，这套系统的稳定运行依赖于以下基石：

1.  **持久化 (Redis Checkpointer)**
    *   **作用**: 必须开启。用户的每一次交互、图在每个节点的状态快照（State Snapshot）都会实时保存在 Redis 中。
    *   **目的**: 实现“暂停-恢复”的关键。即使服务重启，也能从中断点继续执行。

2.  **会话绑定 (Session ID)**
    *   **作用**: 前端生成的 `session_id` 必须在整个对话周期内保持不变。
    *   **目的**: 后端依靠此 ID 从 Redis 中找回“暂停前”的图状态。

---

## 3. 业务流程详解

### 场景一：自动查询模式（无需人工干预）

**用户意图**: 查询非敏感的公开数据或普通统计数据。
**示例指令**: “帮我看看昨天的 Holter 类型统计。”

#### 详细步骤:

1.  **用户输入**
    *   前端发送 `/chat/stream` 请求。
    *   参数: `content: "帮我看看昨天的 Holter 类型统计"`, `session_id: "sess_123"`

2.  **路由分发 (Supervisor)**
    *   主图（Supervisor）分析用户意图，识别关键词 “Holter类型”。
    *   **决策**: 将任务路由给 `yunyou_agent` 子图。

3.  **子图执行 (YunyouAgent)**
    *   **Agent 节点**: 模型思考，决定调用 `holter_type_count` 工具。
    *   **Human Review 节点**: 拦截检查。发现 `holter_type_count` **不在**敏感工具列表中。
    *   **动作**: 放行，状态流转至 `tools` 节点。

4.  **工具执行**
    *   后端自动请求云柚 API 获取数据。
    *   工具返回 JSON 格式的统计结果。

5.  **结果生成**
    *   **Agent 节点**: 接收工具结果，生成自然语言回复（“昨天的统计结果如下：24小时项 15 个，48小时项 3 个...”）。
    *   **Supervisor**: 接收子图完成信号，将最终回复流式传输给前端。

---

### 场景二：敏感操作审批模式（Human-in-the-Loop）

**用户意图**: 查询敏感业务数据（如详细报告），或执行写操作（如修改配置、删除数据）。
**示例指令**: “查询一下上个月的 Holter 报告统计数据。”（注：此操作被定义为敏感操作）

#### 阶段 1：触发与暂停

1.  **用户输入**
    *   用户发送请求: "查询上个月报告统计"。

2.  **路由分发**
    *   Supervisor -> `yunyou_agent`。

3.  **子图执行 (YunyouAgent)**
    *   **Agent 节点**: 模型思考，决定调用 `holter_report_count` 工具。
    *   **Human Review 节点**: 拦截检查。发现 `holter_report_count` **命中**敏感工具列表！
    *   **触发中断**: 调用 `interrupt()` 函数。
    *   **保存状态**: LangGraph 将当前图的执行指针挂起，并将上下文保存至 Redis。

4.  **前端响应**
    *   后端通过 SSE 推送 `type: "interrupt"` 事件，包含工具名称和参数。
    *   前端解析事件，在界面渲染 **“操作确认卡片”**：
        *   🛑 **操作请求**: 查询Holter报告统计
        *   📅 **参数**: `start=2023-10-01, end=2023-10-31`
        *   🔘 **[批准执行]** | 🔘 **[拒绝]**

#### 阶段 2：人工决策与恢复

**分支 A：用户点击 [批准执行]**

1.  **前端请求**:
    *   发送一个新的聊天请求（或专用 resume 接口），携带 `command: "approve"`（或在后端逻辑中通过 `session_id` 自动关联批准动作）。

2.  **后端恢复**:
    *   `BaseAgent` 根据 `session_id` 从 Redis 加载图状态。
    *   发现图处于 `interrupt` 状态，将 "approve" 作为输入传给 `human_review` 节点。

3.  **继续执行**:
    *   **Human Review 节点**: 收到 "approve"，逻辑判断通过 -> 路由到 `tools` 节点。
    *   **工具执行**: 真正调用 `holter_report_count` API。
    *   **结果生成**: Agent 拿到数据：“已为您查询，结果是...”。

**分支 B：用户点击 [拒绝]**

1.  **前端请求**:
    *   发送拒绝指令 `command: "reject"`。

2.  **后端恢复**:
    *   `human_review` 节点收到 "reject"。
    *   **拦截执行**: 状态流转直接跳过 `tools` 节点，或者流转到结束。
    *   **反馈模型**: 向 LLM 注入一条 `ToolMessage`：“User rejected request”。
    *   **生成回复**: Agent 回复：“好的，已取消该查询操作。”

## 4. 技术实现细节

*   **文件路径**: `app/agent/agents/yunyou_agent.py`
*   **图类型**: `StateGraph` (LangGraph 1.0+)
*   **关键节点**:
    *   `agent`: 调用 LLM 绑定工具。
    *   `human_review`: 自定义函数节点，包含 `interrupt()` 逻辑。
    *   `tools`: `ToolNode`，执行实际 API 调用。
*   **中断机制**: 使用 `langgraph.types.interrupt` 实现原生图中断。

## 5. 前后端交互协议 (SSE)

后端通过 Server-Sent Events (SSE) 向前端推送状态：

*   `type: "thinking"`: Agent 正在思考或路由中。
*   `type: "interrupt"`: **关键**。表示需要人工介入。
    *   `content`: 包含需要审核的工具列表 JSON。
*   `type: "stream"`: 普通的文本流式回复。
*   `type: "error"`: 系统异常信息。

---
*文档生成时间: 2026-02-09*
