# PostgreSQL 数据库完整技术文档

> 适用项目：xf-ai-agent  
> 更新时间：2026-03-08  
> 文档目标：说明本项目如何使用 PostgreSQL 承载业务数据、审批状态、数据库迁移等全部技术内容

---

## 1. PostgreSQL 在项目中的角色

### 1.1 角色总览

PostgreSQL 在本项目承担以下角色：

1. **业务主数据存储**：用户、模型配置、用户模型绑定、MCP 配置
2. **聊天持久化**：会话与消息历史（`t_chat_session` / `t_chat_message`）
3. **审批状态仓**：LangGraph `interrupt()` 审批请求与结果（`t_interrupt_approval`）
4. **SQL Agent 查询目标**：本地 SQL Agent 的数据查询目标库

### 1.2 技术栈

- ORM/连接层：SQLAlchemy + psycopg2
- 数据库迁移：Alembic
- 特性使用：JSONB、索引、唯一约束、事务回滚

---

## 2. 连接与会话管理

### 2.1 连接串来源

代码位置：`app/db/__init__.py`

环境变量：
- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_DB`

连接串组装：
```python
SQLALCHEMY_DATABASE_URL = postgresql+psycopg2://user:password@host:port/db
```

### 2.2 连接池参数

- `pool_pre_ping=True`：借连接前探活，减少"连接已断开"异常
- `pool_recycle=3600`：连接最长保活 1 小时
- `pool_size=20`：连接池基础大小
- `max_overflow=30`：最大溢出连接数

### 2.3 会话模式

两种会话获取方式：

1. `get_db()`：API 路由依赖注入
2. `get_db_context()`：Service/后台上下文调用（`with` 语法）

两者都具备：
- 成功提交 `commit`
- 异常回滚 `rollback`
- 最终关闭 `close`

---

## 3. 核心数据表结构

### 3.1 全量表清单

数据来源：`app/models/*.py` + Alembic 迁移

1. `t_user_info` - 用户基础信息
2. `t_model_setting` - 系统级模型服务配置
3. `t_user_model` - 用户模型绑定
4. `t_user_mcp` - 用户 MCP 配置
5. `t_chat_session` - 聊天会话
6. `t_chat_message` - 聊天消息
7. `t_interrupt_approval` - 审批状态

### 3.2 业务主表详解

#### 3.2.1 `t_user_info`

用途：用户基础信息与登录态 token

核心字段：
- `id` - 主键
- `user_name` - 登录用户名
- `nick_name` - 昵称
- `phone` - 手机号
- `password` - 密码哈希
- `token` - 令牌（当前实现保留）
- `create_time` / `update_time` - 时间戳

#### 3.2.2 `t_model_setting`

用途：系统级模型服务配置模板

核心字段：
- `service_name` / `service_type` / `service_url` - 服务信息
- `api_key_template` - API 密钥模板
- `models` (JSON) - 模型列表
- `is_system_default` / `is_enabled` - 配置标志
- `create_time` / `update_time` - 时间戳

#### 3.2.3 `t_user_model`

用途：用户绑定具体模型配置（可激活一个当前模型）

核心字段：
- `user_id` - 用户 ID
- `model_setting_id` (FK) - 模型设置 ID
- `service_name` - 服务名称
- `selected_model` - 选中的模型
- `api_key` / `api_url` - API 配置
- `custom_config` (JSON) - 自定义配置
- `is_active` - 是否激活
- `create_time` / `update_time` - 时间戳

#### 3.2.4 `t_user_mcp`

用途：用户级 MCP 配置

核心字段：
- `user_id` - 用户 ID
- `mcp_setting_json` (JSONB) - MCP 配置
- `create_time` / `update_time` - 时间戳

#### 3.2.5 `t_chat_session`

用途：会话主表（每个 `session_id` 唯一）

核心字段：
- `user_id` - 用户 ID
- `session_id` (唯一) - 会话 ID
- `title` - 会话标题
- `is_deleted` - 是否删除
- `create_time` / `update_time` - 时间戳

#### 3.2.6 `t_chat_message`

用途：消息流水

核心字段：
- `user_id` - 用户 ID
- `session_id` - 会话 ID
- `user_content` - 用户内容
- `model_content` - 模型内容
- `model_name` - 模型名称
- `tokens` / `latency_ms` - 性能指标
- `is_deleted` - 是否删除
- `extra_data` (JSONB) - 扩展数据（tool_calls/引用来源）
- `create_time` - 创建时间

#### 3.2.7 `t_interrupt_approval`

用途：审批状态持久化（LangGraph `interrupt()` 关键状态表）

核心字段：
- `id` - 主键（Bigint 自增）
- `session_id` - 会话维度（与聊天会话绑定，恢复时按会话读取）
- `message_id` - 审批消息标识（与前端审批卡片绑定，支持精确命中）
- `action_name` - 操作名（如 `execute_sql`）
- `action_args` (JSONB) - 操作参数
- `description` - 审批说明
- `status` - 审批状态（pending/approve/reject）
- `user_id` - 审批人
- `decision_time` - 审批时间
- `agent_name` - 源 Agent
- `subgraph_thread_id` - 子图线程 ID（LangGraph 恢复定位字段）
- `checkpoint_id` - 检查点 ID（恢复精确定位）
- `checkpoint_ns` - 检查点命名空间（恢复精确定位）
- `is_consumed` - 消费标志（审批结果是否已被 `[RESUME]` 消费）
- `create_time` / `update_time` - 时间戳

约束与索引：
- 唯一：`(session_id, message_id)`
- 复合索引：`(session_id, status, is_consumed)`
- 时间索引：`create_time`

### 3.3 ER 关系图

```mermaid
erDiagram
    t_user_info ||--o{ t_user_model : "user_id"
    t_model_setting ||--o{ t_user_model : "model_setting_id"
    t_user_info ||--o{ t_user_mcp : "user_id"
    t_user_info ||--o{ t_chat_session : "user_id"
    t_chat_session ||--o{ t_chat_message : "session_id"
    t_chat_session ||--o{ t_interrupt_approval : "session_id"
```

说明：
- `t_chat_message` 当前以 `session_id` 逻辑关联会话（非外键约束方式）
- `t_interrupt_approval` 通过 `session_id + message_id` 实现业务幂等与恢复定位

---

## 4. Alembic 数据库迁移技术

### 4.1 为什么使用 Alembic

1. 避免生产环境依赖 `Base.metadata.create_all()` 隐式建表
2. 保证数据库结构变更可追踪、可回滚、可审计

### 4.2 已落地内容

1. `alembic.ini` - Alembic 配置文件
2. `alembic/env.py` - Alembic 环境配置
3. 初始迁移：`alembic/versions/20260308_01_create_t_interrupt_approval.py` - 审批表创建脚本

### 4.3 使用方式

#### 4.3.1 执行迁移到最新版本

在项目根目录执行：

```bash
alembic upgrade head
```

#### 4.3.2 查看当前数据库版本

```bash
alembic current
```

#### 4.3.3 生成新迁移脚本（自动检测变更）

```bash
alembic revision --autogenerate -m "你的迁移说明"
```

### 4.4 生产环境建议

1. 生产环境设置 `AUTO_CREATE_TABLES=false`，只走 Alembic
2. 发布流程中先执行 `alembic upgrade head` 再启动应用
3. 迁移脚本必须代码评审后入主分支

---

## 5. 审批持久化链路

### 5.1 时序图

```mermaid
sequenceDiagram
    participant GR as GraphRunner
    participant IS as InterruptService
    participant PG as PostgreSQL
    participant FE as Frontend

    GR->>IS: register_pending_approval(...)
    IS->>PG: INSERT status=pending
    GR-->>FE: SSE interrupt 事件

    FE->>IS: handle_approval(approve/reject)
    IS->>PG: UPDATE status + decision_time

    GR->>IS: fetch_latest_resolved_approval(session)
    IS->>PG: SELECT status in (approve,reject) and is_consumed=false

    GR->>IS: mark_approval_consumed(session,message)
    IS->>PG: UPDATE is_consumed=true
    GR-->>FE: SSE 最终答复
```

### 5.2 设计要点

1. `status` 表示审批结果（pending/approve/reject）
2. `is_consumed` 表示恢复流程是否已消费该结果
3. 只有 `status in (approve, reject)` 且 `is_consumed=false` 会触发恢复
4. 恢复成功或终止后必须 `mark_approval_consumed`，防止重复恢复

---

## 6. 运行时读写链路

### 6.1 聊天落库链路

```mermaid
flowchart TD
    A[POST /api/v1/chat/stream] --> B[chat_service]
    B --> C[graph_runner 流式执行]
    C --> D[生成 SSE 输出]
    C --> E[拼接最终 stream 文本]
    E --> F[t_chat_session: get_or_create]
    E --> G[t_chat_message: insert]
```

说明：
- 当前已修复 `stream` 事件提取逻辑，避免"有输出但未入库"导致后续上下文丢失

### 6.2 与 LangGraph 的关键绑定点

#### 6.2.1 中断注册

- 位置：`app/agent/graph_runner.py::_register_interrupts`
- 行为：将 `action_requests` 写入 `t_interrupt_approval`，附带 `agent_name/subgraph_thread_id/checkpoint`

#### 6.2.2 恢复读取

- 位置：`app/services/interrupt_service.py::fetch_latest_resolved_approval`
- 条件：`status in (approve, reject)` 且 `is_consumed=false`

#### 6.2.3 恢复消费标记

- 位置：`app/services/interrupt_service.py::mark_approval_consumed`
- 目的：防重放、防重复恢复

---

## 7. 运维管理

### 7.1 部署与初始化

#### 7.1.1 生产环境配置

环境变量：
- `POSTGRES_HOST` - 数据库主机
- `POSTGRES_PORT` - 数据库端口
- `POSTGRES_USER` - 数据库用户
- `POSTGRES_PASSWORD` - 数据库密码
- `POSTGRES_DB` - 数据库名称

引擎参数：
- `pool_pre_ping=True` - 连接探活
- `pool_recycle=3600` - 连接保活 1 小时
- `pool_size=20` - 连接池大小
- `max_overflow=30` - 最大溢出

#### 7.1.2 初始化步骤

1. 创建数据库和用户
2. 执行 Alembic 迁移：`alembic upgrade head`
3. 验证关键表是否存在
4. 验证审批链路：`pending -> approve -> consumed`

### 7.2 运维策略

#### 7.2.1 数据清理建议

1. `pending` 且超长时间未处理（如 >7 天）可告警
2. `is_consumed=true` 的历史数据可定期归档或清理（如保留 30~90 天）

清理 SQL 示例：
```sql
DELETE FROM t_interrupt_approval
WHERE is_consumed = true
  AND update_time < NOW() - INTERVAL '90 days';
```

#### 7.2.2 监控建议

建议监控指标：
1. 每分钟新增审批量
2. 审批通过率/拒绝率
3. 审批后恢复成功率
4. 长时间 `pending` 数量

#### 7.2.3 性能建议

- 按实例并发能力调整 `pool_size/max_overflow`
- 对慢 SQL 开启日志与分析（`pg_stat_statements`）
- 定期执行 VACUUM 和 ANALYZE

---

## 8. 故障排查

### 8.1 关键运维 SQL

#### 8.1.1 查看待处理审批

```sql
SELECT id, session_id, message_id, action_name, status, is_consumed, create_time
FROM t_interrupt_approval
WHERE status = 'pending' AND is_consumed = false
ORDER BY create_time DESC;
```

#### 8.1.2 查看待恢复审批

```sql
SELECT id, session_id, message_id, status, decision_time, is_consumed
FROM t_interrupt_approval
WHERE status IN ('approve', 'reject') AND is_consumed = false
ORDER BY decision_time DESC NULLS LAST, update_time DESC;
```

#### 8.1.3 聊天会话与消息核对

```sql
SELECT s.session_id, s.title, s.create_time, COUNT(m.id) AS msg_count
FROM t_chat_session s
LEFT JOIN t_chat_message m ON m.session_id = s.session_id
WHERE s.user_id = :user_id
GROUP BY s.session_id, s.title, s.create_time
ORDER BY s.create_time DESC;
```

#### 8.1.4 最近消息（排查"无最终答复"）

```sql
SELECT id, session_id, model_name, LEFT(model_content, 200) AS content_preview, create_time
FROM t_chat_message
WHERE session_id = :session_id
ORDER BY id DESC
LIMIT 20;
```

### 8.2 常见故障与排查

#### 8.2.1 审批已点"批准"但恢复失败

检查：
1. `t_interrupt_approval` 对应记录是否 `status=approve`
2. `is_consumed` 是否提前变为 `true`
3. `subgraph_thread_id/checkpoint_id/checkpoint_ns` 是否为空

#### 8.2.2 找不到最终答复

检查：
1. SSE 是否有 `stream` 事件
2. `t_chat_message.model_content` 是否写入
3. 若只有 thinking 无 stream，检查 `graph_runner` 输出过滤逻辑

#### 8.2.3 查询慢

检查：
1. 是否命中 `session_id`、`status` 等索引
2. 是否存在大范围全表扫描
3. `pool_size/max_overflow` 是否匹配并发量

### 8.3 常见问题

#### 8.3.1 为什么不用 Redis？

当前阶段 PG 已经可满足：
1. 强一致持久化
2. 审批体量相对可控
3. 降低额外基础设施复杂度

后续若审批吞吐显著增加，再拆分 Redis/消息队列。

#### 8.3.2 如果 PG 临时不可用怎么办？

代码里保留了内存兜底：
1. 写 PG 失败时降级写内存
2. 读 PG 失败时降级读内存

但生产仍应优先保证 PG 可用性，避免跨实例时内存态不一致。

---

## 9. 安全与合规建议

1. `api_key`、`password` 建议迁移到加密存储（字段级脱敏/加密）
2. 审批表 `action_args` 可能含敏感 SQL，建议日志侧只输出摘要哈希
3. 聊天消息表建议按保留策略归档（冷热分层）

---

## 10. 性能优化建议

### 10.1 已有重点索引

- `t_chat_session(session_id)` - 唯一索引
- `t_chat_message(session_id)` - 会话索引
- `t_interrupt_approval(session_id, status, is_consumed)` - 审批复合索引

### 10.2 建议新增（按实际压测后执行）

1. `t_chat_message(session_id, create_time DESC)` - 提升会话消息拉取性能
2. `t_chat_message(user_id, create_time DESC)` - 提升用户维度审计查询
3. `t_user_model(user_id, is_active)` - 提升激活模型查询

---

## 11. 上线与回滚清单（PG 视角）

### 11.1 上线前

1. 备份数据库（全量 + 最近增量）
2. 执行 `alembic upgrade head`
3. 验证关键表是否存在（7 张主表）
4. 验证审批链路：`pending -> approve -> consumed`

### 11.2 回滚策略

1. 若迁移失败：回滚应用版本 + 恢复数据库快照
2. 若仅业务异常：保留新表，按应用逻辑热修复，避免数据回退丢失审批记录

---

## 12. 配置清单

### 12.1 连接配置

- `POSTGRES_HOST` - 数据库主机
- `POSTGRES_PORT` - 数据库端口
- `POSTGRES_USER` - 数据库用户
- `POSTGRES_PASSWORD` - 数据库密码
- `POSTGRES_DB` - 数据库名称

### 12.2 迁移与建表策略

- `AUTO_CREATE_TABLES` - 是否自动建表（生产建议 false）

### 12.3 审批/策略/缓存协同配置

- `SQL_LOCAL_TABLE_WHITELIST` - SQL 本地表白名单
- `SQL_YUNYOU_TABLE_WHITELIST` - SQL 云柚表白名单
- `SEMANTIC_CACHE_TTL_SECONDS` - 语义缓存 TTL
- `SEMANTIC_CACHE_MAX_SIZE` - 语义缓存最大大小

---

## 13. 参考文件索引

### 13.1 代码文件

- `app/db/__init__.py` - 数据库连接与会话管理
- `app/models/user_info.py` - 用户信息模型
- `app/models/model_setting.py` - 模型设置模型
- `app/models/user_model.py` - 用户模型模型
- `app/models/user_mcp.py` - MCP 配置模型
- `app/models/chat_history.py` - 聊天历史模型
- `app/models/interrupt_approval.py` - 审批状态模型
- `app/services/interrupt_service.py` - 审批服务
- `app/main.py` - 应用主入口
- `alembic/env.py` - Alembic 环境配置
- `alembic/versions/20260308_01_create_t_interrupt_approval.py` - 审批表迁移脚本

### 13.2 配置文件

- `alembic.ini` - Alembic 配置
- `.env` - 环境变量配置

### 13.3 SQL 脚本

- `docs/sql/20260308_create_t_interrupt_approval.sql` - 审批表创建 SQL

---

*文档最后更新: 2026-03-08*
