-- =====================================================
-- XF-AI-Agent 数据库初始化脚本
-- =====================================================
-- 适用数据库：PostgreSQL 14+
-- 创建时间：2026-03-08
-- 更新时间：2026-03-10（新增会话状态表）
-- 说明：包含项目使用的所有数据表（共8张表）
-- =====================================================

-- 创建数据库（如果需要）
-- CREATE DATABASE xf_ai_agent
--     WITH OWNER = postgres
--     ENCODING = 'UTF8'
--     TABLESPACE = pg_default;
-- 
-- COMMENT ON DATABASE xf_ai_agent IS 'XF-AI-Agent 项目数据库';


-- =====================================================
-- 1. 用户信息表 (t_user_info)
-- =====================================================
-- 说明：存储用户基础信息和登录态
-- =====================================================

CREATE TABLE IF NOT EXISTS t_user_info (
    -- 主键
    id BIGSERIAL PRIMARY KEY,
    
    -- 用户基本信息
    user_name VARCHAR(50) NOT NULL,             -- 用户名
    nick_name VARCHAR(50) NOT NULL,             -- 昵称
    phone VARCHAR(50) NOT NULL,                 -- 手机号
    password VARCHAR(255) NOT NULL,             -- 密码（加密存储）
    token VARCHAR(255),                       -- 登录令牌（当前实现保留）
    
    -- 时间字段
    create_time TIMESTAMP NOT NULL DEFAULT NOW(),  -- 创建时间
    update_time TIMESTAMP NOT NULL DEFAULT NOW()  -- 更新时间
);

-- 添加注释
COMMENT ON TABLE t_user_info IS '用户信息表';
COMMENT ON COLUMN t_user_info.user_name IS '用户名，用于登录';
COMMENT ON COLUMN t_user_info.nick_name IS '用户昵称，用于展示';
COMMENT ON COLUMN t_user_info.phone IS '手机号';
COMMENT ON COLUMN t_user_info.password IS '密码（加密存储）';
COMMENT ON COLUMN t_user_info.token IS '登录令牌（JWT Token）';

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_user_info_user_name ON t_user_info(user_name);
CREATE INDEX IF NOT EXISTS idx_user_info_phone ON t_user_info(phone);


-- =====================================================
-- 2. 系统模型服务配置表 (t_model_setting)
-- =====================================================
-- 说明：存储系统预定义的模型服务配置模板
-- =====================================================

CREATE TABLE IF NOT EXISTS t_model_setting (
    -- 主键
    id BIGSERIAL PRIMARY KEY,
    
    -- 服务信息
    service_name VARCHAR(100) NOT NULL,          -- 服务名称（如 OpenAI、Gemini）
    service_type VARCHAR(50) NOT NULL,           -- 服务类型（如 openai、gemini）
    service_url VARCHAR(500) NOT NULL,           -- API服务基础地址
    api_key_template VARCHAR(100),              -- API密钥格式模板
    
    -- UI展示信息
    icon VARCHAR(50) DEFAULT 'FiCpu',          -- 服务图标名称（Feather Icons）
    models JSONB NOT NULL,                      -- 支持的模型列表（JSON格式）
    description TEXT,                            -- 服务描述信息
    
    -- 配置状态
    is_system_default BOOLEAN DEFAULT TRUE,      -- 是否为系统默认配置
    is_enabled BOOLEAN DEFAULT TRUE,             -- 是否启用该服务
    
    -- 时间字段
    create_time TIMESTAMP NOT NULL DEFAULT NOW(),  -- 创建时间
    update_time TIMESTAMP NOT NULL DEFAULT NOW()   -- 更新时间
);

-- 添加注释
COMMENT ON TABLE t_model_setting IS '系统模型服务配置表';
COMMENT ON COLUMN t_model_setting.service_name IS '服务名称（如 OpenAI、Gemini）';
COMMENT ON COLUMN t_model_setting.service_type IS '服务类型（如 openai、gemini）';
COMMENT ON COLUMN t_model_setting.service_url IS 'API服务基础地址';
COMMENT ON COLUMN t_model_setting.api_key_template IS 'API密钥格式模板';
COMMENT ON COLUMN t_model_setting.icon IS '服务图标名称（Feather Icons）';
COMMENT ON COLUMN t_model_setting.models IS '支持的模型列表（JSON格式）';
COMMENT ON COLUMN t_model_setting.is_system_default IS '是否为系统默认配置';
COMMENT ON COLUMN t_model_setting.is_enabled IS '是否启用该服务';


-- =====================================================
-- 3. 用户模型配置表 (t_user_model)
-- =====================================================
-- 说明：存储用户个人的模型配置，用户可以创建多个配置并激活其中一个
-- =====================================================

CREATE TABLE IF NOT EXISTS t_user_model (
    -- 主键
    id BIGSERIAL PRIMARY KEY,
    
    -- 关联信息
    user_id BIGINT NOT NULL,                   -- 用户ID
    model_setting_id BIGINT NOT NULL,           -- 关联的系统模型服务ID
    
    -- 业务字段
    service_name VARCHAR(100) NOT NULL,         -- 用户自定义服务名称
    selected_model VARCHAR(100) NOT NULL,       -- 用户选择的具体模型名称
    api_key VARCHAR(500) NOT NULL,             -- 用户的API密钥
    api_url VARCHAR(500),                      -- 用户自定义的API地址（可选）
    custom_config JSONB,                        -- 用户自定义配置（JSON格式）
    
    -- 状态标志
    is_active BOOLEAN DEFAULT FALSE,             -- 是否为当前激活的配置
    
    -- 时间字段
    create_time TIMESTAMP NOT NULL DEFAULT NOW(),  -- 创建时间
    update_time TIMESTAMP NOT NULL DEFAULT NOW()   -- 更新时间
);

-- 添加注释
COMMENT ON TABLE t_user_model IS '用户模型配置表';
COMMENT ON COLUMN t_user_model.user_id IS '用户ID';
COMMENT ON COLUMN t_user_model.model_setting_id IS '关联的系统模型服务ID';
COMMENT ON COLUMN t_user_model.service_name IS '用户自定义服务名称';
COMMENT ON COLUMN t_user_model.selected_model IS '用户选择的具体模型名称';
COMMENT ON COLUMN t_user_model.api_key IS '用户的API密钥';
COMMENT ON COLUMN t_user_model.api_url IS '用户自定义的API地址（可选）';
COMMENT ON COLUMN t_user_model.custom_config IS '用户自定义配置（JSON格式）';
COMMENT ON COLUMN t_user_model.is_active IS '是否为当前激活的配置';

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_user_model_user_id ON t_user_model(user_id);
CREATE INDEX IF NOT EXISTS idx_user_model_model_setting_id ON t_user_model(model_setting_id);
CREATE INDEX IF NOT EXISTS idx_user_model_is_active ON t_user_model(is_active) WHERE is_active = TRUE;

-- 添加外键约束
ALTER TABLE t_user_model 
    ADD CONSTRAINT fk_user_model_setting_id 
    FOREIGN KEY (model_setting_id) REFERENCES t_model_setting(id);


-- =====================================================
-- 4. 用户MCP配置表 (t_user_mcp)
-- =====================================================
-- 说明：存储用户的MCP（Model Context Protocol）配置
-- =====================================================

CREATE TABLE IF NOT EXISTS t_user_mcp (
    -- 主键
    id BIGSERIAL PRIMARY KEY,
    
    -- 用户信息
    user_id BIGINT NOT NULL,                   -- 用户ID
    
    -- MCP配置
    mcp_setting_json VARCHAR(255) NOT NULL,     -- MCP配置JSON
    
    -- 时间字段
    create_time TIMESTAMP NOT NULL DEFAULT NOW(),  -- 创建时间
    update_time TIMESTAMP NOT NULL DEFAULT NOW()   -- 更新时间
);

-- 添加注释
COMMENT ON TABLE t_user_mcp IS '用户MCP配置表';
COMMENT ON COLUMN t_user_mcp.user_id IS '用户ID';
COMMENT ON COLUMN t_user_mcp.mcp_setting_json IS 'MCP配置（JSON格式）';

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_user_mcp_user_id ON t_user_mcp(user_id);


-- =====================================================
-- 5. 聊天会话表 (t_chat_session)
-- =====================================================
-- 说明：存储用户的聊天会话，每个session_id唯一
-- =====================================================

CREATE TABLE IF NOT EXISTS t_chat_session (
    -- 主键
    id BIGSERIAL PRIMARY KEY,
    
    -- 用户信息和会话标识
    user_id BIGINT NOT NULL,                   -- 用户ID
    session_id VARCHAR(100) NOT NULL,          -- 会话ID（唯一标识一个会话）
    title VARCHAR(200) NOT NULL DEFAULT '新对话',  -- 会话标题
    
    -- 状态标志
    is_deleted BOOLEAN DEFAULT FALSE,             -- 是否删除（软删除）
    
    -- 时间字段
    create_time TIMESTAMP NOT NULL DEFAULT NOW(),  -- 创建时间
    update_time TIMESTAMP NOT NULL DEFAULT NOW()   -- 更新时间
);

-- 添加注释
COMMENT ON TABLE t_chat_session IS '聊天会话表';
COMMENT ON COLUMN t_chat_session.user_id IS '用户ID';
COMMENT ON COLUMN t_chat_session.session_id IS '会话ID（唯一标识一个会话）';
COMMENT ON COLUMN t_chat_session.title IS '会话标题';
COMMENT ON COLUMN t_chat_session.is_deleted IS '是否删除（软删除）';

-- 创建索引和唯一约束
CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_session_session_id ON t_chat_session(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_session_user_id ON t_chat_session(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_session_create_time ON t_chat_session(create_time DESC);
CREATE INDEX IF NOT EXISTS idx_chat_session_is_deleted ON t_chat_session(is_deleted) WHERE is_deleted = FALSE;


-- =====================================================
-- 6. 聊天消息表 (t_chat_message)
-- =====================================================
-- 说明：存储聊天的所有消息
-- =====================================================

CREATE TABLE IF NOT EXISTS t_chat_message (
    -- 主键
    id BIGSERIAL PRIMARY KEY,
    
    -- 用户和会话信息
    user_id BIGINT NOT NULL,                   -- 用户ID
    session_id VARCHAR(100) NOT NULL,         -- 会话ID
    
    -- 消息内容（使用TEXT类型支持超长内容）
    user_content TEXT NOT NULL,                 -- 用户输入的内容
    model_content TEXT NOT NULL,                -- 模型输出的内容
    
    -- 模型信息
    model_name VARCHAR(100),                    -- 使用的模型名称
    
    -- 性能指标
    tokens BIGINT DEFAULT 0,                     -- Token数量
    latency_ms BIGINT DEFAULT 0,                -- 响应延迟（毫秒）
    
    -- 扩展数据（存储tool_calls、引用来源等）
    extra_data JSONB,                            -- 扩展数据（JSONB格式，支持SQL查询）
    
    -- 状态标志
    is_deleted BOOLEAN DEFAULT FALSE,             -- 是否删除（软删除）
    
    -- 时间字段
    create_time TIMESTAMP NOT NULL DEFAULT NOW()   -- 创建时间
);

-- 添加注释
COMMENT ON TABLE t_chat_message IS '聊天消息表';
COMMENT ON COLUMN t_chat_message.user_id IS '用户ID';
COMMENT ON COLUMN t_chat_message.session_id IS '会话ID';
COMMENT ON COLUMN t_chat_message.user_content IS '用户输入的内容';
COMMENT ON COLUMN t_chat_message.model_content IS '模型输出的内容';
COMMENT ON COLUMN t_chat_message.model_name IS '使用的模型名称';
COMMENT ON COLUMN t_chat_message.tokens IS 'Token数量';
COMMENT ON COLUMN t_chat_message.latency_ms IS '响应延迟（毫秒）';
COMMENT ON COLUMN t_chat_message.extra_data IS '扩展数据（JSONB格式）';
COMMENT ON COLUMN t_chat_message.is_deleted IS '是否删除（软删除）';

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_chat_message_session_id ON t_chat_message(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_message_user_id ON t_chat_message(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_message_create_time ON t_chat_message(create_time DESC);
CREATE INDEX IF NOT EXISTS idx_chat_message_session_time ON t_chat_message(session_id, create_time DESC);
CREATE INDEX IF NOT EXISTS idx_chat_message_user_time ON t_chat_message(user_id, create_time DESC);


-- =====================================================
-- 7. 审批状态表 (t_interrupt_approval)
-- =====================================================
-- 说明：存储LangGraph interrupt()的审批请求和结果
--       实现人工审批流程的持久化和恢复
-- =====================================================

CREATE TABLE IF NOT EXISTS t_interrupt_approval (
    -- 主键
    id BIGSERIAL PRIMARY KEY,
    
    -- 审批标识
    session_id VARCHAR(120) NOT NULL,          -- 会话维度（与聊天会话绑定，恢复时按会话读取）
    message_id VARCHAR(200) NOT NULL,          -- 审批消息标识（与前端审批卡片绑定，支持精确命中）
    
    -- 审批操作信息
    action_name VARCHAR(120) NOT NULL,           -- 操作名（如 execute_sql）
    action_args JSONB,                          -- 操作参数（JSONB，保留原始工具入参）
    description TEXT,                            -- 审批说明（展示给用户/审计用途）
    
    -- 审批状态
    status VARCHAR(20) NOT NULL DEFAULT 'pending', -- 审批状态（pending/approve/reject）
    user_id BIGINT,                             -- 审批人ID（执行了审批的用户）
    decision_time TIMESTAMP,                     -- 审批时间（何时approve/reject）
    
    -- 恢复定位信息（LangGraph专用）
    agent_name VARCHAR(120),                     -- 触发审批的Agent名称
    subgraph_thread_id VARCHAR(180),             -- 子图线程ID（LangGraph恢复定位字段）
    checkpoint_id VARCHAR(200),                 -- 检查点ID（恢复精确定位）
    checkpoint_ns VARCHAR(200),                  -- 检查点命名空间（恢复精确定位）
    
    -- 消费标志
    is_consumed BOOLEAN NOT NULL DEFAULT FALSE,   -- 是否已被恢复流程消费（防重放、防重复恢复）
    
    -- 时间字段
    create_time TIMESTAMP NOT NULL DEFAULT NOW(), -- 创建时间
    update_time TIMESTAMP NOT NULL DEFAULT NOW()   -- 更新时间
);

-- 添加注释
COMMENT ON TABLE t_interrupt_approval IS '审批状态持久化表（LangGraph interrupt()关键状态表）';
COMMENT ON COLUMN t_interrupt_approval.session_id IS '会话维度（与聊天会话绑定，恢复时按会话读取）';
COMMENT ON COLUMN t_interrupt_approval.message_id IS '审批消息标识（与前端审批卡片绑定）';
COMMENT ON COLUMN t_interrupt_approval.action_name IS '操作名（如 execute_sql）';
COMMENT ON COLUMN t_interrupt_approval.action_args IS '操作参数（JSONB，保留原始工具入参）';
COMMENT ON COLUMN t_interrupt_approval.description IS '审批说明';
COMMENT ON COLUMN t_interrupt_approval.status IS '审批状态（pending/approve/reject）';
COMMENT ON COLUMN t_interrupt_approval.user_id IS '审批人ID';
COMMENT ON COLUMN t_interrupt_approval.decision_time IS '审批时间';
COMMENT ON COLUMN t_interrupt_approval.agent_name IS '触发审批的Agent名称';
COMMENT ON COLUMN t_interrupt_approval.subgraph_thread_id IS '子图线程ID（LangGraph恢复定位）';
COMMENT ON COLUMN t_interrupt_approval.checkpoint_id IS '检查点ID（恢复精确定位）';
COMMENT ON COLUMN t_interrupt_approval.checkpoint_ns IS '检查点命名空间（恢复精确定位）';
COMMENT ON COLUMN t_interrupt_approval.is_consumed IS '是否已被恢复流程消费';

-- 创建唯一约束（防止重复审批记录）
CREATE UNIQUE INDEX IF NOT EXISTS uq_interrupt_session_message 
    ON t_interrupt_approval(session_id, message_id);

-- 创建复合索引（用于查询可恢复的审批记录）
CREATE INDEX IF NOT EXISTS idx_interrupt_session_status_consumed 
    ON t_interrupt_approval(session_id, status, is_consumed);

-- 创建时间索引（用于审计和清理）
CREATE INDEX IF NOT EXISTS idx_interrupt_create_time 
    ON t_interrupt_approval(create_time DESC);


-- =====================================================
-- 8. 会话状态表 (t_session_state)
-- =====================================================
-- 说明：存储会话级的结构化槽位和上下文信息
--       用于维护多轮对话的结构化上下文，降低重复追问和误路由
-- =====================================================

CREATE TABLE IF NOT EXISTS t_session_state (
    -- 主键
    id BIGSERIAL PRIMARY KEY,
    
    -- 用户和会话信息
    user_id BIGINT NOT NULL,                   -- 用户ID
    session_id VARCHAR(100) NOT NULL,          -- 会话ID（每个session_id唯一）
    
    -- 槽位字典（存储结构化上下文）
    slots JSONB,                                -- 槽位字典（如 city/name/age/gender/height_cm/weight_kg/last_topic/key_facts）
    
    -- 人类可读摘要
    summary_text TEXT,                          -- 摘要文本（用于注入系统上下文）
    
    -- 路由快照
    last_route JSONB,                           -- 保存最近一次路由快照，用于排障和策略分析
    
    -- 统计信息
    turn_count BIGINT DEFAULT 0,                 -- 累积轮次，便于后续策略扩展
    
    -- 状态标志
    is_deleted BOOLEAN DEFAULT FALSE,             -- 是否删除（软删除）
    
    -- 时间字段
    create_time TIMESTAMP NOT NULL DEFAULT NOW(),  -- 创建时间
    update_time TIMESTAMP NOT NULL DEFAULT NOW()   -- 更新时间
);

-- 添加注释
COMMENT ON TABLE t_session_state IS '会话状态表（每个session_id一条）';
COMMENT ON COLUMN t_session_state.user_id IS '用户ID';
COMMENT ON COLUMN t_session_state.session_id IS '会话ID（唯一标识一个会话）';
COMMENT ON COLUMN t_session_state.slots IS '槽位字典（JSONB格式，包含：city/name/age/gender/height_cm/weight_kg/last_topic/key_facts）';
COMMENT ON COLUMN t_session_state.summary_text IS '人类可读摘要（用于注入系统上下文）';
COMMENT ON COLUMN t_session_state.last_route IS '最近一次路由快照（JSONB格式）';
COMMENT ON COLUMN t_session_state.turn_count IS '累积轮次';
COMMENT ON COLUMN t_session_state.is_deleted IS '是否删除（软删除）';

-- 创建唯一约束
CREATE UNIQUE INDEX IF NOT EXISTS idx_session_state_session_id ON t_session_state(session_id);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_session_state_user_id ON t_session_state(user_id);
CREATE INDEX IF NOT EXISTS idx_session_state_create_time ON t_session_state(create_time DESC);
CREATE INDEX IF NOT EXISTS idx_session_state_is_deleted ON t_session_state(is_deleted) WHERE is_deleted = FALSE;


-- =====================================================
-- 9. 初始化系统数据
-- =====================================================
-- 说明：插入系统默认的模型服务配置
-- =====================================================

-- 插入OpenAI服务配置
INSERT INTO t_model_setting (service_name, service_type, service_url, api_key_template, icon, models, description)
VALUES (
    'OpenAI',
    'openai',
    'https://api.openai.com/v1',
    'sk-{api_key}',
    'FiCpu',
    '["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]'::jsonb,
    'OpenAI 官方模型服务'
) ON CONFLICT DO NOTHING;

-- 插入Gemini服务配置
INSERT INTO t_model_setting (service_name, service_type, service_url, api_key_template, icon, models, description)
VALUES (
    'Gemini',
    'gemini',
    'https://generativelanguage.googleapis.com/v1beta',
    '',
    'FiCpu',
    '["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]'::jsonb,
    'Google Gemini 模型服务'
) ON CONFLICT DO NOTHING;

-- 插入Ollama服务配置
INSERT INTO t_model_setting (service_name, service_type, service_url, api_key_template, icon, models, description)
VALUES (
    'Ollama',
    'ollama',
    'http://localhost:11434',
    '',
    'FiCpu',
    '["llama3:8b", "llama3:70b", "mistral:7b"]'::jsonb,
    '本地Ollama模型服务'
) ON CONFLICT DO NOTHING;


-- =====================================================
-- 10. 数据清理策略（可选）
-- =====================================================
-- 说明：定期清理历史数据，保持数据库性能
-- =====================================================

-- 清理已消费且超过90天的审批记录（注释状态，按需启用）
-- DELETE FROM t_interrupt_approval
-- WHERE is_consumed = TRUE
--   AND update_time < NOW() - INTERVAL '90 days';

-- 清理已删除且超过180天的会话状态记录（注释状态，按需启用）
-- DELETE FROM t_session_state
-- WHERE is_deleted = TRUE
--   AND update_time < NOW() - INTERVAL '180 days';


-- =====================================================
-- 11. 运维建议
-- =====================================================
-- 1. 定期执行 VACUUM 和 ANALYZE
--    VACUUM ANALYZE;
-- 
-- 2. 监控表大小和增长趋势
--    SELECT schemaname, tablename, 
--           pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
--    FROM pg_tables
--    WHERE schemaname = 'public'
--    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
-- 
-- 3. 监控审批记录数量
--    SELECT status, COUNT(*) as count
--    FROM t_interrupt_approval
--    GROUP BY status;
-- 
-- 4. 监控会话状态记录数量
--    SELECT COUNT(*) as total_states,
--           COUNT(*) FILTER (WHERE turn_count > 10) as long_sessions
--    FROM t_session_state;
-- 
-- 5. 监控长时间未处理的审批记录（超过7天）
--    SELECT id, session_id, action_name, status, create_time
--    FROM t_interrupt_approval
--    WHERE status = 'pending'
--      AND create_time < NOW() - INTERVAL '7 days'
--    ORDER BY create_time;
-- 
-- 6. 监控会话槽位使用情况
--    SELECT session_id, 
--           slots->>'city' as city,
--           slots->>'name' as name,
--           slots->>'last_topic' as last_topic,
--           turn_count,
--           update_time
--    FROM t_session_state
--    WHERE turn_count > 0
--    ORDER BY update_time DESC
--    LIMIT 20;
-- =====================================================


-- 初始化脚本执行完成
-- 数据库：xf_ai_agent
-- 表数量：8（新增会话状态表）
-- 初始化时间：2026-03-10
-- =====================================================
