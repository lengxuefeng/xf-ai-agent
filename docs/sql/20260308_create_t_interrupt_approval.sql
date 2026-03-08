-- 审批状态持久化表（PostgreSQL）
-- 执行前请确认当前数据库连接与权限

CREATE TABLE IF NOT EXISTS t_interrupt_approval (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(120) NOT NULL,
    message_id VARCHAR(200) NOT NULL,

    action_name VARCHAR(120) NOT NULL,
    action_args JSONB NULL,
    description TEXT NULL,

    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    user_id BIGINT NULL,
    decision_time TIMESTAMP NULL,

    agent_name VARCHAR(120) NULL,
    subgraph_thread_id VARCHAR(180) NULL,
    checkpoint_id VARCHAR(200) NULL,
    checkpoint_ns VARCHAR(200) NULL,

    is_consumed BOOLEAN NOT NULL DEFAULT FALSE,

    create_time TIMESTAMP NOT NULL DEFAULT NOW(),
    update_time TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_interrupt_session_message UNIQUE (session_id, message_id)
);

CREATE INDEX IF NOT EXISTS idx_interrupt_session_status_consumed
    ON t_interrupt_approval (session_id, status, is_consumed);

CREATE INDEX IF NOT EXISTS idx_interrupt_create_time
    ON t_interrupt_approval (create_time);

-- 建议：配合应用层定期清理已消费历史数据
-- DELETE FROM t_interrupt_approval
-- WHERE is_consumed = TRUE
--   AND update_time < NOW() - INTERVAL '90 days';
