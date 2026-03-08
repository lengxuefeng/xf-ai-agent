# Alembic 迁移指南

> 适用项目：`xf-ai-agent`

## 1. 为什么引入 Alembic

1. 避免生产环境依赖 `Base.metadata.create_all()` 隐式建表。
2. 保证数据库结构变更可追踪、可回滚、可审计。

## 2. 已落地内容

1. `alembic.ini`
2. `alembic/env.py`
3. 初始迁移：`alembic/versions/20260308_01_create_t_interrupt_approval.py`

## 3. 使用方式

在项目根目录执行：

```bash
alembic upgrade head
```

查看当前版本：

```bash
alembic current
```

生成新迁移（自动检测）：

```bash
alembic revision --autogenerate -m "your migration message"
```

## 4. 生产建议

1. 生产环境设置 `AUTO_CREATE_TABLES=false`，只走 Alembic。
2. 发布流程中先执行 `alembic upgrade head` 再启动应用。
3. 迁移脚本必须代码评审后入主分支。

