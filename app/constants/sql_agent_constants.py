# -*- coding: utf-8 -*-
"""SQL Agent 固定文案与模板常量。"""


SQL_AGENT_SCHEMA_LOADED_MESSAGE = "数据库表结构上下文已加载。"
SQL_AGENT_ORDER_RULE_HINT = "若用户要求按 id 倒序前 N 条，优先输出 ORDER BY id DESC LIMIT N。"
SQL_AGENT_APPROVAL_MESSAGE = "需要审批 SQL 执行"
SQL_AGENT_APPROVAL_DESC_TEMPLATE = "即将执行 SQL: {sql}，请确认是否安全。"
SQL_AGENT_REJECTED_MESSAGE = "用户拒绝了 SQL 执行。"
SQL_AGENT_RESULT_PREFIX = "SQL 执行结果：\n"

