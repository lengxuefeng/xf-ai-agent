# -*- coding: utf-8 -*-
"""SQL 工具模板与文案常量。"""


SCHEMA_TABLE_LIST_SQL = """
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_type = 'BASE TABLE'
ORDER BY table_name;
"""


SCHEMA_COLUMN_LIST_SQL = """
SELECT
  column_name,
  data_type,
  is_nullable,
  column_default
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = :table_name
ORDER BY ordinal_position;
"""


SQL_MSG_EMPTY = "SQL 为空，无法执行。"
SQL_MSG_EXEC_ERROR_PREFIX = "执行 SQL 时发生错误: "
SQL_MSG_NO_DATA = "查询成功，但没有返回结果。"
SQL_MSG_SUCCESS_AFFECT_PREFIX = "操作成功，影响行数: "
SQL_MSG_SCHEMA_ERROR_PREFIX = "获取表结构时发生错误: "

