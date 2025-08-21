from langchain_core.tools import tool
from sqlalchemy import text

from db.mysql import get_db


# @tool
def execute_sql(query: str) -> str:
    """
    执行一个 SQL 查询并返回结果。
    - 对于修改操作（INSERT/UPDATE/DELETE），会自动提交事务。
    - 对于查询操作（SELECT），直接返回结果。

    Args:
        query (str): 要执行的 SQL 查询语句。

    Returns:
        str: 查询结果或错误信息，格式化为字符串。
    """
    try:
        with get_db() as db:
            if query.strip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
                db.execute(text(query))
                return "操作成功完成。"
            else:
                rows = db.execute(text(query)).fetchall()
                if not rows:
                    return "查询成功，但没有返回结果。"
                return "\n".join(str(row) for row in rows)
    except Exception as e:
        return f"执行 SQL 时发生错误: {e}"


# @tool
def get_schema() -> str:
    """
    获取数据库中所有表的表结构信息（schema）。
    当需要知道数据库中有哪些表、每个表有哪些字段时，调用此工具。
    """
    try:
        with get_db() as db:
            schema_info = "数据库表结构如下:\n"
            tables_result = db.execute(text("SHOW TABLES;"))
            tables = [row[0] for row in tables_result.fetchall()]

            for table_name in tables:
                columns_result = db.execute(text(f"SHOW COLUMNS FROM `{table_name}`;"))
                columns = columns_result.fetchall()

                schema_info += f"\n表名: {table_name}\n"
                schema_info += "字段名 | 数据类型 | 是否允许为空 | 键 | 默认值 | 额外信息\n"
                schema_info += "---|---|---|---|---|---\n"

                for field, col_type, is_null, key, default, extra in columns:
                    schema_info += f"{field} | {col_type} | {is_null} | {key} | {default} | {extra}\n"

            return schema_info
    except Exception as e:
        return f"获取表结构时发生错误: {e}"


if __name__ == '__main__':
    # 1. 测试 SELECT
    print("===== 测试 SELECT 语句 =====")
    result = execute_sql("SELECT NOW();")
    print(result)

    # 2. 测试 INSERT
    print("\n===== 测试 INSERT 语句 =====")
    result = execute_sql("INSERT INTO t_user_mcp (user_id,mcp_setting_json) VALUES ('123','{}');")
    print(result)

    # 3. 测试获取表结构
    print("\n===== 测试获取表结构 =====")
    result = get_schema()
    print(result)
