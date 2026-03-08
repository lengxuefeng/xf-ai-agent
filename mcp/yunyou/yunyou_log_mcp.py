from typing import List

from mcp.server import FastMCP

yy_log_mcp = FastMCP("YunYouLogMCP")


@yy_log_mcp.tool()
def get_log_info(user_id: str) -> List[dict]:
    """获取用户日志信息"""
    return yy_log_mcp.get_log_info(user_id)
