"""
Rule Engine 动作处理器
"""
import datetime
from zhdate import ZhDate
from utils.custom_logger import get_logger

log = get_logger(__name__)

def handle_action(action: str) -> dict:
    """
    根据给定的 action URI执行对应的动作能力，
    将其返回结果组装为字典形式，供外部 response_template 注入。
    
    Args:
        action: 规则配置中的动作标识 (长啥样如: "call_tool://system/date")
        
    Returns:
        dict: 键值对字典以被 str.format(**kwargs) 消费
    """
    try:
        # ------- 系统时间工具 -------
        if action == "call_tool://system/time":
            now = datetime.datetime.now()
            return {"time": now.strftime("%H:%M:%S")}
            
        # ------- 系统日期工具 -------
        elif action == "call_tool://system/date":
            now = datetime.datetime.now()
            
            # 使用 zhdate 将公历转为中国农历
            lunar = ZhDate.from_datetime(now)
            
            # 基础星期换算
            weekdays = ["一", "二", "三", "四", "五", "六", "日"]
            
            return {
                "gregorian": now.strftime("%Y年%m月%d日"),
                "lunar": f"{lunar.chinese()}",
                "weekday": f"周{weekdays[now.weekday()]}"
            }
            
        # ------- 系统版本工具 -------
        elif action == "call_tool://system/version":
            now = datetime.datetime.now()
            return {
                "version": "Agent Core v2.1 (3-Tier DAG)",
                "compile_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "engine_status": "Pre-Graph 极速拦截 + DAG 并行调度架构在线运行中 🚀"
            }

        # ------- 清空上下文工具 -------
        elif action == "call_tool://system/clear_context":
            # 规则层只负责返回模板变量；真正的上下文清理由前端 session 控制
            return {}
            
        # ------- 静态空请求兜底 -------
        elif action == "static_reply":
            return {}
            
    except Exception as exc:
        log.error(f"Rule Engine 执行 {action} 故障: {exc}")
        return {}

    return {}
