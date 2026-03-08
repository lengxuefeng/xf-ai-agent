from datetime import datetime
import json

# ======================
# 1. 模拟配置中心/数据库中的规则（启动时加载到内存）
# ======================
RULES = [
    {
        "intent": "QUERY_DATE",
        "keywords": ["今天几号", "现在日期", "几月几日", "是什么日期"],
        "priority": 10,
        "action": "call_tool://system/date",
        "response_template": """今天是：
- 公历：{gregorian}
- 农历：{lunar}
- 干支：{ganzhi}

需要我帮你查一下今天的黄历宜忌或吉时吗？"""
    },
    {
        "intent": "QUERY_TIME",
        "keywords": ["现在几点", "现在时间", "几点了"],
        "priority": 10,
        "action": "call_tool://system/time",
        "response_template": "现在是 {hour} 点 {minute} 分。"
    }
]

# ======================
# 2. 工具注册中心（模拟系统工具）
# ======================
def get_date_tool():
    """获取日期的工具，返回结构化数据"""
    now = datetime.now()
    return {
        "gregorian": now.strftime("%Y年%m月%d日，%A").replace("Monday", "星期一")
                                               .replace("Tuesday", "星期二")
                                               .replace("Wednesday", "星期三")
                                               .replace("Thursday", "星期四")
                                               .replace("Friday", "星期五")
                                               .replace("Saturday", "星期六")
                                               .replace("Sunday", "星期日"),
        "lunar": "丙午年 正月十六",  # 实际生产中会调用农历库
        "ganzhi": "丙午年 庚寅月 丁丑日"  # 实际生产中会调用干支库
    }

def get_time_tool():
    """获取时间的工具，返回结构化数据"""
    now = datetime.now()
    return {
        "hour": now.hour,
        "minute": now.minute
    }

# 工具映射表
TOOL_REGISTRY = {
    "call_tool://system/date": get_date_tool,
    "call_tool://system/time": get_time_tool
}

# ======================
# 3. 规则引擎核心：匹配 + 执行
# ======================
def preprocess_query(query: str) -> str:
    """预处理用户输入：去掉标点、空格等"""
    import re
    return re.sub(r'[^\w\s]', '', query).strip()

def match_intent(query: str, rules: list) -> dict:
    """
    匹配意图：
    - 遍历规则，统计命中关键词数量
    - 按意图聚合，计算置信度
    - 返回置信度最高的意图
    """
    intent_scores = {}

    for rule in rules:
        for keyword in rule["keywords"]:
            if keyword in query:
                intent = rule["intent"]
                if intent not in intent_scores:
                    intent_scores[intent] = {
                        "score": 0,
                        "priority": rule["priority"],
                        "action": rule["action"],
                        "template": rule["response_template"]
                    }
                intent_scores[intent]["score"] += 1

    if not intent_scores:
        return None

    # 按得分 + 优先级排序，取最高的
    sorted_intents = sorted(
        intent_scores.items(),
        key=lambda x: (x[1]["score"], x[1]["priority"]),
        reverse=True
    )
    best_intent = sorted_intents[0][1]
    return best_intent

def execute_action(action: str) -> dict:
    """执行动作：调用对应的工具"""
    if action not in TOOL_REGISTRY:
        raise ValueError(f"未知的动作: {action}")
    return TOOL_REGISTRY[action]()

def render_response(template: str, data: dict) -> str:
    """渲染响应模板：把工具返回的数据填充到模板里"""
    return template.format(**data)

# ======================
# 4. 主入口：完整流程
# ======================
def rule_engine_main(query: str) -> str:
    # 1. 预处理
    processed_query = preprocess_query(query)
    print(f"预处理后: {processed_query}")

    # 2. 匹配意图
    intent = match_intent(processed_query, RULES)
    if not intent:
        return "规则未命中，进入大模型..."

    print(f"命中意图: {intent}")

    # 3. 执行动作（调用工具）
    tool_data = execute_action(intent["action"])
    print(f"工具返回数据: {json.dumps(tool_data, ensure_ascii=False, indent=2)}")

    # 4. 渲染响应
    response = render_response(intent["template"], tool_data)
    return response

# ======================
# 5. 测试
# ======================
if __name__ == "__main__":
    user_query = "今天几号？是什么日期？昨天是什么时间"
    print(f"用户输入: {user_query}")
    result = rule_engine_main(user_query)
    print("\n最终回答:")
    print(result)