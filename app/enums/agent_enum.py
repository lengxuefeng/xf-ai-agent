from enum import Enum

"""
所有Agent枚举配置
"""


class AgentTypeEnum(Enum):
    YUNYOU = ('yunyou_agent', "处理云柚医疗设备平台的业务数据查询")
    MEDICAL = ('medical_agent', "回答医疗健康专业问题")
    CODE = ('code_agent', '代码相关')
    SQL = ('sql_agent', '根据自然语言生成 SQL 查询语句并执行')
    WEATHER = ('weather_agent', "实时查询指定城市的天气情况和气象预报")
    SEARCH = ('search_agent', '联网搜索最新新闻、实时事件、网页内容')
    CHAT = ('CHAT', '大模型直接对话返回')

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message


AGENT_MEMBERS_TUPLE = tuple(agent.code for agent in AgentTypeEnum)

AGENT_MEMBERS_LIST = list(agent.code for agent in AgentTypeEnum)
