# -*- coding: utf-8 -*-
"""
Agent 注册表

该文件作为系统中所有可用 Agent 的中央注册表。
通过将 Agent 的定义与图的构建逻辑分离，使得系统更易于管理和扩展。
要添加新的 Agent，只需在此文件中进行注册即可。
"""
from typing import Dict, List

# 导入所有 Agent 类
from agent.agents.code_agent import CodeAgent
from agent.agents.medical_agent import MedicalAgent
from agent.agents.search_agent import SearchAgent
from agent.agents.sql_agent import SqlAgent
from agent.agents.weather_agent import WeatherAgent
from agent.agents.yunyou_agent import YunyouAgent


class AgentInfo:
    """一个简单的类，用于存储 Agent 的元数据。"""

    def __init__(self, cls, description: str, keywords: List[str]):
        self.cls = cls
        self.description = description
        self.keywords = keywords


# 定义系统中所有 Agent
agent_classes: Dict[str, AgentInfo] = {
    "yunyou_agent": AgentInfo(
        cls=YunyouAgent,
        description="处理云柚相关问题",
        keywords=["云柚", "holter", "智纤", "心电", "审核", "上传", "holter类型"]
    ),
    "medical_agent": AgentInfo(
        cls=MedicalAgent,
        description="回答医疗健康问题，例如症状、药物咨询，输出附带免责声明",
        keywords=["病", "症状", "药", "健康", "血压", "心率"]
    ),
    "code_agent": AgentInfo(
        cls=CodeAgent,
        description="提供编程代码生成、分析、优化服务",
        keywords=["代码", "函数", "bug", "报错", "SQL", "Python", "Java"]
    ),
    "sql_agent": AgentInfo(
        cls=SqlAgent,
        description="处理数据库查询和 SQL 优化问题",
        keywords=["SQL", "数据库", "查询", "表", "索引"]
    ),
    "weather_agent": AgentInfo(
        cls=WeatherAgent,
        description="提供天气查询和预报服务",
        keywords=["天气", "气温", "预报", "降雨"]
    ),
    "search_agent": AgentInfo(
        cls=SearchAgent,
        description="处理通用搜索和信息检索",
        keywords=["搜索", "查找", "信息", "资料"]
    )
}

# 导出所有 Agent 的名称列表，方便在其他地方使用
MEMBERS = list(agent_classes.keys())
