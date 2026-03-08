# -*- coding: utf-8 -*-
"""
Agent 注册表与发现配置模块。

充当系统中所有可用业务 Agent 的中央管理注册中心。
将 Agent 的抽象定义与执行引擎解耦，使得系统的扩展、组合及管理更加便捷。
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
    """
    负责存储并封装单个业务智能体元数据的基座类。

    Attributes:
        cls: 该 Agent 对应的具体实现类。
        description: 对该 Agent 所支持领域的业务描述，主要用于语义路由。
        keywords: 该 Agent 支持处理的焦点分词或关键词，用于意图匹配与规则降级。
    """

    def __init__(self, cls, description: str, keywords: List[str]):
        """初始化一个智能体元数据对象。"""
        self.cls = cls
        self.description = description
        self.keywords = keywords


# 定义系统中所有 Agent
agent_classes: Dict[str, AgentInfo] = {
    "yunyou_agent": AgentInfo(
        cls=YunyouAgent,
        description="处理云柚医疗设备平台的业务数据查询（Holter 设备、心电报告、审核记录等）",
        keywords=["云柚", "holter", "智纤", "心电", "审核", "上传", "holter类型", "报告"]
    ),
    "medical_agent": AgentInfo(
        cls=MedicalAgent,
        description="回答医疗健康专业问题（症状分析、药物咨询、疾病科普），输出附带免责声明",
        keywords=["病", "症状", "药", "健康", "血压", "心率", "诊断", "治疗"]
    ),
    "code_agent": AgentInfo(
        cls=CodeAgent,
        description="编写、调试、优化 Python 代码，并可执行代码查看运行结果",
        keywords=["代码", "函数", "bug", "报错", "Python", "Java", "编程", "脚本"]
    ),
    "sql_agent": AgentInfo(
        cls=SqlAgent,
        description="根据自然语言生成 SQL 查询语句并执行，需人工审核后才会执行",
        keywords=["SQL", "数据库", "查询", "表", "索引", "select", "insert"]
    ),
    "weather_agent": AgentInfo(
        cls=WeatherAgent,
        description="实时查询指定城市的天气情况和气象预报（需调用天气 API）",
        keywords=["天气", "气温", "预报", "降雨", "温度", "几度"]
    ),
    "search_agent": AgentInfo(
        cls=SearchAgent,
        description="联网搜索最新新闻、实时事件、网页内容（仅用于模型训练数据无法覆盖的实时信息检索）",
        keywords=["搜索", "搜一下", "查一下最新", "新闻", "百度", "谷歌", "最新消息"]
    )
}

# 导出所有 Agent 的名称列表，方便在其他地方使用
MEMBERS = list(agent_classes.keys())
