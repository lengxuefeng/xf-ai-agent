# -*- coding: utf-8 -*-
"""
Agent 注册表与发现配置模块。

充当系统中所有可用业务 Agent 的中央管理注册中心。
将 Agent 的抽象定义与执行引擎解耦，使得系统的扩展、组合及管理更加便捷。

设计要点：
1. 集中管理：所有Agent的元数据集中定义
2. 灵活扩展：添加新Agent只需在此注册
3. 元数据丰富：包含类、描述、关键词等信息
4. 语义路由：支持基于描述和关键词的智能路由

使用场景：
- Supervisor根据用户意图选择Agent
- 前端展示Agent列表和描述
- 关键词匹配用于快速路由
- Agent发现和动态加载

架构说明：
- agent_classes: Agent名称到AgentInfo的映射
- AgentInfo: Agent的元数据封装
- MEMBERS: Agent名称列表，便于遍历
"""
from typing import Dict, List

# 导入所有 Agent 类
from agent.agents.code_agent import CodeAgent
from agent.agents.medical_agent import MedicalAgent
from agent.agents.search_agent import SearchAgent
from agent.agents.sql_agent import SqlAgent
from agent.agents.weather_agent import WeatherAgent
from agent.agents.yunyou_agent import YunyouAgent
from constants.agent_registry_keywords import AGENT_KEYWORDS, AgentKeywordGroup


class AgentInfo:
    """
    负责存储并封装单个业务智能体元数据的基座类。

    核心职责：
    1. 封装Agent的实现类
    2. 提供Agent的业务描述
    3. 提供Agent的关键词列表

    设计理由：
    1. 元数据化，便于路由和发现
    2. 描述用于语义路由，LLM可以理解
    3. 关键词用于规则降级，快速匹配

    Attributes:
        cls: 该 Agent 对应的具体实现类
        description: 对该 Agent 所支持领域的业务描述，主要用于语义路由
        keywords: 该 Agent 支持处理的焦点分词或关键词，用于意图匹配与规则降级

    使用场景：
    - 路由时根据描述选择合适的Agent
    - 前端展示Agent的职责范围
    - 关键词匹配用于快速决策
    """

    def __init__(self, cls, description: str, keywords: List[str]):
        """
        初始化一个智能体元数据对象。

        Args:
            cls: Agent的实现类
            description: Agent的业务描述，用于语义路由
            keywords: Agent的关键词列表，用于规则匹配
        """
        self.cls = cls
        self.description = description
        self.keywords = keywords


# 定义系统中所有 Agent
# 这是Agent注册的核心，所有Agent都必须在这里注册
# 新增Agent时，需要添加对应的导入和注册信息
agent_classes: Dict[str, AgentInfo] = {
    # 云柚专员：处理云柚医疗设备平台的业务数据查询
    "yunyou_agent": AgentInfo(
        cls=YunyouAgent,
        description="处理云柚医疗设备平台的业务数据查询（Holter 设备、心电报告、审核记录等）",
        keywords=list(AGENT_KEYWORDS[AgentKeywordGroup.YUNYOU]),
    ),
    # 医馆参谋：回答医疗健康专业问题
    "medical_agent": AgentInfo(
        cls=MedicalAgent,
        description="回答医疗健康专业问题（症状分析、药物咨询、疾病科普），输出附带免责声明",
        keywords=list(AGENT_KEYWORDS[AgentKeywordGroup.MEDICAL]),
    ),
    # 工坊司：编写、调试、优化 Python 代码
    "code_agent": AgentInfo(
        cls=CodeAgent,
        description="编写、调试、优化 Python 代码，并可执行代码查看运行结果",
        keywords=list(AGENT_KEYWORDS[AgentKeywordGroup.CODE]),
    ),
    # 账房先生：根据自然语言生成 SQL 查询语句并执行
    "sql_agent": AgentInfo(
        cls=SqlAgent,
        description="根据自然语言生成 SQL 查询语句并执行，需人工审核后才会执行",
        keywords=list(AGENT_KEYWORDS[AgentKeywordGroup.SQL]),
    ),
    # 天象司：实时查询指定城市的天气情况和气象预报
    "weather_agent": AgentInfo(
        cls=WeatherAgent,
        description="实时查询指定城市的天气情况和气象预报（需调用天气 API）",
        keywords=list(AGENT_KEYWORDS[AgentKeywordGroup.WEATHER]),
    ),
    # 典籍司：联网搜索最新新闻、实时事件、网页内容
    "search_agent": AgentInfo(
        cls=SearchAgent,
        description="联网搜索最新新闻、实时事件、网页内容（仅用于模型训练数据无法覆盖的实时信息检索）",
        keywords=list(AGENT_KEYWORDS[AgentKeywordGroup.SEARCH]),
    )
}

# 导出所有 Agent 的名称列表，方便在其他地方使用
# 场景：
# - 遍历所有Agent进行检查
# - 前端展示Agent列表
# - 路由器查询可用Agent
MEMBERS = list(agent_classes.keys())
