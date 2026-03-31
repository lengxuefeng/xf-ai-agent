"""
云柚业务 Agent：处理 Holter 相关查询的智能体

核心职责：
- 处理云柚业务相关的 Holter（动态心电）数据查询
- 支持直接列表查询、统计查询、SQL 示例生成
- 敏感工具调用需要人工审批
- 优化查询性能，提供快路径直接查询
- 生成友好的表格展示结果

业务场景示例：
1. 用户问"查询最近 5 条 Holter 数据" → 直接查询并返回表格
2. 用户问"查询最近 7 天的 Holter 数据" → 解析日期范围，查询并返回
3. 用户问"统计 Holter 报告数量" → 调用敏感工具，需要审批
4. 用户问"给一个 Holter 查询 SQL 示例" → 生成 SQL 示例模板
5. 用户问"按 ID 降序前 10 条" → 解析排序和 LIMIT，查询并返回

设计要点：
- 基于 LangGraph 构建多节点子图：agent → human_review → tools → agent
- 快路径优化：识别简单的列表查询，直接调用工具返回结果
- 审批机制：敏感工具（如统计报告数量）需要人工审批
- SQL 示例：工具不可用时生成 SQL 示例
- 失败检测：检测工具失败，快速返回错误提示
- 日期范围解析：支持多种日期格式（相对日期、绝对日期、范围日期）
- 过滤器提取：支持上传状态、报告状态、Holter 类型的过滤
- 结果格式化：友好的 Markdown 表格展示

与其他 Agent 的区别：
- weather_agent：使用天气 API
- search_agent：使用搜索引擎
- sql_agent：使用数据库查询
- code_agent：生成和执行代码
- yunyou_agent：处理云柚业务（Holter）查询，有快路径优化和审批机制

核心功能：
1. 直接列表查询：识别简单查询意图，直接调用工具返回结果
2. 统计查询：调用敏感工具，需要审批
3. SQL 示例生成：工具不可用时提供 SQL 示例
4. 日期范围解析：支持多种日期格式
5. 结果格式化：Markdown 表格展示

快路径优化：
- 识别"查询 Holter 数据"、"最近 N 条"、"按 ID 排序"等简单查询
- 直接调用 YunYouTools.holter_list 工具
- 避免经过 LLM 工具调用链路，提升性能

审批机制：
- SENSITIVE_TOOLS 定义需要审批的工具
- 使用 LangGraph interrupt 挂起执行
- 支持批准、拒绝等决策
- 拒绝后返回友好提示

输出结果：
- 查询结果表格
- 统计结果（需审批）
- SQL 示例
- 错误提示

错误处理：
- 检测工具失败（连接超时、表不存在等）
- 快速返回友好错误提示
- 降级到 SQL 示例
"""

import os
import re
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from supervisor.base import BaseAgent
from supervisor.graph_state import AgentRequest
from prompts.agent_prompts.yunyou_prompt import YunyouPrompt
from config.constants.approval_constants import ApprovalDecision, DEFAULT_ALLOWED_DECISIONS
from config.constants.agent_messages import (
    YUNYOU_REVIEW_DESCRIPTION,
    YUNYOU_REVIEW_INTERRUPT_MESSAGE,
)
from config.constants.yunyou_keywords import (
    YUNYOU_HOLTER_TYPE_MAP,
    YUNYOU_HOLTER_TYPE_FILTERS,
    YUNYOU_HOLTER_TABLE_COLUMNS,
    YUNYOU_KEYWORDS,
    YUNYOU_LIMIT_PATTERNS,
    YUNYOU_RECORD_KEYS,
    YUNYOU_RELATIVE_DATE_KEYWORDS,
    YUNYOU_REPORT_STATUS_FILTERS,
    YUNYOU_REPORT_STATUS_MAP,
    YUNYOU_UPLOAD_STATUS_MAP,
    YunyouKeywordGroup,
)
from tools.agent_tools.yunyou_tools import (
    YunYouTools,
    holter_list,
    holter_recent_db,
    holter_report_count,
    holter_type_count,
    holter_log_info,
)
from config.runtime_settings import AGENT_LOOP_CONFIG, YUNYOU_DB_POOL_CONFIG
from models.schemas.base_skill import ConfigurableSkillMiddleware
from common.utils.custom_logger import get_logger

log = get_logger(__name__)

"""
云柚业务相关的子图(Agent)实现。

展示了如何将工具(ToolNode)调用、人工中断与审批(Interrupt)
以及通过状态修剪防止上下文溢出的高阶功能结合。
"""


class YunyouState(TypedDict):
    """云柚业务图状态

    状态字段说明：
    - messages: 消息列表，包括用户消息、AI 消息、工具消息
    - tool_loop_count: 工具调用次数计数器

    使用场景：
    - 在子图执行过程中维护状态
    - 跨节点传递消息和控制信息
    - 追踪工具调用次数，实现循环控制
    """
    messages: Annotated[List[BaseMessage], add_messages]
    tool_loop_count: int


class YunyouAgent(BaseAgent):
    """云柚业务 Agent：处理 Holter 相关查询

    主要功能：
    - 处理 Holter 数据查询（列表查询、统计查询）
    - 生成 SQL 查询示例
    - 敏感工具调用需要审批
    - 快路径优化（直接查询）
    - 友好的结果格式化

    工作流程：
    1. 接收用户查询
    2. 尝试快路径（识别简单查询，直接调用工具）
    3. 如果不是快路径，调用 LLM 生成响应
    4. 如果需要调用工具，进入审批节点
    5. 审批通过后执行工具
    6. 返回 agent 节点生成最终响应
    7. 格式化结果并返回

    典型使用场景：
    - "查询最近 5 条 Holter 数据"
    - "查询最近 7 天的 Holter 数据"
    - "统计 Holter 报告数量"（需审批）
    - "给一个 Holter 查询 SQL 示例"
    - "按 ID 降序前 10 条"

    特殊功能：
    - 快路径优化：简单查询直接返回，避免 LLM 调用
    - 审批机制：敏感工具需要人工审批
    - SQL 示例：工具不可用时提供 SQL 模板
    - 日期解析：支持多种日期格式
    - 结果格式化：Markdown 表格展示

    敏感工具：
    - holter_report_count：统计报告数量
    - 其他可能涉及敏感数据查询的工具

    快路径触发条件：
    - 明确的列表查询意图（"查询"、"列表"等）
    - 提及 Holter 或在 Holter 上下文中
    - 提供时间范围或排序信息
    - 不是统计类问题
    """

    # 需要审批的敏感工具
    SENSITIVE_TOOLS = {"holter_report_count"}

    # 工具循环超过限制时的提示消息
    TOOL_LOOP_EXCEEDED_MESSAGE = "⚠️ 云柚查询步骤过多，已自动停止本轮工具循环。请缩小问题范围后重试。"

    # 状态映射表
    HOLTER_TYPE_MAP = YUNYOU_HOLTER_TYPE_MAP
    REPORT_STATUS_MAP = YUNYOU_REPORT_STATUS_MAP
    UPLOAD_STATUS_MAP = YUNYOU_UPLOAD_STATUS_MAP

    def __init__(self, req: AgentRequest):
        """
        初始化云柚智能体并创建相关工作流子图

        参数说明：
        - req: Agent 请求对象，包含模型配置、会话信息等

        初始化步骤：
        1. 调用父类 BaseAgent 初始化
        2. 验证模型配置，确保模型可用
        3. 创建 LLM 实例
        4. 构建云柚业务子图

        异常处理：
        - 如果模型未配置，抛出 ValueError 提示检查配置

        示例：
        >>> req = AgentRequest(model=llm, session_id="xxx")
        >>> yunyou_agent = YunyouAgent(req)
        >>> result = yunyou_agent.invoke({"messages": [HumanMessage("查询最近 5 条 Holter 数据")]})
        """
        super().__init__(req)
        if not req.model:
            raise ValueError("Yunyou Agent 模型初始化失败。")
        self.llm = req.model
        self.graph = self._build_graph()

    @staticmethod
    def _message_text(msg: BaseMessage) -> str:
        """
        兼容字符串与 content block 两种消息格式

        设计目的：
        - 统一处理不同类型的消息内容（str、list、dict）
        - 提取纯文本，便于后续处理
        - 避免复杂结构导致的解析问题

        处理逻辑：
        1. 如果 content 是字符串，直接返回
        2. 如果 content 是列表，提取所有文本部分
        3. 如果 content 是其他类型，转为字符串

        参数说明：
        - msg: 消息对象

        返回值：
        - str: 提取的文本内容

        示例：
        >>> msg = AIMessage(content="这是一条消息")
        >>> YunyouAgent._message_text(msg)
        "这是一条消息"
        >>> msg = AIMessage(content=[{"type": "text", "text": "这是一条消息"}])
        >>> YunyouAgent._message_text(msg)
        "这是一条消息"
        """
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if isinstance(text, str):
                        parts.append(text)
            return " ".join(parts)
        return str(content or "")

    @classmethod
    def _extract_recent_human_texts(cls, messages: List[BaseMessage], limit: int = 6) -> List[str]:
        """
        提取最近的人类消息文本

        设计目的：
        - 获取最近用户输入，用于意图识别
        - 支持多轮对话上下文
        - 限制返回数量，避免上下文过长

        参数说明：
        - messages: 消息列表
        - limit: 最多返回的条数，默认 6

        返回值：
        - List[str]: 最近的人类消息文本列表

        示例：
        >>> messages = [
        ...     HumanMessage("查询 Holter 数据"),
        ...     AIMessage("好的"),
        ...     HumanMessage("最近 5 条")
        ... ]
        >>> YunyouAgent._extract_recent_human_texts(messages)
        ["查询 Holter 数据", "最近 5 条"]
        """
        texts: List[str] = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                text = cls._message_text(msg).strip()
                if text:
                    texts.append(text)
        return texts[-limit:]

    @staticmethod
    def _normalize_date_token(token: str) -> Optional[str]:
        """
        标准化日期格式为 YYYY-MM-DD

        设计目的：
        - 统一不同格式的日期输入
        - 支持中文日期（年月日）和符号分隔（-/./）
        - 标准化为 YYYY-MM-DD 格式

        支持格式：
        - 2024年01月01日
        - 2024-01-01
        - 2024/01/01
        - 2024.01.01

        参数说明：
        - token: 日期字符串

        返回值：
        - Optional[str]: 标准化后的日期，如果解析失败返回 None

        示例：
        >>> YunyouAgent._normalize_date_token("2024年01月01日")
        "2024-01-01"
        >>> YunyouAgent._normalize_date_token("2024/01/01")
        "2024-01-01"
        """
        normalized = (
            token.replace("年", "-")
            .replace("月", "-")
            .replace("日", "")
            .replace("/", "-")
            .replace(".", "-")
            .strip()
        )
        try:
            dt = datetime.strptime(normalized, "%Y-%m-%d")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    @staticmethod
    def _last_n_days_range(days: int) -> Tuple[str, str]:
        """
        计算最近 N 天的日期范围

        设计目的：
        - 根据用户输入的"最近 N 天"计算日期范围
        - 用于日期范围查询
        - 限制天数范围（1-3650 天）

        计算逻辑：
        - 结束日期：今天
        - 开始日期：今天 - (N - 1) 天
        - 例如：最近 7 天 = 今天往前 6 天到今天

        参数说明：
        - days: 天数

        返回值：
        - Tuple[str, str]: (开始日期, 结束日期)

        示例：
        >>> # 假设今天是 2024-01-07
        >>> YunyouAgent._last_n_days_range(7)
        ("2024-01-01", "2024-01-07")
        """
        n = max(1, min(days, 3650))
        end = datetime.now().date()
        start = end - timedelta(days=n - 1)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    @classmethod
    def _extract_date_range(cls, text: str) -> Optional[Tuple[str, str]]:
        """
        提取日期范围

        设计目的：
        - 从用户文本中提取日期范围
        - 支持多种日期表达方式
        - 用于查询参数构建

        支持的日期表达方式：
        1. 绝对日期范围：2024-01-01 到 2024-01-07
        2. 相对日期：最近 7 天、本月、上周等
        3. 单日查询：今天、昨天、2024-01-01

        检测逻辑：
        1. 显式日期区间：提取两个日期
        2. "最近 N 天"：计算最近 N 天范围
        3. 相对日期：今天、昨天、本周、上周、本月、上月
        4. 固定范围关键词：最近一周、最近一月、最近三月等

        参数说明：
        - text: 用户文本

        返回值：
        - Optional[Tuple[str, str]]: (开始日期, 结束日期)，如果未检测到返回 None

        示例：
        >>> YunyouAgent._extract_date_range("查询最近 7 天的数据")
        ("2024-01-01", "2024-01-07")
        >>> YunyouAgent._extract_date_range("查询 2024-01-01 到 2024-01-07 的数据")
        ("2024-01-01", "2024-01-07")
        >>> YunyouAgent._extract_date_range("查询今天的数据")
        ("2024-01-07", "2024-01-07")
        """
        t = (text or "").strip()
        if not t:
            return None

        # 显式日期区间：YYYY-MM-DD / YYYY/MM/DD / YYYY.MM.DD / YYYY年MM月DD日
        date_tokens = re.findall(r"(20\d{2}[年\-/\.]\d{1,2}[月\-/\.]\d{1,2}日?)", t)
        parsed_dates = [cls._normalize_date_token(tok) for tok in date_tokens]
        parsed_dates = [d for d in parsed_dates if d]

        if len(parsed_dates) >= 2:
            start, end = parsed_dates[0], parsed_dates[1]
            if start > end:
                start, end = end, start
            return start, end

        if len(parsed_dates) == 1 and any(k in t for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.SAME_DAY]):
            return parsed_dates[0], parsed_dates[0]

        # "最近 N 天"
        lower = t.lower()
        m = re.search(r"(?:最近|近)\s*(\d{1,3})\s*天", lower)
        if m:
            return cls._last_n_days_range(int(m.group(1)))

        # 固定范围关键词
        if any(k in lower for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.RANGE_7_DAYS]):
            return cls._last_n_days_range(7)
        if any(k in lower for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.RANGE_30_DAYS]):
            return cls._last_n_days_range(30)
        if any(k in lower for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.RANGE_90_DAYS]):
            return cls._last_n_days_range(90)
        if any(k in lower for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.RANGE_180_DAYS]):
            return cls._last_n_days_range(180)
        if any(k in lower for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.RANGE_365_DAYS]):
            return cls._last_n_days_range(365)

        # 相对日期
        now = datetime.now()
        today = now.date()

        if any(k in lower for k in YUNYOU_RELATIVE_DATE_KEYWORDS["yesterday"]):
            y = today - timedelta(days=1)
            return y.strftime("%Y-%m-%d"), y.strftime("%Y-%m-%d")

        if any(k in lower for k in YUNYOU_RELATIVE_DATE_KEYWORDS["today"]):
            d = today.strftime("%Y-%m-%d")
            return d, d

        if any(k in lower for k in YUNYOU_RELATIVE_DATE_KEYWORDS["this_week"]):
            start = today - timedelta(days=today.weekday())
            return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

        if any(k in lower for k in YUNYOU_RELATIVE_DATE_KEYWORDS["last_week"]):
            this_week_start = today - timedelta(days=today.weekday())
            last_week_start = this_week_start - timedelta(days=7)
            last_week_end = this_week_start - timedelta(days=1)
            return last_week_start.strftime("%Y-%m-%d"), last_week_end.strftime("%Y-%m-%d")

        if any(k in lower for k in YUNYOU_RELATIVE_DATE_KEYWORDS["this_month"]):
            month_start = today.replace(day=1)
            return month_start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

        if any(k in lower for k in YUNYOU_RELATIVE_DATE_KEYWORDS["last_month"]):
            month_start = today.replace(day=1)
            last_month_end = month_start - timedelta(days=1)
            last_month_start = last_month_end.replace(day=1)
            return last_month_start.strftime("%Y-%m-%d"), last_month_end.strftime("%Y-%m-%d")

        return None

    @staticmethod
    def _extract_limit(text: str, default_limit: int = 5) -> int:
        """
        提取查询条数限制

        设计目的：
        - 从用户文本中提取查询的条数限制
        - 用于 SQL 的 LIMIT 子句
        - 限制最大条数（200），防止查询过多数据

        检测模式：
        - "前 N 条" → N
        - "查询 N 条" → N
        - "N 个" → N
        - 等等（定义在 YUNYOU_LIMIT_PATTERNS 中）

        参数说明：
        - text: 用户文本
        - default_limit: 默认条数，当未检测到时使用

        返回值：
        - int: 提取到的条数（1-200）

        示例：
        >>> YunyouAgent._extract_limit("查询前 10 条记录")
        10
        >>> YunyouAgent._extract_limit("查询所有记录", default_limit=5)
        5
        """
        t = (text or "").lower()
        for p in YUNYOU_LIMIT_PATTERNS:
            m = re.search(p, t)
            if m:
                return max(1, min(int(m.group(1)), 200))
        return default_limit

    @staticmethod
    def _wants_desc_order(text: str) -> bool:
        """
        判断是否需要降序排列

        设计目的：
        - 检测用户是否要求降序排序
        - 用于 SQL 的 ORDER BY 子句
        - 默认为降序（DESC）

        检测逻辑：
        - 如果用户明确要求升序（"升序"、"从小到大"等），返回 False
        - 如果用户要求降序（"降序"、"从大到小"、"倒序"等）或未指定，返回 True

        参数说明：
        - text: 用户文本

        返回值：
        - bool: True 表示降序，False 表示升序

        示例：
        >>> YunyouAgent._wants_desc_order("按 ID 降序")
        True
        >>> YunyouAgent._wants_desc_order("按 ID 升序")
        False
        >>> YunyouAgent._wants_desc_order("按 ID 排序")
        True  # 默认降序
        """
        t = (text or "").lower()
        if any(k in t for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.ORDER_ASC]):
            return False
        return any(k in t for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.ORDER_DESC])

    @classmethod
    def _is_holter_list_intent(cls, latest_text: str, history_text: str) -> bool:
        """
        判断是否应命中"Holter 列表直查"快路径

        设计目的：
        - 识别简单的列表查询意图
        - 触发快路径，直接调用工具返回结果
        - 避免经过 LLM 工具调用链路，提升性能

        判断条件：
        1. 提及 Holter 或在 Holter 上下文中
        2. 有明确的列表查询意图（"查询"、"列表"、"显示"等）
        3. 不是统计类问题（"统计"、"数量"、"多少"等）
        4. 可能包含排序或 LIMIT 信息

        特殊情况：
        - 上下文补充：如果历史已经在 Holter 语境，当前只是补充排序或 LIMIT，也应触发
        - 默认理解：如果只说"按 ID 倒序前 N 条"，默认理解为 Holter 列表查询

        参数说明：
        - latest_text: 最新用户消息
        - history_text: 历史用户消息

        返回值：
        - bool: True 表示应该触发快路径，False 表示使用常规模型流程

        示例：
        >>> YunyouAgent._is_holter_list_intent("查询最近 5 条 Holter 数据", "")
        True
        >>> YunyouAgent._is_holter_list_intent("按 ID 倒序前 10 条", "查询 Holter 数据")
        True
        >>> YunyouAgent._is_holter_list_intent("统计 Holter 报告数量", "")
        False  # 统计类问题不触发快路径
        """
        latest = (latest_text or "").lower().strip()
        history = (history_text or "").lower()
        merged = f"{history} {latest}".strip()
        if not merged:
            return False

        holter_keywords = YUNYOU_KEYWORDS[YunyouKeywordGroup.HOLTER_DOMAIN]
        holter_hit = any(k in merged for k in holter_keywords)

        # 统计类问题交给常规模型流程，避免误判
        if any(k in latest for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.STATS_BLOCK]):
            return False

        list_keywords = YUNYOU_KEYWORDS[YunyouKeywordGroup.LIST_INTENT]
        list_hit = any(k in latest for k in list_keywords)
        order_hit = any(k in latest for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.ORDER_INTENT])

        # 若当前句没显式说 holter，但历史已经在 holter 语境，且当前是排序/条数补充，也应直查
        if not holter_hit and history and order_hit and any(k in history for k in holter_keywords):
            holter_hit = True

        # 当前路由已进入 yunyou_agent，且用户明确只说"按 id 倒序前 N 条"这类强列表诉求，
        # 默认按 holter 列表理解，避免重复追问日期或表名
        if not holter_hit and not history and order_hit:
            holter_hit = True

        if not holter_hit:
            return False

        if list_hit:
            return True

        # 用户补充参数（如"最近7天"）时，沿用上文列表查询意图
        if (
            len(latest) <= 32
            and any(k in latest for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.FOLLOWUP_DATE])
            and any(k in history for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.FOLLOWUP_LIST_HISTORY])
        ):
            return True

        return False

    @staticmethod
    def _extract_filters(text: str) -> Dict[str, Optional[int]]:
        """
        提取查询过滤器

        设计目的：
        - 从用户文本中提取查询条件
        - 支持上传状态、报告状态、Holter 类型的过滤
        - 用于查询参数构建

        支持的过滤器：
        1. 上传状态：已上传（1）、未上传（0）
        2. 报告状态：待审核、已审核、已驳回等
        3. Holter 类型：常规 Holter、无线 Holter 等

        参数说明：
        - text: 用户文本

        返回值：
        - Dict[str, Optional[int]]: 过滤器字典，包含以下字段：
          - isUploaded: 上传状态（1 或 0）
          - reportStatus: 报告状态（状态码）
          - holterType: Holter 类型（类型码）

        示例：
        >>> YunyouAgent._extract_filters("查询已上传的 Holter 数据")
        {"isUploaded": 1, "reportStatus": None, "holterType": None}
        >>> YunyouAgent._extract_filters("查询常规 Holter 数据")
        {"isUploaded": None, "reportStatus": None, "holterType": 1}  # 假设 1 是常规 Holter
        """
        t = (text or "").lower()
        filters: Dict[str, Optional[int]] = {
            "isUploaded": None,
            "reportStatus": None,
            "holterType": None,
        }

        # 上传状态
        if any(k in t for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.UPLOADED_YES]):
            filters["isUploaded"] = 1
        elif any(k in t for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.UPLOADED_NO]):
            filters["isUploaded"] = 0

        # 报告状态
        for status_code, keywords in YUNYOU_REPORT_STATUS_FILTERS.items():
            if any(k in t for k in keywords):
                filters["reportStatus"] = status_code
                break

        # Holter 类型
        for holter_type, keywords in YUNYOU_HOLTER_TYPE_FILTERS.items():
            if any(k in t for k in keywords):
                filters["holterType"] = holter_type
                break

        return filters

    @classmethod
    def _extract_records(cls, payload: Any, depth: int = 0) -> List[Dict[str, Any]]:
        """
        从工具返回的复杂结构中提取记录列表

        设计目的：
        - 工具返回的可能是嵌套结构，需要提取实际的记录列表
        - 递归查找，支持多层嵌套
        - 避免无限递归（depth > 2）

        查找策略：
        1. 如果 payload 是列表，返回其中的字典项
        2. 如果 payload 是字典，检查常用键名（data、records、list、items 等）
        3. 如果找到列表，返回其中的字典项
        4. 如果找不到，递归查找嵌套结构
        5. 如果 payload 本身包含 id 字段，返回 [payload]

        参数说明：
        - payload: 工具返回的原始数据
        - depth: 递归深度，防止无限递归

        返回值：
        - List[Dict[str, Any]]: 提取的记录列表

        示例：
        >>> payload = {"data": {"records": [{"id": 1}, {"id": 2}]}}
        >>> YunyouAgent._extract_records(payload)
        [{"id": 1}, {"id": 2}]
        """
        if depth > 2:
            return []

        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]

        if not isinstance(payload, dict):
            return []

        # 检查常用键名
        for key in YUNYOU_RECORD_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = cls._extract_records(value, depth + 1)
                if nested:
                    return nested

        # 如果本身包含 id，可能是单条记录
        if "id" in payload:
            return [payload]

        # 递归查找其他嵌套结构
        for value in payload.values():
            if isinstance(value, (dict, list)):
                nested = cls._extract_records(value, depth + 1)
                if nested:
                    return nested

        return []

    @staticmethod
    def _pick_value(row: Dict[str, Any], keys: Sequence[str]) -> Any:
        """
        从行中提取第一个非空字段值

        设计目的：
        - 支持字段别名，按优先级提取值
        - 避免空值和空字符串
        - 如果都为空，返回 "-"

        使用场景：
        - 字段可能有多个名称（如 user_id 和 userId）
        - 按优先级查找第一个非空值

        参数说明：
        - row: 数据行
        - keys: 字段名列表（按优先级排序）

        返回值：
        - Any: 第一个非空值，如果都为空返回 "-"

        示例：
        >>> row = {"userId": 123, "user_id": None}
        >>> YunyouAgent._pick_value(row, ["user_id", "userId"])
        123
        >>> row = {"userId": None, "user_id": None}
        >>> YunyouAgent._pick_value(row, ["user_id", "userId"])
        "-"
        """
        for k in keys:
            if k in row and row.get(k) is not None and row.get(k) != "":
                return row.get(k)
        return "-"

    @staticmethod
    def _clip_value(value: Any, max_len: int = 36) -> str:
        """
        截断过长文本

        设计目的：
        - 限制单元格显示长度
        - 避免表格被长文本撑开
        - 超长文本显示前 N-1 个字符 + "…"

        参数说明：
        - value: 原始值
        - max_len: 最大长度，默认 36

        返回值：
        - str: 截断后的文本

        示例：
        >>> YunyouAgent._clip_value("这是一段很长的文本", max_len=10)
        "这是一段很长…"
        >>> YunyouAgent._clip_value("短文本", max_len=10)
        "短文本"
        """
        text = str(value)
        if len(text) > max_len:
            return text[: max_len - 1] + "…"
        return text

    @staticmethod
    def _holter_row_sort_key(row: Dict[str, Any]) -> int:
        """
        获取 Holter 行的排序键（ID）

        设计目的：
        - 用于排序 Holter 记录
        - 按 ID 排序，确保结果有序

        参数说明：
        - row: 数据行

        返回值：
        - int: ID 值，如果无法解析返回 -1

        示例：
        >>> YunyouAgent._holter_row_sort_key({"id": 123})
        123
        """
        try:
            return int(row.get("id"))
        except Exception:
            return -1

    @classmethod
    def _format_holter_rows(
        cls,
        raw_result: Any,
        start_day: Optional[str],
        end_day: Optional[str],
        limit: int,
        desc: bool,
        default_range_days: Optional[int],
    ) -> str:
        """
        格式化 Holter 查询结果为 Markdown 表格

        设计目的：
        - 将查询结果格式化为友好的表格
        - 显示查询范围、排序、返回条数等信息
        - 截断过长文本，避免表格过宽
        - 将状态码转为可读文本

        格式说明：
        - 头部：查询信息（时间范围、排序、返回条数）
        - 表格：Markdown 表格格式
        - 列：自动选择可用列（优先显示关键字段）
        - 状态映射：将状态码转为可读文本（上传状态、报告状态、Holter 类型）

        参数说明：
        - raw_result: 工具返回的原始数据
        - start_day: 开始日期
        - end_day: 结束日期
        - limit: 返回条数
        - desc: 是否降序
        - default_range_days: 默认天数范围（如使用）

        返回值：
        - str: 格式化后的 Markdown 表格

        示例：
        >>> raw_result = [{"id": 1, "user_id": 123, "use_day": "2024-01-01"}]
        >>> cls._format_holter_rows(raw_result, "2024-01-01", "2024-01-01", 5, True, None)
        "✅ 已直接查询 Holter 数据。\\n- 时间范围：`2024-01-01` ~ `2024-01-01`\\n- 排序：`id DESC`\\n- 返回：`1` 条（上限 `5`）\\n\\n| ID | User ID | Use Day |\\n|---|---|---|\\n| 1 | 123 | 2024-01-01 |"
        """
        records = cls._extract_records(raw_result)

        # 构建时间范围文本
        if start_day and end_day:
            range_text = f"`{start_day}` ~ `{end_day}`"
        elif start_day:
            range_text = f"`>= {start_day}`"
        elif end_day:
            range_text = f"`<= {end_day}`"
        else:
            range_text = "全部时间"

        # 如果没有记录
        if not records:
            base = [
                "✅ 查询已执行，但未命中符合条件的数据。",
                f"- 时间范围：{range_text}",
                f"- 排序：`id {'DESC' if desc else 'ASC'}`",
            ]
            if default_range_days:
                base.append(f"- 说明：未提供明确时间范围，已按默认最近 {default_range_days} 天查询。")
            return "\n".join(base)

        # 按 ID 排序
        if any("id" in r for r in records):
            records = sorted(records, key=cls._holter_row_sort_key, reverse=desc)
        rows = records[:limit]

        # 选择要显示的列
        columns = YUNYOU_HOLTER_TABLE_COLUMNS
        selected = [
            (title, keys)
            for title, keys in columns
            if any(any(key in row for key in keys) for row in rows)
        ]
        if not selected:
            fallback_keys = [k for k in rows[0].keys() if k not in {"token", "password"}][:6]
            selected = [(k, [k]) for k in fallback_keys]

        # 构建头部信息
        lines = [
            "✅ 已直接查询 Holter 数据。",
            f"- 时间范围：{range_text}",
            f"- 排序：`id {'DESC' if desc else 'ASC'}`",
            f"- 返回：`{len(rows)}` 条（上限 `{limit}`）",
        ]
        if default_range_days:
            lines.append(f"- 说明：未提供明确时间范围，已按默认最近 {default_range_days} 天查询。")
        lines.append("")

        # 构建表格
        header = "| " + " | ".join(title for title, _ in selected) + " |"
        separator = "| " + " | ".join(["---"] * len(selected)) + " |"
        lines.append(header)
        lines.append(separator)

        # 填充表格内容
        for row in rows:
            rendered: List[str] = []
            for title, keys in selected:
                value = cls._pick_value(row, keys)

                # 状态映射
                if title == "上传状态":
                    try:
                        value = cls.UPLOAD_STATUS_MAP.get(int(value), value)
                    except Exception:
                        pass
                elif title == "报告状态":
                    try:
                        value = cls.REPORT_STATUS_MAP.get(int(value), value)
                    except Exception:
                        pass
                elif title == "Holter类型":
                    try:
                        value = cls.HOLTER_TYPE_MAP.get(int(value), value)
                    except Exception:
                        pass

                rendered.append(cls._clip_value(value).replace("|", "\\|"))
            lines.append("| " + " | ".join(rendered) + " |")

        return "\n".join(lines)

    @staticmethod
    def _has_recent_tool_failure(messages: List[BaseMessage]) -> bool:
        """
        检测最近工具调用是否出现连接/可用性错误

        设计目的：
        - 避免工具失败后模型反复重试造成卡顿
        - 让 Agent 能快速降级给出可读错误提示
        - 检测数据库连接失败、表不存在等错误

        检测逻辑：
        - 检查最近 6 条消息中的 ToolMessage
        - 提取工具返回的文本内容
        - 检测失败关键词

        失败关键词：
        - connection error / connection failed / 连接失败 / 连接超时
        - timed out / timeout
        - does not exist / undefinedtable / 未找到可用
        - 查询时发生未知错误

        参数说明：
        - messages: 消息列表

        返回值：
        - bool: True 表示检测到工具失败，False 表示未检测到

        示例：
        >>> messages = [
        ...     HumanMessage("查询 Holter 数据"),
        ...     AIMessage(content="", tool_calls=[{"name": "holter_list"}]),
        ...     ToolMessage(content="connection failed")
        ... ]
        >>> YunyouAgent._has_recent_tool_failure(messages)
        True
        """
        failure_keywords = (
            "connection error",
            "connection failed",
            "连接失败",
            "连接超时",
            "timed out",
            "timeout",
            "does not exist",
            "undefinedtable",
            "未找到可用",
            "查询时发生未知错误",
        )

        for msg in messages[-6:]:
            if not isinstance(msg, ToolMessage):
                continue

            content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
            lower_text = content.lower()

            if any(keyword in lower_text for keyword in failure_keywords):
                return True

        return False

    @staticmethod
    def _build_tool_failure_reply(error_detail: str = "") -> str:
        """
        构建工具失败时的统一用户提示

        设计目的：
        - 提供友好的错误提示
        - 给出检查建议
        - 引导用户排查问题

        参数说明：
        - error_detail: 错误详情（可选）

        返回值：
        - str: 格式化的错误提示

        示例：
        >>> YunyouAgent._build_tool_failure_reply("connection failed")
        "❌ 云柚数据服务暂时不可用，当前无法完成本次查询。\\n\\n建议检查：\\n- `YUNYOU_DB_URL` 是否指向云柚业务库\\n- `YUNYOU_HOLTER_TABLE` 是否为真实存在的表名\\n- `YY_BASE_URL` 对应接口是否可访问\\n请稍后重试。\\n\\n错误摘要：connection failed"
        """
        detail_line = f"\n\n错误摘要：{error_detail}" if error_detail else ""
        return (
            "❌ 云柚数据服务暂时不可用，当前无法完成本次查询。\n\n"
            "建议检查：\n"
            "- `YUNYOU_DB_URL` 是否指向云柚业务库\n"
            "- `YUNYOU_HOLTER_TABLE` 是否为真实存在的表名\n"
            "- `YY_BASE_URL` 对应接口是否可访问\n"
            f"请稍后重试。{detail_line}"
        )

    @staticmethod
    def _wants_sql_example_intent(text: str) -> bool:
        """
        识别用户是否明确要求"给 SQL 写法/示例"

        设计目的：
        - 当工具不可用时，提供 SQL 示例作为降级方案
        - 避免用户无法获得任何有用信息
        - 引导用户使用 SQL 直接查询

        检测关键词：
        - SQL、sql
        - 写法
        - 示例
        - 语句
        - 查询语句

        参数说明：
        - text: 用户文本

        返回值：
        - bool: True 表示要求 SQL 示例，False 表示不需要

        示例：
        >>> YunyouAgent._wants_sql_example_intent("给一个 Holter 查询 SQL 示例")
        True
        >>> YunyouAgent._wants_sql_example_intent("查询 Holter 数据")
        False
        """
        lower_text = (text or "").strip().lower()
        if not lower_text:
            return False
        keywords = YUNYOU_KEYWORDS[YunyouKeywordGroup.SQL_EXAMPLE_INTENT]
        return any(keyword in lower_text for keyword in keywords)

    @classmethod
    def _build_holter_sql_example(
        cls,
        *,
        start_day: Optional[str],
        end_day: Optional[str],
        limit: int,
        desc: bool,
        filters: Dict[str, Optional[int]],
    ) -> str:
        """
        根据当前用户意图生成 Holter 查询 SQL 示例

        设计目的：
        - 当工具不可用时，提供 SQL 示例作为降级方案
        - 根据用户意图生成带过滤条件的 SQL
        - 引导用户使用 SQL 直接查询数据库

        SQL 结构：
        - SELECT 主要字段
        - FROM holter_table
        - WHERE 条件（时间范围、上传状态、报告状态、Holter 类型）
        - ORDER BY id DESC/ASC
        - LIMIT N

        注意：
        - 该 SQL 为示例模板，真实字段名请以云柚业务库实际结构为准
        - 如果表名不存在，需要替换为实际表名

        参数说明：
        - start_day: 开始日期
        - end_day: 结束日期
        - limit: 返回条数
        - desc: 是否降序
        - filters: 过滤器

        返回值：
        - str: SQL 示例

        示例：
        >>> cls._build_holter_sql_example(
        ...     start_day="2024-01-01",
        ...     end_day="2024-01-07",
        ...     limit=5,
        ...     desc=True,
        ...     filters={}
        ... )
        "⚠️ 当前工具不可用，我先给你一条可参考的 Holter SQL。\\n\\n```\\nSELECT\\n  id,\\n  user_id,\\n  use_day,\\n  begin_date_time,\\n  end_date_time,\\n  is_uploaded,\\n  report_status,\\n  holter_type,\\n  add_time,\\n  update_time\\nFROM your_holter_table\\nWHERE use_day BETWEEN '2024-01-01' AND '2024-01-07'\\nORDER BY id DESC\\nLIMIT 5;\\n```\\n\\n说明：如果该表名在你的库不存在，请替换为实际 Holter 业务表。"
        """
        table_name = str(YUNYOU_DB_POOL_CONFIG.holter_table_name or "your_holter_table").strip() or "your_holter_table"

        # 构建 WHERE 条件
        where_clauses: List[str] = []
        if start_day and end_day:
            where_clauses.append(f"use_day BETWEEN '{start_day}' AND '{end_day}'")
        elif start_day:
            where_clauses.append(f"use_day >= '{start_day}'")
        elif end_day:
            where_clauses.append(f"use_day <= '{end_day}'")

        if filters.get("isUploaded") is not None:
            where_clauses.append(f"is_uploaded = {int(filters['isUploaded'])}")
        if filters.get("reportStatus") is not None:
            where_clauses.append(f"report_status = {int(filters['reportStatus'])}")
        if filters.get("holterType") is not None:
            where_clauses.append(f"holter_type = {int(filters['holterType'])}")

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + "\n  AND ".join(where_clauses)

        order_sql = "DESC" if desc else "ASC"

        sql = (
            "SELECT\n"
            "  id,\n"
            "  user_id,\n"
            "  use_day,\n"
            "  begin_date_time,\n"
            "  end_date_time,\n"
            "  is_uploaded,\n"
            "  report_status,\n"
            "  holter_type,\n"
            "  add_time,\n"
            "  update_time\n"
            f"FROM {table_name}\n"
            f"{where_sql}\n"
            f"ORDER BY id {order_sql}\n"
            f"LIMIT {max(1, int(limit))};"
        )

        return (
            "⚠️ 当前工具不可用，我先给你一条可参考的 Holter SQL。\n\n"
            "```\n"
            f"{sql}\n"
            "```\n\n"
            "说明：如果该表名在你的库不存在，请替换为实际 Holter 业务表。"
        )

    def _try_direct_holter_list_query(self, messages: List[BaseMessage]) -> Optional[str]:
        """
        对"Holter 列表 + 最近/按id/前N条"做确定性快路径

        设计目的：
        - 信息足够时不再反复追问，直接查询并格式化输出
        - 避免 LLM 工具调用链路，提升性能
        - 快速响应用户简单查询

        触发条件：
        1. 用户明确要求 SQL 示例 → 生成 SQL 示例
        2. 识别为 Holter 列表查询 → 直接查询工具并格式化结果

        处理流程：
        1. 提取最近的人类消息
        2. 检查是否要求 SQL 示例
        3. 检查是否为 Holter 列表查询
        4. 解析日期范围（优先从最新消息，其次从历史）
        5. 解析 LIMIT、排序、过滤器
        6. 如果要求 SQL 示例，生成 SQL
        7. 否则，调用 YunYouTools.holter_list 工具
        8. 格式化结果并返回

        异常处理：
        - 工具调用失败时，返回友好错误提示
        - 降级到 SQL 示例（如果用户要求）

        参数说明：
        - messages: 消息列表

        返回值：
        - Optional[str]: 格式化的查询结果，如果不触发快路径返回 None

        示例：
        >>> messages = [HumanMessage("查询最近 5 条 Holter 数据")]
        >>> yunyou_agent._try_direct_holter_list_query(messages)
        "✅ 已直接查询 Holter 数据。\\n..."
        """
        human_texts = self._extract_recent_human_texts(messages)
        if not human_texts:
            return None

        latest = human_texts[-1]
        history = " ".join(human_texts[:-1][-4:])

        # 检查是否要求 SQL 示例或 Holter 列表查询
        sql_example_intent = self._wants_sql_example_intent(latest)
        if (not sql_example_intent) and (not self._is_holter_list_intent(latest, history)):
            return None

        # 解析日期范围
        latest_range = self._extract_date_range(latest)
        history_range = self._extract_date_range(history) if history else None
        date_range = latest_range or history_range
        merged = f"{history}\n{latest}".strip()

        default_range_days: Optional[int] = None
        start_day = date_range[0] if date_range else None
        end_day = date_range[1] if date_range else None

        # 解析 LIMIT、排序、过滤器
        limit = self._extract_limit(merged, default_limit=5)
        desc = self._wants_desc_order(merged)
        filters = self._extract_filters(merged)

        # 如果要求 SQL 示例，生成 SQL
        if sql_example_intent:
            return self._build_holter_sql_example(
                start_day=start_day,
                end_day=end_day,
                limit=limit,
                desc=desc,
                filters=filters,
            )

        # 工具查询：优先使用 yunyou tool 的接口查询
        # 接口需要日期；若缺失则给宽窗口兜底（10 年）
        if not start_day or not end_day:
            default_range_days = 3650
            fallback_range = self._last_n_days_range(default_range_days)
            start_day = start_day or fallback_range[0]
            end_day = end_day or fallback_range[1]

        # 构建查询参数
        params = {
            "startUseDay": start_day,
            "endUseDay": end_day,
            "isUploaded": filters.get("isUploaded"),
            "reportStatus": filters.get("reportStatus"),
            "holterType": filters.get("holterType"),
        }

        try:
            raw_result = YunYouTools().common_post("holter/list", params)
            formatted = self._format_holter_rows(
                raw_result=raw_result,
                start_day=start_day,
                end_day=end_day,
                limit=limit,
                desc=desc,
                default_range_days=default_range_days,
            )
            return formatted
        except Exception as exc:
            log.warning(f"Yunyou tool 查询失败，返回可读错误并提示重试: {exc}")
            error_preview = self._clip_value(exc, max_len=240)
            return self._build_tool_failure_reply(error_preview)

    def _build_system_prompt(self, base_prompt: str) -> str:
        """
        生成智能体所需的系统提示词

        设计目的：
        - 补充当前系统时间等实时上下文维度
        - 强化遵守工具调用的指令
        - 提供执行规则

        补充信息：
        - 当前系统时间
        - 当前星期
        - 执行规则（工具调用规则、输出格式等）

        参数说明：
        - base_prompt: 基础提示词

        返回值：
        - str: 拼接增强规则后的最终系统提示词

        示例：
        >>> YunyouAgent()._build_system_prompt("你是一个云柚业务助手")
        "你是一个云柚业务助手\\n\\n执行规则：\\n- 当前系统时间：2024-01-07 10:30:00\\n- 当前星期：Sunday\\n..."
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_weekday = datetime.now().strftime("%A")
        rules = YunyouPrompt.EXECUTION_RULES.format(current_time, current_weekday)
        return (base_prompt or YunyouPrompt.DEFAULT_SYSTEM_ROLE) + rules

    async def _model_node(self, state: YunyouState):
        """
        模型处理节点：接收并响应最新对话

        设计目的：
        - 尝试快路径（直接查询）
        - 检查工具循环次数
        - 检测工具失败
        - 调用 LLM 生成响应或工具调用

        处理流程：
        1. 检查工具循环次数，超过上限则终止
        2. 尝试快路径查询
        3. 检测最近工具是否失败
        4. 调用 LLM 生成响应
        5. 如果 LLM 调用失败，降级到快路径或错误提示

        参数说明：
        - state: 子图状态

        返回值：
        - dict: 更新后的状态，包含 tool_loop_count 和 messages

        异常处理：
        - LLM 调用失败时，尝试快路径降级
        - 快路径也失败时，返回友好错误提示
        """
        loop_count = int(state.get("tool_loop_count", 0) or 0)

        # 检查是否超过最大工具调用次数
        if loop_count >= AGENT_LOOP_CONFIG.yunyou_max_tool_loops:
            return {
                "tool_loop_count": loop_count,
                "messages": [AIMessage(content=self.TOOL_LOOP_EXCEEDED_MESSAGE)],
            }

        recent_messages = state["messages"][-12:]

        # 尝试快路径查询
        fast_path_response = self._try_direct_holter_list_query(recent_messages)
        if fast_path_response:
            return {"tool_loop_count": loop_count + 1, "messages": [AIMessage(content=fast_path_response)]}

        # 检测最近工具是否失败
        if self._has_recent_tool_failure(recent_messages):
            return {
                "tool_loop_count": loop_count + 1,
                "messages": [AIMessage(content=self._build_tool_failure_reply())],
            }

        # 调用 LLM
        chain = self.prompt | self.model_with_tools
        try:
            response = await chain.ainvoke({"messages": recent_messages})
            return {"tool_loop_count": loop_count + 1, "messages": [response]}
        except Exception as exc:
            log.exception(f"yunyou model_node 调用失败，执行降级回复: {exc}")
            fallback_response = self._try_direct_holter_list_query(recent_messages)
            if fallback_response:
                return {"tool_loop_count": loop_count + 1, "messages": [AIMessage(content=fallback_response)]}
            return {
                "tool_loop_count": loop_count + 1,
                "messages": [AIMessage(content=self._build_tool_failure_reply())],
            }

    def _human_review_node(self, state: YunyouState):
        """
        人工审批节点：检查是否需要调用敏感工具

        设计目的：
        - 检查生成的消息是否需要调用受保护的敏感工具
        - 如果需要调用，使用 LangGraph interrupt 挂起等待外部审批
        - 支持批准、拒绝等决策

        审批流程：
        1. 获取最新 AI 消息
        2. 检查是否有工具调用
        3. 筛选敏感工具调用
        4. 如果没有敏感工具调用，直接进入 tools 节点
        5. 如果有敏感工具调用，挂起执行等待审批
        6. 根据审批决策：
           - 批准：进入 tools 节点
           - 拒绝：返回 agent 节点，并返回拒绝消息

        敏感工具：
        - holter_report_count：统计报告数量

        参数说明：
        - state: 子图状态

        返回值：
        - Command: 控制流命令，指定下一个节点

        示例：
        >>> state = {"messages": [AIMessage(content="", tool_calls=[{"name": "holter_report_count"}])]}
        >>> yunyou_agent._human_review_node(state)
        # 挂起执行，等待审批
        """
        last_message = state["messages"][-1]

        # 如果没有工具调用，直接结束
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return Command(goto=END)

        # 筛选敏感工具调用
        sensitive_calls = [tc for tc in last_message.tool_calls if tc["name"] in self.SENSITIVE_TOOLS]
        if not sensitive_calls:
            return Command(goto="tools")

        # 构建审批请求数据
        decision_payload = {
            "message": YUNYOU_REVIEW_INTERRUPT_MESSAGE,
            "allowed_decisions": list(DEFAULT_ALLOWED_DECISIONS),
            "action_requests": [{
                "type": "tool_approval",
                "name": call["name"],
                "args": call["args"],
                "description": YUNYOU_REVIEW_DESCRIPTION,
                "id": call["id"],
            } for call in sensitive_calls],
        }

        # 挂起执行等待审批
        decision = interrupt(decision_payload)
        action = decision.get("action") if isinstance(decision, dict) else decision

        # 处理审批决策
        if action == ApprovalDecision.REJECT.value:
            rejection_messages = [
                ToolMessage(
                    tool_call_id=call["id"],
                    name=call["name"],
                    content="Error: 管理员已拒绝执行该敏感操作。请婉拒用户的请求。"
                )
                for call in last_message.tool_calls
            ]
            return Command(goto="agent", update={"messages": rejection_messages})

        return Command(goto="tools")

    @staticmethod
    def _should_continue(state: YunyouState):
        """
        决定下一个节点是人机审核还是结束图执行

        设计目的：
        - 条件路由函数
        - 根据是否有工具调用决定路由

        路由规则：
        - 如果 AI 消息没有工具调用 → END（直接返回）
        - 如果 AI 消息有工具调用 → human_review（进入审批节点）

        参数说明：
        - state: 子图状态

        返回值：
        - str: 下一个节点名称或 "END"

        示例：
        >>> state = {"messages": [AIMessage(content="好的")]}
        >>> YunyouAgent._should_continue(state)
        "END"
        >>> state = {"messages": [AIMessage(content="", tool_calls=[{"name": "holter_list"}])]}
        >>> YunyouAgent._should_continue(state)
        "human_review"
        """
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return END
        return "human_review"

    def _build_graph(self) -> Runnable:
        """
        构建并返回云柚业务的状态图

        设计目的：
        - 整合基础分析工具与自定义技能工具
        - 配置人工审批的人机回环 (HITL)
        - 支持快路径和常规模型流程

        子图结构：
        1. agent：模型节点，生成响应或工具调用
        2. human_review：审批节点，检查敏感工具调用
        3. tools：工具节点，执行工具调用

        流程：
        - START → agent
        - agent → (条件判断) → human_review 或 END
        - human_review → tools 或 agent
        - tools → agent

        特殊配置：
        - ToolNode 使用 handle_tool_errors=True，工具异常转为 ToolMessage
        - 使用全局 checkpointer 支持中断恢复
        - 支持自定义技能工具（通过 ConfigurableSkillMiddleware）

        返回值：
        - Runnable: 编译后的 LangGraph 运行图
        """
        # 加载自定义技能工具
        skill_source = os.getenv("SKILL_YUNYOU", "")
        skill_middleware = ConfigurableSkillMiddleware(skill_source) if skill_source else None

        # 基础工具 + 自定义技能工具
        base_tools = [holter_list, holter_recent_db, holter_type_count, holter_report_count, holter_log_info]
        skill_tools = skill_middleware.get_tools() if skill_middleware else []
        all_tools = base_tools + skill_tools

        # 绑定工具到模型
        self.model_with_tools = self.llm.bind_tools(all_tools)

        # 构建系统提示词
        sys_prompt = self._build_system_prompt(skill_middleware.get_prompt() if skill_middleware else YunyouPrompt.DEFAULT_SYSTEM_ROLE)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        # 构建子图
        workflow = StateGraph(YunyouState)
        workflow.add_node("agent", self._model_node, retry_policy=self.RETRY_POLICY)
        workflow.add_node("human_review", self._human_review_node, retry_policy=self.RETRY_POLICY)

        # 关键配置：工具异常转成 ToolMessage，而不是直接抛异常中断整个子图
        workflow.add_node("tools", ToolNode(all_tools, handle_tool_errors=True), retry_policy=self.RETRY_POLICY)

        workflow.add_edge(START, "agent")

        # 条件判断：如果在分析阶段大模型决定不调用任何工具直接作答，直接 END
        # 如果调用了工具，则首先去人工审核节点
        workflow.add_conditional_edges("agent", self._should_continue, ["human_review", END])

        # Tools 运行结束后，返回 agent 继续让大模型总结结果。避免图强制切断导致无响应。
        workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=self.checkpointer)
