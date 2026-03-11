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

from agent.base import BaseAgent
from agent.graph_state import AgentRequest
from agent.prompts.yunyou_prompt import YunyouPrompt
from constants.approval_constants import ApprovalDecision, DEFAULT_ALLOWED_DECISIONS
from constants.agent_messages import (
    YUNYOU_REVIEW_DESCRIPTION,
    YUNYOU_REVIEW_INTERRUPT_MESSAGE,
)
from constants.yunyou_keywords import (
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
from agent.tools.yunyou_tools import (
    YunYouTools,
    holter_list,
    holter_recent_db,
    holter_report_count,
    holter_type_count,
)
from config.runtime_settings import AGENT_LOOP_CONFIG, YUNYOU_DB_POOL_CONFIG
from schemas.base_skill import ConfigurableSkillMiddleware
from utils.custom_logger import get_logger

log = get_logger(__name__)

"""
云柚业务相关的子图(Agent)实现。

展示了如何将工具(ToolNode)调用、人工中断与审批(Interrupt)
以及通过状态修剪防止上下文溢出的高阶功能结合。
"""


class YunyouState(TypedDict):
    """云柚业务图状态"""
    messages: Annotated[List[BaseMessage], add_messages]
    tool_loop_count: int


class YunyouAgent(BaseAgent):
    """云柚业务Agent：处理Holter相关查询"""

    SENSITIVE_TOOLS = {"holter_report_count"}  # 需要审批的敏感工具
    TOOL_LOOP_EXCEEDED_MESSAGE = "⚠️ 云柚查询步骤过多，已自动停止本轮工具循环。请缩小问题范围后重试。"
    HOLTER_TYPE_MAP = YUNYOU_HOLTER_TYPE_MAP
    REPORT_STATUS_MAP = YUNYOU_REPORT_STATUS_MAP
    UPLOAD_STATUS_MAP = YUNYOU_UPLOAD_STATUS_MAP

    def __init__(self, req: AgentRequest):
        """
        初始化云柚智能体并创建相关工作流子图。

        Args:
            req (AgentRequest): 用户请求上下文，如包含绑定的语言模型。
        """
        super().__init__(req)
        if not req.model:
            raise ValueError("Yunyou Agent 模型初始化失败。")
        self.llm = req.model
        self.graph = self._build_graph()

    @staticmethod
    def _message_text(msg: BaseMessage) -> str:
        """兼容字符串与 content block 两种消息格式。"""
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
        texts: List[str] = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                text = cls._message_text(msg).strip()
                if text:
                    texts.append(text)
        return texts[-limit:]

    @staticmethod
    def _normalize_date_token(token: str) -> Optional[str]:
        """标准化日期格式为YYYY-MM-DD"""
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
        """计算最近N天的日期范围"""
        n = max(1, min(days, 3650))
        end = datetime.now().date()
        start = end - timedelta(days=n - 1)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    @classmethod
    def _extract_date_range(cls, text: str) -> Optional[Tuple[str, str]]:
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

        lower = t.lower()
        m = re.search(r"(?:最近|近)\s*(\d{1,3})\s*天", lower)
        if m:
            return cls._last_n_days_range(int(m.group(1)))

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
        """提取查询条数限制"""
        t = (text or "").lower()
        for p in YUNYOU_LIMIT_PATTERNS:
            m = re.search(p, t)
            if m:
                return max(1, min(int(m.group(1)), 200))
        return default_limit

    @staticmethod
    def _wants_desc_order(text: str) -> bool:
        """判断是否需要降序排列"""
        t = (text or "").lower()
        if any(k in t for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.ORDER_ASC]):
            return False
        return any(k in t for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.ORDER_DESC])

    @classmethod
    def _is_holter_list_intent(cls, latest_text: str, history_text: str) -> bool:
        """判断是否应命中“Holter 列表直查”快路径。"""
        latest = (latest_text or "").lower().strip()
        history = (history_text or "").lower()
        merged = f"{history} {latest}".strip()
        if not merged:
            return False

        holter_keywords = YUNYOU_KEYWORDS[YunyouKeywordGroup.HOLTER_DOMAIN]
        holter_hit = any(k in merged for k in holter_keywords)

        # 统计类问题交给常规模型流程，避免误判。
        if any(k in latest for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.STATS_BLOCK]):
            return False

        list_keywords = YUNYOU_KEYWORDS[YunyouKeywordGroup.LIST_INTENT]
        list_hit = any(k in latest for k in list_keywords)
        order_hit = any(k in latest for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.ORDER_INTENT])

        # 若当前句没显式说 holter，但历史已经在 holter 语境，且当前是排序/条数补充，也应直查。
        if not holter_hit and history and order_hit and any(k in history for k in holter_keywords):
            holter_hit = True

        # 当前路由已进入 yunyou_agent，且用户明确只说“按 id 倒序前 N 条”这类强列表诉求，
        # 默认按 holter 列表理解，避免重复追问日期或表名。
        if not holter_hit and not history and order_hit:
            holter_hit = True

        if not holter_hit:
            return False

        if list_hit:
            return True

        # 用户补充参数（如“最近7天”）时，沿用上文列表查询意图
        if (
            len(latest) <= 32
            and any(k in latest for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.FOLLOWUP_DATE])
            and any(k in history for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.FOLLOWUP_LIST_HISTORY])
        ):
            return True

        return False

    @staticmethod
    def _extract_filters(text: str) -> Dict[str, Optional[int]]:
        t = (text or "").lower()
        filters: Dict[str, Optional[int]] = {
            "isUploaded": None,
            "reportStatus": None,
            "holterType": None,
        }
        if any(k in t for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.UPLOADED_YES]):
            filters["isUploaded"] = 1
        elif any(k in t for k in YUNYOU_KEYWORDS[YunyouKeywordGroup.UPLOADED_NO]):
            filters["isUploaded"] = 0

        for status_code, keywords in YUNYOU_REPORT_STATUS_FILTERS.items():
            if any(k in t for k in keywords):
                filters["reportStatus"] = status_code
                break

        for holter_type, keywords in YUNYOU_HOLTER_TYPE_FILTERS.items():
            if any(k in t for k in keywords):
                filters["holterType"] = holter_type
                break

        return filters

    @classmethod
    def _extract_records(cls, payload: Any, depth: int = 0) -> List[Dict[str, Any]]:
        if depth > 2:
            return []
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []

        for key in YUNYOU_RECORD_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = cls._extract_records(value, depth + 1)
                if nested:
                    return nested

        if "id" in payload:
            return [payload]

        for value in payload.values():
            if isinstance(value, (dict, list)):
                nested = cls._extract_records(value, depth + 1)
                if nested:
                    return nested
        return []

    @staticmethod
    def _pick_value(row: Dict[str, Any], keys: Sequence[str]) -> Any:
        """从行中提取第一个非空字段值"""
        for k in keys:
            if k in row and row.get(k) is not None and row.get(k) != "":
                return row.get(k)
        return "-"

    @staticmethod
    def _clip_value(value: Any, max_len: int = 36) -> str:
        """截断过长文本"""
        text = str(value)
        if len(text) > max_len:
            return text[: max_len - 1] + "…"
        return text

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
        records = cls._extract_records(raw_result)
        if start_day and end_day:
            range_text = f"`{start_day}` ~ `{end_day}`"
        elif start_day:
            range_text = f"`>= {start_day}`"
        elif end_day:
            range_text = f"`<= {end_day}`"
        else:
            range_text = "全部时间"

        if not records:
            base = [
                "✅ 查询已执行，但未命中符合条件的数据。",
                f"- 时间范围：{range_text}",
                f"- 排序：`id {'DESC' if desc else 'ASC'}`",
            ]
            if default_range_days:
                base.append(f"- 说明：未提供明确时间范围，已按默认最近 {default_range_days} 天查询。")
            return "\n".join(base)

        if any("id" in r for r in records):
            def sort_key(row: Dict[str, Any]) -> int:
                try:
                    return int(row.get("id"))
                except Exception:
                    return -1

            records = sorted(records, key=sort_key, reverse=desc)
        rows = records[:limit]

        columns = YUNYOU_HOLTER_TABLE_COLUMNS
        selected = [
            (title, keys)
            for title, keys in columns
            if any(any(key in row for key in keys) for row in rows)
        ]
        if not selected:
            fallback_keys = [k for k in rows[0].keys() if k not in {"token", "password"}][:6]
            selected = [(k, [k]) for k in fallback_keys]

        lines = [
            "✅ 已直接查询 Holter 数据。",
            f"- 时间范围：{range_text}",
            f"- 排序：`id {'DESC' if desc else 'ASC'}`",
            f"- 返回：`{len(rows)}` 条（上限 `{limit}`）",
        ]
        if default_range_days:
            lines.append(f"- 说明：未提供明确时间范围，已按默认最近 {default_range_days} 天查询。")
        lines.append("")

        header = "| " + " | ".join(title for title, _ in selected) + " |"
        separator = "| " + " | ".join(["---"] * len(selected)) + " |"
        lines.append(header)
        lines.append(separator)

        for row in rows:
            rendered: List[str] = []
            for title, keys in selected:
                value = cls._pick_value(row, keys)
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
        检测最近工具调用是否出现连接/可用性错误。

        目的：
        - 避免工具失败后模型反复重试造成卡顿；
        - 让 Agent 能快速降级给出可读错误提示。
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
        """构建工具失败时的统一用户提示。"""
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
        """识别用户是否明确要求“给 SQL 写法/示例”。"""
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
        根据当前用户意图生成 Holter 查询 SQL 示例。

        注意：该 SQL 为示例模板，真实字段名请以云柚业务库实际结构为准。
        """
        table_name = str(YUNYOU_DB_POOL_CONFIG.holter_table_name or "your_holter_table").strip() or "your_holter_table"
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
            "⚠️ 当前工具不可用，以下是可参考的 Holter SQL 示例：\n\n"
            "```sql\n"
            f"{sql}\n"
            "```\n"
            "\n说明：如果该表名在你的库不存在，请替换为实际 Holter 业务表。"
        )

    def _try_direct_holter_list_query(self, messages: List[BaseMessage]) -> Optional[str]:
        """
        对“Holter 列表 + 最近/按id/前N条”做确定性快路径。
        目标：信息足够时不再反复追问，直接查询并格式化输出。
        """
        human_texts = self._extract_recent_human_texts(messages)
        if not human_texts:
            return None

        latest = human_texts[-1]
        history = " ".join(human_texts[:-1][-4:])
        sql_example_intent = self._wants_sql_example_intent(latest)
        if (not sql_example_intent) and (not self._is_holter_list_intent(latest, history)):
            return None

        # 优先从本轮解析时间范围；若本轮无，则复用最近历史。
        latest_range = self._extract_date_range(latest)
        history_range = self._extract_date_range(history) if history else None
        date_range = latest_range or history_range
        merged = f"{history}\n{latest}".strip()
        default_range_days: Optional[int] = None
        start_day = date_range[0] if date_range else None
        end_day = date_range[1] if date_range else None

        limit = self._extract_limit(merged, default_limit=5)
        desc = self._wants_desc_order(merged)
        filters = self._extract_filters(merged)
        if sql_example_intent:
            return self._build_holter_sql_example(
                start_day=start_day,
                end_day=end_day,
                limit=limit,
                desc=desc,
                filters=filters,
            )

        # 工具查询优先：优先使用 yunyou tool 的接口查询。
        # 接口需要日期；若缺失则给宽窗口兜底。
        if not start_day or not end_day:
            default_range_days = 3650
            fallback_range = self._last_n_days_range(default_range_days)
            start_day = start_day or fallback_range[0]
            end_day = end_day or fallback_range[1]
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
        生成智能体所需的系统提示词 (System Prompt)。

        补充当前系统时间等实时上下文维度，并强化遵守工具调用的指令。

        Args:
            base_prompt (str): 从中间件或默认项获取的基础提示词。

        Returns:
            str: 拼接增强规则后的最终系统提示词。
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_weekday = datetime.now().strftime("%A")
        rules = YunyouPrompt.EXECUTION_RULES.format(current_time, current_weekday)
        return (base_prompt or YunyouPrompt.DEFAULT_SYSTEM_ROLE) + rules

    def _build_graph(self) -> Runnable:
        """
        构建并返回云柚业务的状态图 (StateGraph)。

        整合基础分析工具与自定义技能工具，并配置人工审批的人机回环 (HITL)。

        Returns:
            Runnable: 编译后的 LangGraph 运行图。
        """
        skill_source = os.getenv("SKILL_YUNYOU", "")
        skill_middleware = ConfigurableSkillMiddleware(skill_source) if skill_source else None

        base_tools = [holter_list, holter_recent_db, holter_type_count, holter_report_count]
        skill_tools = skill_middleware.get_tools() if skill_middleware else []
        all_tools = base_tools + skill_tools

        self.model_with_tools = self.llm.bind_tools(all_tools)
        sys_prompt = self._build_system_prompt(skill_middleware.get_prompt() if skill_middleware else YunyouPrompt.DEFAULT_SYSTEM_ROLE)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        def model_node(state: YunyouState):
            """
            模型处理节点。接收并响应最新对话。
            """
            loop_count = int(state.get("tool_loop_count", 0) or 0)
            if loop_count >= AGENT_LOOP_CONFIG.yunyou_max_tool_loops:
                return {
                    "tool_loop_count": loop_count,
                    "messages": [AIMessage(content=self.TOOL_LOOP_EXCEEDED_MESSAGE)],
                }
            # 保留最近多轮消息，避免用户补充参数时丢失上文导致重复追问。
            recent_messages = state["messages"][-12:]
            fast_path_response = self._try_direct_holter_list_query(recent_messages)
            if fast_path_response:
                # 命中确定性查询路径：直接返回格式化结果，避免反复追问。
                return {"tool_loop_count": loop_count + 1, "messages": [AIMessage(content=fast_path_response)]}
            if self._has_recent_tool_failure(recent_messages):
                # 工具已经明确失败，直接收敛输出，避免继续循环调用工具。
                return {
                    "tool_loop_count": loop_count + 1,
                    "messages": [AIMessage(content=self._build_tool_failure_reply())],
                }
            chain = self.prompt | self.model_with_tools
            try:
                response = chain.invoke({"messages": recent_messages})
                return {"tool_loop_count": loop_count + 1, "messages": [response]}
            except Exception as exc:
                # 兜底：模型/工具执行异常时，不让错误冒泡到上层主流程。
                log.exception(f"yunyou model_node 调用失败，执行降级回复: {exc}")
                fallback_response = self._try_direct_holter_list_query(recent_messages)
                if fallback_response:
                    return {"tool_loop_count": loop_count + 1, "messages": [AIMessage(content=fallback_response)]}
                return {
                    "tool_loop_count": loop_count + 1,
                    "messages": [AIMessage(content=self._build_tool_failure_reply())],
                }

        def human_review_node(state: YunyouState):
            """
            人工审批节点。检查生成的消息是否需要调用受保护的敏感工具。
            如果需要调用，使用 LangGraph 原生 interrupt() 挂起等待外部审批。
            """
            last_message = state["messages"][-1]
            if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
                return Command(goto=END)

            sensitive_calls = [tc for tc in last_message.tool_calls if tc["name"] in self.SENSITIVE_TOOLS]
            if not sensitive_calls:
                # 非敏感调用，直接去工具节点
                return Command(goto="tools")

            # 首次进入，需要人工审批，返回中断载荷
            decision_payload = {
                "message": YUNYOU_REVIEW_INTERRUPT_MESSAGE,
                "allowed_decisions": list(DEFAULT_ALLOWED_DECISIONS),
                "action_requests": [{
                    "type": "tool_approval", "name": call["name"], "args": call["args"],
                    "description": YUNYOU_REVIEW_DESCRIPTION, "id": call["id"],
                } for call in sensitive_calls],
            }
            decision = interrupt(decision_payload)
            action = decision.get("action") if isinstance(decision, dict) else decision

            if action == ApprovalDecision.REJECT.value:
                rejection_messages = [
                    ToolMessage(
                        tool_call_id=call["id"], name=call["name"],
                        content="Error: 管理员已拒绝执行该敏感操作。请婉拒用户的请求。"
                    ) for call in last_message.tool_calls
                ]
                return Command(goto="agent", update={"messages": rejection_messages})

            return Command(goto="tools")

        def should_continue(state: YunyouState):
            """决定下一个节点是人机审核还是结束图执行。"""
            last_message = state["messages"][-1]
            if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
                return END
            return "human_review"

        workflow = StateGraph(YunyouState)
        workflow.add_node("agent", model_node)
        workflow.add_node("human_review", human_review_node)
        # 关键配置：工具异常转成 ToolMessage，而不是直接抛异常中断整个子图。
        workflow.add_node("tools", ToolNode(all_tools, handle_tool_errors=True))

        workflow.add_edge(START, "agent")
        # 修改原来的单向边界为条件判断，如果在分析阶段大模型决定不调用任何工具直接作答，直接 END
        # 如果调用了工具，则首先去人工审核节点
        workflow.add_conditional_edges("agent", should_continue, ["human_review", END])
        # Tools 运行结束后，返回 agent 继续让大模型总结结果。避免图强制切断导致无响应。
        workflow.add_edge("tools", "agent")
        return workflow.compile(checkpointer=self.checkpointer)
