import os
import re
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, List, Optional, Tuple, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from agent.base import BaseAgent
from agent.gateway.federated_query_gateway import federated_query_gateway
from agent.graph_state import AgentRequest
from agent.prompts.yunyou_prompt import YunyouPrompt
from agent.tools.yunyou_tools import (
    YunYouTools,
    holter_list,
    holter_recent_db,
    holter_report_count,
    holter_type_count,
)
from schemas.base_skill import ConfigurableSkillMiddleware
from utils.custom_logger import get_logger

log = get_logger(__name__)

"""
云柚业务相关的子图(Agent)实现。

展示了如何将工具(ToolNode)调用、人工中断与审批(Interrupt)
以及通过状态修剪防止上下文溢出的高阶功能结合。
"""


class YunyouState(TypedDict):
    """
    云柚业务图的状态对象。

    Attributes:
        messages: 对话消息列表，支持通过 add_messages 自动累加。
    """
    messages: Annotated[List[BaseMessage], add_messages]


class YunyouAgent(BaseAgent):
    """
    云柚业务领域智能体，负责处理与云柚数据分析和处理相关的问题。

    通过绑定包含企业敏感操作约束的工具集合来解答问题。
    """

    SENSITIVE_TOOLS = {"holter_report_count"}
    HOLTER_TYPE_MAP = {0: "24小时", 1: "2小时", 2: "24小时(夜间)", 3: "48小时"}
    REPORT_STATUS_MAP = {-1: "无数据", 0: "待审核", 1: "审核中", 2: "人工审核完成", 3: "自动审核完成"}
    UPLOAD_STATUS_MAP = {-1: "无数据", 0: "未上传", 1: "已上传"}

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
        if len(parsed_dates) == 1 and any(k in t for k in ["当天", "当日", "今天"]):
            return parsed_dates[0], parsed_dates[0]

        lower = t.lower()
        m = re.search(r"(?:最近|近)\s*(\d{1,3})\s*天", lower)
        if m:
            return cls._last_n_days_range(int(m.group(1)))

        if any(k in lower for k in ["最近一周", "近一周", "最近7天", "近7天"]):
            return cls._last_n_days_range(7)
        if any(k in lower for k in ["最近30天", "近30天", "最近一个月", "近一个月"]):
            return cls._last_n_days_range(30)
        if any(k in lower for k in ["最近90天", "近90天", "最近三个月", "近三个月"]):
            return cls._last_n_days_range(90)
        if any(k in lower for k in ["最近半年", "近半年"]):
            return cls._last_n_days_range(180)
        if any(k in lower for k in ["最近一年", "近一年"]):
            return cls._last_n_days_range(365)

        now = datetime.now()
        today = now.date()
        if "昨天" in lower:
            y = today - timedelta(days=1)
            return y.strftime("%Y-%m-%d"), y.strftime("%Y-%m-%d")
        if "今天" in lower or "当日" in lower:
            d = today.strftime("%Y-%m-%d")
            return d, d
        if "本周" in lower or "这周" in lower:
            start = today - timedelta(days=today.weekday())
            return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
        if "上周" in lower:
            this_week_start = today - timedelta(days=today.weekday())
            last_week_start = this_week_start - timedelta(days=7)
            last_week_end = this_week_start - timedelta(days=1)
            return last_week_start.strftime("%Y-%m-%d"), last_week_end.strftime("%Y-%m-%d")
        if "本月" in lower or "这个月" in lower:
            month_start = today.replace(day=1)
            return month_start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
        if "上月" in lower or "上个月" in lower:
            month_start = today.replace(day=1)
            last_month_end = month_start - timedelta(days=1)
            last_month_start = last_month_end.replace(day=1)
            return last_month_start.strftime("%Y-%m-%d"), last_month_end.strftime("%Y-%m-%d")

        return None

    @staticmethod
    def _extract_limit(text: str, default_limit: int = 5) -> int:
        t = (text or "").lower()
        patterns = [
            r"(?:前|最后|最新)\s*(\d{1,3})\s*条",
            r"\btop\s*(\d{1,3})\b",
            r"\blimit\s*(\d{1,3})\b",
        ]
        for p in patterns:
            m = re.search(p, t)
            if m:
                return max(1, min(int(m.group(1)), 200))
        return default_limit

    @staticmethod
    def _wants_desc_order(text: str) -> bool:
        t = (text or "").lower()
        if any(k in t for k in ["升序", "asc", "最早"]):
            return False
        return any(k in t for k in ["倒序", "倒叙", "降序", "desc", "最近", "最新", "最后"])

    @classmethod
    def _is_holter_list_intent(cls, latest_text: str, history_text: str) -> bool:
        latest = (latest_text or "").lower().strip()
        history = (history_text or "").lower()
        merged = f"{history} {latest}".strip()
        if not merged:
            return False

        # 必须在当前或上下文中命中 Holter 业务域
        if not any(k in merged for k in ["holter", "云柚", "动态心电", "贴片"]):
            return False

        # 统计类问题交给常规模型流程，避免误判。
        if any(k in latest for k in ["类型统计", "报告统计", "报告状态统计"]):
            return False

        list_keywords = ["列表", "明细", "记录", "用户", "有哪些", "最近", "最新", "最后", "按id", "按 id", "根据id",
                         "根据 id", "id倒", "倒序", "倒叙", "limit", "top", "前"]
        if any(k in latest for k in list_keywords):
            return True

        # 用户补充参数（如“最近7天”）时，沿用上文列表查询意图
        if (
            len(latest) <= 32
            and any(k in latest for k in ["今天", "昨天", "本周", "上周", "本月", "上月", "最近", "近"])
            and any(k in history for k in ["列表", "记录", "用户", "按id", "按 id", "倒序", "倒叙", "limit", "top", "前"])
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
        if any(k in t for k in ["已上传", "上传完成", "上传完"]):
            filters["isUploaded"] = 1
        elif any(k in t for k in ["未上传", "没上传"]):
            filters["isUploaded"] = 0

        if "待审核" in t:
            filters["reportStatus"] = 0
        elif "审核中" in t:
            filters["reportStatus"] = 1
        elif "人工审核完成" in t:
            filters["reportStatus"] = 2
        elif "自动审核完成" in t:
            filters["reportStatus"] = 3

        if any(k in t for k in ["2小时", "两小时"]):
            filters["holterType"] = 1
        elif any(k in t for k in ["48小时"]):
            filters["holterType"] = 3
        elif any(k in t for k in ["夜间", "24小时（夜间）", "24小时(夜间)"]):
            filters["holterType"] = 2
        elif any(k in t for k in ["24小时"]):
            filters["holterType"] = 0

        return filters

    @classmethod
    def _extract_records(cls, payload: Any, depth: int = 0) -> List[Dict[str, Any]]:
        if depth > 2:
            return []
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []

        for key in ["records", "list", "rows", "items", "result", "data"]:
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
    def _pick_value(row: Dict[str, Any], keys: List[str]) -> Any:
        for k in keys:
            if k in row and row.get(k) is not None and row.get(k) != "":
                return row.get(k)
        return "-"

    @staticmethod
    def _clip_value(value: Any, max_len: int = 36) -> str:
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

        columns = [
            ("ID", ["id"]),
            ("用户ID", ["user_id", "userId"]),
            ("用户名", ["nick_name", "nickName", "user_name", "userName"]),
            ("使用日期", ["use_day", "useDay"]),
            ("开始时间", ["begin_date_time", "beginDateTime"]),
            ("结束时间", ["end_date_time", "endDateTime"]),
            ("上传状态", ["is_uploaded", "isUploaded"]),
            ("报告状态", ["report_status", "reportStatus"]),
            ("Holter类型", ["holter_type", "holterType"]),
        ]
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
        if not self._is_holter_list_intent(latest, history):
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
        db_fallback_reason: Optional[str] = None

        def _safe_int(value: Any, default: int = -999) -> int:
            try:
                return int(value)
            except Exception:
                return default

        # 路径1：优先直连云柚业务库（无日期也可查，符合“按 id 倒序最近 N 条”诉求）
        try:
            db_result = federated_query_gateway.query_yunyou_holter_recent(
                limit=limit,
                order_desc=desc,
                start_use_day=start_day,
                end_use_day=end_day,
            )
            rows = self._extract_records(db_result)
            if rows:
                # 本地附加筛选，避免 DB 工具参数过于复杂。
                if filters.get("isUploaded") is not None:
                    target_uploaded = _safe_int(filters["isUploaded"])
                    rows = [r for r in rows if _safe_int(r.get("is_uploaded", -999)) == target_uploaded]
                if filters.get("reportStatus") is not None:
                    target_report_status = _safe_int(filters["reportStatus"])
                    rows = [r for r in rows if _safe_int(r.get("report_status", -999)) == target_report_status]
                if filters.get("holterType") is not None:
                    target_holter_type = _safe_int(filters["holterType"])
                    rows = [r for r in rows if _safe_int(r.get("holter_type", -999)) == target_holter_type]
                db_result = {**db_result, "rows": rows}

            return self._format_holter_rows(
                raw_result=db_result,
                start_day=start_day,
                end_day=end_day,
                limit=limit,
                desc=desc,
                default_range_days=default_range_days,
            )
        except Exception as db_exc:
            log.warning(f"Yunyou DB 直查失败，回退 API: {db_exc}")
            db_fallback_reason = str(db_exc)

        # 路径2：回退到云柚 API（接口需要日期；若缺失则给宽窗口兜底）
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
            if db_fallback_reason:
                formatted += (
                    "\n\n> 说明：已尝试直连云柚业务库查询，但当前不可用，已回退到云柚 API。"
                    "\n> 若希望完全按 SKILL 的数据库表查询，请配置 `YUNYOU_DB_URL` 到云柚业务库。"
                )
            return formatted
        except Exception as exc:
            log.exception(f"Yunyou fast-path 查询失败: {exc}")
            return (
                "❌ 查询 Holter 数据失败，请稍后重试。\n\n"
                f"错误信息：{exc}\n"
                "如果持续失败，请检查云柚服务地址、鉴权配置和接口可用性。"
            )

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
            # 保留最近多轮消息，避免用户补充参数时丢失上文导致重复追问。
            recent_messages = state["messages"][-12:]
            fast_path_response = self._try_direct_holter_list_query(recent_messages)
            if fast_path_response:
                # 命中确定性查询路径：直接返回格式化结果，避免反复追问。
                return {"messages": [AIMessage(content=fast_path_response)]}
            chain = self.prompt | self.model_with_tools
            response = chain.invoke({"messages": recent_messages})
            return {"messages": [response]}

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
                "message": "检测到敏感操作，请审核。",
                "allowed_decisions": ["approve", "reject"],
                "action_requests": [{
                    "type": "tool_approval", "name": call["name"], "args": call["args"],
                    "description": "⚠️ 敏感业务数据操作，需审批。", "id": call["id"],
                } for call in sensitive_calls],
            }
            decision = interrupt(decision_payload)
            action = decision.get("action") if isinstance(decision, dict) else decision

            if action == "reject":
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
        workflow.add_node("tools", ToolNode(all_tools))

        workflow.add_edge(START, "agent")
        # 修改原来的单向边界为条件判断，如果在分析阶段大模型决定不调用任何工具直接作答，直接 END
        # 如果调用了工具，则首先去人工审核节点
        workflow.add_conditional_edges("agent", should_continue, ["human_review", END])
        # Tools 运行结束后，返回 agent 继续让大模型总结结果。避免图强制切断导致无响应。
        workflow.add_edge("tools", "agent")
        return workflow.compile(checkpointer=self.checkpointer)
