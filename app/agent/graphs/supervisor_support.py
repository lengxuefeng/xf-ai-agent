import json
import queue
import re
import threading
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from constants.search_keywords import SEARCH_REAL_ESTATE_KEYWORDS, SEARCH_WEATHER_KEYWORDS
from constants.supervisor_keywords import SUPERVISOR_KEYWORDS, SupervisorKeywordGroup
from enums.agent_enum import AgentTypeEnum


def parse_json_from_text(text: str, *, log=None, preview_limit: int = 800) -> dict:
    """
    从模型文本里提取第一个完整 JSON 对象。

    这里不用简单正则截取，是因为规划器输出里经常会出现嵌套花括号，
    直接做括号配对会更稳。
    """
    start = text.find('{')
    if start == -1:
        preview_text = (text or "")[:preview_limit]
        if len(text or "") > preview_limit:
            preview_text += "...(truncated)"
        if log is not None:
            log.error(f"[parse_json_from_text]未能在文本中找到JSON对象。原始文本预览:\n{preview_text}")
        raise ValueError("[parse_json_from_text]在文本中找不到JSON对象")

    depth = 0
    for index in range(start, len(text)):
        if text[index] == '{':
            depth += 1
        elif text[index] == '}':
            depth -= 1
            if depth == 0:
                json_str = text[start:index + 1]
                json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
                return json.loads(json_str)

    raise ValueError("[parse_json_from_text]JSON中的不平衡大括号")


def build_non_streaming_config(config):
    """复制一份运行配置，并移除 callbacks，避免路由阶段把 token 噪声写进流。"""
    runtime_config = dict(config or {})
    runtime_config.pop("callbacks", None)
    return runtime_config


def _invoke_with_timeout_runner(callable_fn: Callable[..., Any], result_queue: queue.Queue) -> None:
    try:
        result_queue.put(("ok", callable_fn()))
    except Exception as exc:
        result_queue.put(("error", exc))


def invoke_with_timeout(callable_fn, *, timeout_sec: float, timeout_label: str):
    """
    为同步模型调用增加超时保护。

    超时后不等待后台线程自然结束，直接失败回退，避免路由节点被长时间拖住。
    """
    safe_timeout = max(1.0, float(timeout_sec or 1.0))
    result_queue: queue.Queue = queue.Queue(maxsize=1)
    worker = threading.Thread(
        target=_invoke_with_timeout_runner,
        args=(callable_fn, result_queue),
        daemon=True,
        name=f"invoke-timeout-{timeout_label}",
    )
    worker.start()
    try:
        status, payload = result_queue.get(timeout=safe_timeout)
    except queue.Empty as exc:
        raise TimeoutError(f"{timeout_label} 超时: {safe_timeout:.1f}s") from exc

    if status == "error":
        if isinstance(payload, Exception):
            raise payload
        raise RuntimeError(str(payload))
    return payload


def normalize_interrupt_payload(
        value: Any,
        *,
        default_message: str,
        default_allowed_decisions: list[str],
) -> dict:
    """把不同来源的 Interrupt 值整理成统一 payload。"""
    if hasattr(value, "value"):
        return normalize_interrupt_payload(
            getattr(value, "value"),
            default_message=default_message,
            default_allowed_decisions=default_allowed_decisions,
        )

    if isinstance(value, dict):
        payload = dict(value)
    elif hasattr(value, "__dict__"):
        payload = dict(getattr(value, "__dict__", {}))
    else:
        payload = {"message": str(value)}

    payload.setdefault("message", default_message)
    payload.setdefault("allowed_decisions", list(default_allowed_decisions))
    payload.setdefault("action_requests", [])
    return payload


def extract_interrupt_from_snapshot(
        snapshot: Any,
        *,
        default_message: str,
        default_allowed_decisions: list[str],
):
    """从 LangGraph 快照里取出第一条 interrupt，并规整成统一结构。"""
    tasks = getattr(snapshot, "tasks", None) or []
    for task in tasks:
        interrupts = getattr(task, "interrupts", None) or []
        if not interrupts:
            continue
        first_interrupt = interrupts[0]
        payload = getattr(first_interrupt, "value", first_interrupt)
        return normalize_interrupt_payload(
            payload,
            default_message=default_message,
            default_allowed_decisions=default_allowed_decisions,
        )
    return None


def latest_human_message(messages: List[BaseMessage]) -> str:
    """提取最近一条用户消息，并把 block/list 结构规整为纯文本。"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        text = item.get("text") or item.get("content") or ""
                        if isinstance(text, str):
                            parts.append(text)
                return " ".join(parts).strip()
            return str(content or "").strip()
    return ""


def content_to_text(content: Any) -> str:
    """将 LLM content（str / block list / dict）统一转成可展示文本。"""
    if content is None:
        return ""
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
                if isinstance(text, str) and text:
                    parts.append(text)
        if parts:
            return "\n".join(parts)
        return str(content)
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        if isinstance(text, str):
            return text
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def history_requests_location(messages: Optional[List[BaseMessage]]) -> bool:
    """判断最近几轮 AI 是否明确向用户追问了城市信息。"""
    if not messages:
        return False
    city_request_hints = (
        "哪个城市",
        "所在城市",
        "请告诉我城市",
        "请提供城市",
        "城市名",
        "告诉我你在哪个城市",
    )
    for msg in reversed(messages[-6:]):
        if not isinstance(msg, AIMessage):
            continue
        text = content_to_text(getattr(msg, "content", "")).lower()
        if any(hint in text for hint in city_request_hints):
            return True
    return False


def looks_like_location_fragment(text: str) -> bool:
    """识别“南京”“郑州高新区”这类仅补地点的短片段输入。"""
    normalized = (text or "").strip()
    if not normalized or len(normalized) > 40:
        return False
    if any(sep in normalized for sep in ("，", ",", "、", "；", ";", "\n")):
        parts = [part.strip() for part in re.split(r"[，,、；;\n]+", normalized) if part.strip()]
        if not (1 <= len(parts) <= 5):
            return False
        return all(looks_like_location_fragment(part) for part in parts)
    if any(sep in normalized for sep in ("。", "！", "!", "？", "?")):
        return False
    if any(token in normalized for token in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.NON_LOCATION_EMOTION_WORDS]):
        return False

    patterns = SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.FOLLOWUP_LOCATION_REGEX]
    if not any(re.fullmatch(pattern, normalized) for pattern in patterns):
        return False

    if normalized.endswith(("市", "县", "区", "州", "盟", "旗")):
        return True
    return 1 < len(normalized) <= 4


def is_followup_supplement(text: str, messages: Optional[List[BaseMessage]] = None) -> bool:
    """识别用户是否在补充上一轮追问缺失的参数或确认信息。"""
    normalized = (text or "").strip().lower()
    if not normalized:
        return False

    if normalized in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.FOLLOWUP_CONFIRM]:
        return True
    if any(token in normalized for token in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.FOLLOWUP_REFERENCE]):
        return True
    if len(normalized) <= 60 and re.fullmatch(r"[0-9\-\s:~至到/,，.]+", normalized):
        return True
    if len(normalized) <= 80 and re.search(r"\b20\d{2}-\d{2}-\d{2}\b", normalized):
        return True
    if len(normalized) <= 120 and any(
            token in normalized for token in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.FOLLOWUP_ORDER_HINT]
    ):
        return True
    if history_requests_location(messages) and looks_like_location_fragment(normalized):
        return True
    return False


def extract_recent_city_from_history(messages: List[BaseMessage]) -> Optional[str]:
    """从最近上下文里提取城市名，给后续追问补足地点锚点。"""
    if not messages:
        return None

    ignore_keywords = ("附近", "哪里", "什么", "怎么", "为什么", "好玩", "天气", "活动", "房价", "小区", "推荐")
    suffix_particles_pattern = re.compile(r"[的了呀啊呢吧吗嘛～~\s,，。！!？?]+")

    for msg in reversed(messages[-12:]):
        if not isinstance(msg, HumanMessage):
            continue
        text = content_to_text(getattr(msg, "content", "")).strip()
        if not text:
            continue

        direct_match = re.search(r"(?:在|到|去|位于)([\u4e00-\u9fa5]{2,8}(?:市|县|区|州|盟|旗)?)", text)
        if direct_match:
            city = direct_match.group(1).strip()
            if city and not any(token in city for token in ignore_keywords):
                return city

        normalized = suffix_particles_pattern.sub("", text)
        if 1 < len(normalized) <= 6 and looks_like_location_fragment(normalized):
            if not any(token in normalized for token in ignore_keywords):
                return normalized

    for msg in reversed(messages[-12:]):
        if not isinstance(msg, AIMessage):
            continue
        text = content_to_text(getattr(msg, "content", "")).strip()
        if not text:
            continue
        weather_match = re.search(r"([\u4e00-\u9fa5]{2,8})(?:市)?的(?:实时)?天气", text)
        if weather_match:
            city = weather_match.group(1).strip()
            if city and not any(token in city for token in ignore_keywords):
                return city

    return None


def extract_city_from_context_slots(context_slots: Optional[Dict[str, Any]]) -> Optional[str]:
    """从结构化上下文槽位中读取已经确定过的城市。"""
    if not isinstance(context_slots, dict):
        return None
    city_value = context_slots.get("city")
    if isinstance(city_value, str):
        normalized_city = city_value.strip()
        if normalized_city:
            return normalized_city
    return None


def input_has_location_anchor(text: str) -> bool:
    """判断当前输入是否已经显式携带地点，避免重复补注城市。"""
    normalized = (text or "").strip()
    if not normalized:
        return False
    if looks_like_location_fragment(normalized):
        return True
    if re.search(r"(?:在|到|去|位于)\s*[\u4e00-\u9fa5]{2,8}(?:市|县|区|州|盟|旗)?", normalized):
        return True
    if re.search(r"[\u4e00-\u9fa5]{2,8}(?:市|县|区|州|盟|旗)", normalized):
        return True
    return False


def history_hint_intent(messages: List[BaseMessage], latest_user_text: str = "") -> Optional[str]:
    """结合最近上下文推断意图，避免补充输入被错误路由为普通闲聊。"""
    recent = " ".join(
        content_to_text(getattr(msg, "content", "")) for msg in messages[-8:]
        if isinstance(msg, (HumanMessage, AIMessage))
    ).lower()
    latest = (latest_user_text or "").strip().lower()

    if latest:
        if any(token in latest for token in SEARCH_REAL_ESTATE_KEYWORDS) and not any(
                token in latest for token in SEARCH_WEATHER_KEYWORDS):
            return AgentTypeEnum.SEARCH.code
        if any(token in latest for token in SEARCH_WEATHER_KEYWORDS):
            return AgentTypeEnum.WEATHER.code
        if any(token in latest for token in ("要的是", "改成", "换成", "不对", "错了")) and any(
                marker in latest for marker in
                ("java", "python", "javascript", "typescript", "go", "rust", "c++", "c#", "main方法")
        ):
            return AgentTypeEnum.CODE.code

    if any(token in recent for token in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.HISTORY_YUNYOU]):
        return AgentTypeEnum.YUNYOU.code
    if any(token in recent for token in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.HISTORY_SQL]):
        return AgentTypeEnum.SQL.code
    if any(token in recent for token in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.HISTORY_WEATHER]):
        return AgentTypeEnum.WEATHER.code
    if any(token in recent for token in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.HISTORY_SEARCH]):
        return AgentTypeEnum.SEARCH.code
    if "```" in recent or any(token in recent for token in ("public static void main", "class ", "def ", "code_agent")):
        return AgentTypeEnum.CODE.code
    return None


def has_recent_weather_fact(messages: List[BaseMessage]) -> bool:
    """判断最近对话是否已有天气事实，可供建议类追问直接复用。"""
    recent_ai_texts = [
        content_to_text(getattr(msg, "content", ""))
        for msg in messages[-10:]
        if isinstance(msg, AIMessage)
    ]
    if not recent_ai_texts:
        return False
    joined = "\n".join(recent_ai_texts).lower()
    return any(token in joined for token in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.HISTORY_WEATHER_FACT])


def looks_like_weather_reuse_query(text: str) -> bool:
    """识别“基于已有天气结果继续问建议”的追问。"""
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    return any(token in normalized for token in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.WEATHER_REUSE_QUERY])


def wants_weather_refresh(text: str) -> bool:
    """识别用户是否明确要求重新拉取实时天气。"""
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    return any(token in normalized for token in SUPERVISOR_KEYWORDS[SupervisorKeywordGroup.WEATHER_REFRESH_HINT])


def can_reuse_weather_context(messages: List[BaseMessage], latest_user_text: str) -> bool:
    """判断天气追问是否可直接复用历史天气结果，避免重复调用天气查询。"""
    if not looks_like_weather_reuse_query(latest_user_text):
        return False
    if wants_weather_refresh(latest_user_text):
        return False
    return has_recent_weather_fact(messages)
