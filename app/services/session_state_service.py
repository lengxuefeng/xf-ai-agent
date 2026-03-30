# -*- coding: utf-8 -*-
import re
from typing import Any, Dict, Optional, List

from sqlalchemy.orm import Session

from constants.session_state_constants import (
    SESSION_CITY_STOPWORDS,
    SESSION_TOPIC_KEYWORDS,
    SESSION_MAX_KEY_FACTS,
)
from db.crud import session_state_db, user_info_db
from schemas.session_state_schemas import SessionStateCreate
from services.route_metrics_service import route_metrics_service
from utils.custom_logger import get_logger
from utils.location_parser import extract_valid_city_candidate, is_reliable_city_name

log = get_logger(__name__)


class SessionStateService:
    """
    会话状态服务。

    职责：
    1. 维护会话级结构化槽位（城市、姓名、年龄、性别、身高、体重）。
    2. 在每一轮请求前提供可复用上下文，降低重复追问和误路由。
    3. 在每一轮结束后回写路由快照与轮次统计。
    """

    # 画像槽位清单：用于统一摘要与清洗
    PROFILE_SLOT_KEYS = ("name", "age", "gender", "height_cm", "weight_kg")

    def get_or_create_state(self, db: Session, user_id: int, session_id: str):
        """获取会话状态，不存在则创建。"""
        # 先查是否已有会话状态
        state = session_state_db.get_by_session_id(db, session_id)
        if state:
            return state

        # 从用户基础资料里做轻量初始化，避免首次对话完全无上下文
        seed_slots = self._seed_slots_from_user(db, user_id)
        # 生成人类可读摘要，供系统上下文注入
        summary_text = self._build_summary_text(seed_slots)

        create_data = SessionStateCreate(
            user_id=user_id,
            session_id=session_id,
            slots=seed_slots,
            summary_text=summary_text,
            last_route={},
            turn_count=0,
            is_deleted=False,
        )
        return session_state_db.create(db, create_data)

    def build_runtime_context(
        self,
        db: Session,
        user_id: int,
        session_id: str,
        user_input: str,
        history_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        在请求进入图前构造运行时上下文。

        说明：
        1. 先读取已有槽位；
        2. 再从本轮用户输入中抽取增量槽位；
        3. 若槽位变化则立即落库，保证本轮路由即可复用。
        """
        # 读取或创建会话状态记录
        state = self.get_or_create_state(db, user_id, session_id)
        # 标准化已有槽位，确保字段类型稳定
        existed_slots = self._normalize_slots(state.slots)
        # 从本轮输入提取槽位
        extracted_slots = self._extract_slots_from_text(user_input)
        # 若当前无城市，尝试从历史对话回填一次
        if not extracted_slots.get("city") and not existed_slots.get("city"):
            history_city = self._extract_city_from_history(history_messages or [])
            if history_city:
                extracted_slots["city"] = history_city

        # 合并槽位（新值覆盖旧值，空值不覆盖）
        merged_slots = self._merge_slots(existed_slots, extracted_slots)
        # 生成摘要文本，用于注入系统消息
        summary_text = self._build_summary_text(merged_slots)

        # 若状态发生变化，立刻回写，避免同轮路由读不到最新槽位
        if merged_slots != existed_slots or (state.summary_text or "") != summary_text:
            session_state_db.update(
                db,
                state,
                {
                    "slots": merged_slots,
                    "summary_text": summary_text,
                },
            )

        return {
            "context_slots": merged_slots,
            "context_summary": summary_text,
        }

    def update_after_turn(
        self,
        db: Session,
        user_id: int,
        session_id: str,
        user_input: str,
        ai_response: str = "",
        runtime_context: Optional[Dict[str, Any]] = None,
        route_snapshot: Optional[Dict[str, Any]] = None,
    ):
        """在一轮对话结束后回写会话状态（轮次 + 路由快照 + 槽位）。"""
        # 读取会话状态（不存在则创建）
        state = self.get_or_create_state(db, user_id, session_id)
        context_slots = (runtime_context or {}).get("context_slots")
        context_summary = (runtime_context or {}).get("context_summary")
        if isinstance(context_slots, dict):
            merged_slots = self._normalize_slots(context_slots)
            summary_text = str(context_summary or "").strip() or self._build_summary_text(merged_slots)
        else:
            # 读取旧槽位
            existed_slots = self._normalize_slots(state.slots)
            # 从用户输入再提取一遍，兜底补齐漏提取场景
            extracted_slots = self._extract_slots_from_text(user_input)
            # 合并最新槽位
            merged_slots = self._merge_slots(existed_slots, extracted_slots)
            # 构造摘要
            summary_text = self._build_summary_text(merged_slots)
        # 获取最近路由快照，用于后续排障和审计
        last_route = route_snapshot if isinstance(route_snapshot, dict) else route_metrics_service.get_last_route(session_id)
        # 更新轮次计数
        next_turn_count = int(getattr(state, "turn_count", 0) or 0) + 1

        session_state_db.update(
            db,
            state,
            {
                "slots": merged_slots,
                "summary_text": summary_text,
                "last_route": last_route,
                "turn_count": next_turn_count,
            },
        )

    def _seed_slots_from_user(self, db: Session, user_id: int) -> Dict[str, Any]:
        """从用户基础资料中初始化槽位（当前仅姓名相关）。"""
        # 默认空槽位
        seed_slots: Dict[str, Any] = {}
        user = user_info_db.get(db, user_id)
        if not user:
            return seed_slots

        # 优先昵称，其次用户名
        name_value = (getattr(user, "nick_name", "") or "").strip() or (getattr(user, "user_name", "") or "").strip()
        if name_value:
            seed_slots["name"] = name_value
        return seed_slots

    def _extract_slots_from_text(self, text: str) -> Dict[str, Any]:
        """从一段文本中抽取结构化槽位。"""
        # 统一字符串输入
        raw_text = (text or "").strip()
        if not raw_text:
            return {}

        # 小写副本用于英文关键词匹配
        lower_text = raw_text.lower()
        slots: Dict[str, Any] = {}

        # 1) 城市
        city_value = self._extract_city(raw_text)
        if city_value:
            slots["city"] = city_value

        # 2) 姓名
        name_value = self._extract_name(raw_text)
        if name_value:
            slots["name"] = name_value

        # 3) 年龄
        age_value = self._extract_age(raw_text)
        if age_value is not None:
            slots["age"] = age_value

        # 4) 性别
        gender_value = self._extract_gender(raw_text, lower_text)
        if gender_value:
            slots["gender"] = gender_value

        # 5) 身高（单位统一为 cm）
        height_value = self._extract_height_cm(raw_text, lower_text)
        if height_value is not None:
            slots["height_cm"] = height_value

        # 6) 体重（单位统一为 kg）
        weight_value = self._extract_weight_kg(raw_text, lower_text)
        if weight_value is not None:
            slots["weight_kg"] = weight_value

        # 7) 最近主题（用于路由和上下文聚焦）
        topic_value = self._extract_topic(raw_text, lower_text)
        if topic_value:
            slots["last_topic"] = topic_value

        # 8) 用户任务关键信息（目标、约束、偏好等）
        key_facts = self._extract_key_facts(raw_text)
        if key_facts:
            slots["key_facts"] = key_facts

        return slots

    @staticmethod
    def _normalize_slots(slots: Any) -> Dict[str, Any]:
        """把槽位标准化成 dict，避免 None/异常类型污染后续逻辑。"""
        if not isinstance(slots, dict):
            return {}

        # 统一拷贝一份，避免原对象被外部引用修改
        normalized = dict(slots)
        # 清理空字符串
        for key in list(normalized.keys()):
            value = normalized.get(key)
            if isinstance(value, str):
                normalized[key] = value.strip()
                if not normalized[key]:
                    normalized.pop(key, None)
        return normalized

    def _merge_slots(self, old_slots: Dict[str, Any], new_slots: Dict[str, Any]) -> Dict[str, Any]:
        """合并槽位：新值优先，空值不覆盖旧值。"""
        # 先复制旧槽位，避免原地修改
        merged_slots = dict(old_slots or {})
        for key, value in (new_slots or {}).items():
            # 关键事实采用“去重追加”策略，避免每轮覆盖丢信息
            if key == "key_facts":
                old_facts = merged_slots.get("key_facts", [])
                merged_slots["key_facts"] = self._merge_key_facts(old_facts, value)
                continue
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            merged_slots[key] = value
        return merged_slots

    def _build_summary_text(self, slots: Dict[str, Any]) -> str:
        """将槽位渲染为简洁上下文摘要，控制 token 成本。"""
        # 标准化槽位，避免脏数据
        clean_slots = self._normalize_slots(slots)
        if not clean_slots:
            return ""

        # 城市摘要
        city_value = str(clean_slots.get("city", "") or "").strip()
        # 画像摘要片段
        profile_parts: List[str] = []

        name_value = clean_slots.get("name")
        if name_value:
            profile_parts.append(f"姓名={name_value}")

        age_value = clean_slots.get("age")
        if isinstance(age_value, int):
            profile_parts.append(f"年龄={age_value}岁")

        gender_value = clean_slots.get("gender")
        if gender_value:
            # 性别统一展示为中文，便于提示词直接使用
            gender_label = "男" if str(gender_value).lower() == "male" else "女" if str(gender_value).lower() == "female" else str(gender_value)
            profile_parts.append(f"性别={gender_label}")

        height_value = clean_slots.get("height_cm")
        if isinstance(height_value, int):
            profile_parts.append(f"身高={height_value}cm")

        weight_value = clean_slots.get("weight_kg")
        if isinstance(weight_value, (int, float)):
            profile_parts.append(f"体重={float(weight_value):.1f}kg")

        # 任务关键信息摘要（最多展示 3 条，避免上下文过长）
        key_facts = clean_slots.get("key_facts", [])
        fact_preview: List[str] = []
        if isinstance(key_facts, list):
            fact_preview = [str(item).strip() for item in key_facts if str(item).strip()][:3]

        # 组装为系统上下文
        lines: List[str] = ["【会话关键上下文】"]
        if city_value:
            lines.append(f"- 当前城市: {city_value}")
        if profile_parts:
            lines.append(f"- 用户画像: {'，'.join(profile_parts)}")
        if fact_preview:
            lines.append(f"- 近期关键信息: {'；'.join(fact_preview)}")
        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    @staticmethod
    def _extract_topic(raw_text: str, lower_text: str) -> Optional[str]:
        """抽取最近主题标签（weather/search/sql/holter）。"""
        # 遍历主题词典，命中即返回
        for topic_name, keywords in SESSION_TOPIC_KEYWORDS.items():
            if any(str(keyword).lower() in lower_text for keyword in keywords):
                return topic_name
        return None

    def _extract_key_facts(self, raw_text: str) -> List[str]:
        """抽取用户提问中的关键事实（需求/预算/时间/关注地点等）。"""
        facts: List[str] = []
        text = (raw_text or "").strip()
        if not text:
            return facts

        # 1) 提炼主需求（截取第一句，控制长度）
        main_clause = re.split(r"[。！？!?；;\n]", text)[0].strip()
        if len(main_clause) >= 6:
            facts.append(f"需求={self._trim_fact_text(main_clause, limit=60)}")

        # 2) 预算信息（如：预算5000、预算1万、2000元以内）
        budget_match = re.search(r"(预算|价格|花费|金额)\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?(?:万|千|百|元|块)?)", text)
        if budget_match:
            facts.append(f"预算={budget_match.group(2)}")

        # 3) 时间信息（如：今天/明天/最近7天/2026-03-01）
        time_match = re.search(r"(今天|明天|后天|本周|下周|最近\d+天|\d{4}-\d{2}-\d{2})", text)
        if time_match:
            facts.append(f"时间={time_match.group(1)}")

        # 4) 关注地点/对象（如：启迪后面小区、万象城活动）
        entity_match = re.findall(
            r"([\u4e00-\u9fa5A-Za-z0-9]{2,24}(?:小区|商场|万象城|广场|酒店|学校|医院|公园|地铁站|写字楼|社区))",
            text,
        )
        for entity in entity_match[:3]:
            # 清理前缀口语（如“今天在郑州东站万象城” -> “郑州东站万象城”）
            cleaned_entity = re.sub(r"^(?:今天|明天|后天|本周|下周)?(?:在|到|去|位于)?", "", entity).strip()
            if cleaned_entity:
                facts.append(f"关注对象={cleaned_entity}")

        # 去重并截断
        deduped_facts: List[str] = []
        for item in facts:
            cleaned_item = self._trim_fact_text(item, limit=80)
            if cleaned_item and cleaned_item not in deduped_facts:
                deduped_facts.append(cleaned_item)
        return deduped_facts[:SESSION_MAX_KEY_FACTS]

    @staticmethod
    def _trim_fact_text(text: str, limit: int = 80) -> str:
        """裁剪事实文本，避免单条事实过长。"""
        cleaned_text = (text or "").strip()
        if len(cleaned_text) <= limit:
            return cleaned_text
        return cleaned_text[:limit].rstrip() + "..."

    @staticmethod
    def _merge_key_facts(old_facts: Any, new_facts: Any) -> List[str]:
        """合并 key_facts 列表并去重，保留最近内容。"""
        old_list = old_facts if isinstance(old_facts, list) else []
        new_list = new_facts if isinstance(new_facts, list) else []
        merged: List[str] = []
        # 先放历史再放新事实，后者会在去重后保留“最近顺序”
        for item in [*old_list, *new_list]:
            text = str(item).strip()
            if not text:
                continue
            if text in merged:
                merged.remove(text)
            merged.append(text)
        return merged[-SESSION_MAX_KEY_FACTS:]

    def _extract_city(self, text: str) -> Optional[str]:
        """抽取城市槽位。"""
        # 使用统一城市解析器，避免各模块规则不一致
        city_value = extract_valid_city_candidate(text or "")
        # location id 不写入会话城市槽位，只接受自然语言城市名
        if city_value and (not city_value.isdigit()):
            return city_value
        return None

    @staticmethod
    def _extract_name(text: str) -> Optional[str]:
        """抽取姓名槽位。"""
        patterns = (
            r"(?:我叫|我的名字是|叫我|姓名是|名字是)\s*([A-Za-z\u4e00-\u9fa5·•]{2,20})",
        )
        for pattern in patterns:
            matched = re.search(pattern, text)
            if not matched:
                continue
            name_value = (matched.group(1) or "").strip("，,。.!！？? ")
            if not name_value:
                continue
            if any(ch.isdigit() for ch in name_value):
                continue
            return name_value
        return None

    @staticmethod
    def _extract_age(text: str) -> Optional[int]:
        """抽取年龄槽位。"""
        patterns = (
            r"(?:我(?:今)?年|年龄|岁数)\s*(\d{1,3})\s*岁?",
            r"我\s*(\d{1,3})\s*岁",
        )
        for pattern in patterns:
            matched = re.search(pattern, text)
            if not matched:
                continue
            age_value = int(matched.group(1))
            if 1 <= age_value <= 120:
                return age_value

        # 兜底：句子中出现“我”且出现“XX岁”，也认为是本人年龄描述
        if "我" in text:
            fallback = re.search(r"(\d{1,3})\s*岁", text)
            if fallback:
                age_value = int(fallback.group(1))
                if 1 <= age_value <= 120:
                    return age_value
        return None

    @staticmethod
    def _extract_gender(raw_text: str, lower_text: str) -> Optional[str]:
        """抽取性别槽位。返回值统一为 male/female。"""
        # 优先解析“性别:男/女”这种明确表达
        matched = re.search(r"性别\s*[:：]?\s*(男|女)", raw_text)
        if matched:
            return "male" if matched.group(1) == "男" else "female"

        # 常见自然表达
        female_hit = any(k in lower_text for k in ("女", "女生", "女性", "女士", "female", "girl"))
        male_hit = any(k in lower_text for k in ("男", "男生", "男性", "先生", "male", "boy"))
        if female_hit and not male_hit:
            return "female"
        if male_hit and not female_hit:
            return "male"
        return None

    @staticmethod
    def _extract_height_cm(raw_text: str, lower_text: str) -> Optional[int]:
        """抽取身高槽位，统一转为厘米。"""
        # 形如“身高175cm”
        cm_match = re.search(r"(?:身高|高)\s*([1-2]\d{2})\s*(?:cm|厘米)", lower_text)
        if cm_match:
            height_cm = int(cm_match.group(1))
            if 80 <= height_cm <= 250:
                return height_cm

        # 形如“身高1.75m”
        meter_match = re.search(r"(?:身高|高)\s*([1-2](?:\.\d{1,2}))\s*(?:m|米)", lower_text)
        if meter_match:
            height_cm = int(round(float(meter_match.group(1)) * 100))
            if 80 <= height_cm <= 250:
                return height_cm

        # 形如“身高175”
        naked_match = re.search(r"(?:身高|高)\s*([1-2]\d{2})(?!\d)", raw_text)
        if naked_match:
            height_cm = int(naked_match.group(1))
            if 80 <= height_cm <= 250:
                return height_cm
        return None

    @staticmethod
    def _extract_weight_kg(raw_text: str, lower_text: str) -> Optional[float]:
        """抽取体重槽位，统一转为千克。"""
        # 形如“体重65kg”
        kg_match = re.search(r"(?:体重|重)\s*(\d{2,3}(?:\.\d+)?)\s*(?:kg|公斤|千克)", lower_text)
        if kg_match:
            weight_kg = float(kg_match.group(1))
            if 20 <= weight_kg <= 300:
                return round(weight_kg, 1)

        # 形如“体重130斤”
        jin_match = re.search(r"(?:体重|重)\s*(\d{2,3}(?:\.\d+)?)\s*斤", raw_text)
        if jin_match:
            weight_kg = float(jin_match.group(1)) * 0.5
            if 20 <= weight_kg <= 300:
                return round(weight_kg, 1)
        return None

    def _extract_city_from_history(self, history_messages: List[Dict[str, Any]]) -> Optional[str]:
        """从历史消息中回填城市（仅在当前槽位没有城市时使用）。"""
        if not history_messages:
            return None

        # 从最近消息向前扫描，优先用户消息
        for item in reversed(history_messages[-16:]):
            if not isinstance(item, dict):
                continue
            # 先看用户侧文本
            user_text = (item.get("user_content") or "").strip()
            city_value = self._extract_city(user_text)
            if city_value:
                return city_value
            # 再看模型侧文本（如“郑州的实时天气...”）
            model_text = (item.get("model_content") or "").strip()
            matched = re.search(r"([\u4e00-\u9fa5]{2,8})(?:市)?的(?:实时)?天气", model_text)
            if matched:
                candidate = (matched.group(1) or "").strip()
                if is_reliable_city_name(candidate, strict_short_name=False):
                    return candidate
        return None

    @staticmethod
    def _is_valid_city(city_value: str) -> bool:
        """校验城市候选值，避免把普通词误当城市。"""
        value = (city_value or "").strip()
        if not value:
            return False
        # 兼容历史调用：统一委托给 location_parser
        if value in SESSION_CITY_STOPWORDS:
            return False
        return is_reliable_city_name(value, strict_short_name=False)


session_state_service = SessionStateService()
