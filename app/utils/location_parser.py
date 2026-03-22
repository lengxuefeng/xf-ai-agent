# -*- coding: utf-8 -*-
import re

from constants.weather_tool_constants import CITY_INVALID_SUFFIXES, CITY_STOPWORDS

# 城市实体的基础模式
CITY_BASE_PATTERN = r"[\u4e00-\u9fa5]{2,8}(?:市|县|区|州|盟|旗)?"

# 行政后缀列表：用于提升"无后缀城市名"的判定精度
CITY_ADMIN_SUFFIXES = ("市", "县", "区", "州", "盟", "旗")

# 时间/疑问/代词前缀：以这些前缀开头的短词不可能是城市名
# 例如「今天是什」「什么天气」「怎么回事」等被误提取为候选时，直接过滤
_NON_CITY_PREFIXES = (
    "今天", "明天", "后天", "昨天", "前天", "大后天",
    "今年", "明年", "去年", "今晚", "今早",
    "什么", "怎么", "为什", "为何", "如何",
    "哪里", "哪个", "哪些", "哪儿",
    "这里", "那里", "这个", "那个",
    "现在", "最近", "多少", "几个", "几天", "几月", "几点",
    "好的", "好吧", "可以", "没有", "没啥", "没什",
    "是什", "是否", "是不", "能否", "可否",
    "要不", "会不", "有没", "需不",
)

# 噪声词：含以下词的候选直接排除（不依赖前缀位置）
_NON_CITY_NOISE_WORDS = (
    "出去", "去玩", "玩吧", "玩吗", "一下",
    "天气", "今天", "明天", "后天", "昨天",
    "什么", "怎么", "为什", "为何", "如何",
    "哪里", "哪个", "现在", "最近",
    "是什", "是否", "可以", "没有",
    "这个", "那个",
)


def normalize_city_candidate(raw_city: str) -> str:
    """标准化城市候选词，去除口语尾缀和天气描述尾词。"""
    city_text = (raw_city or "").strip()
    if not city_text:
        return ""

    # 去掉标点和空白
    city_text = re.sub(r"[\s,，。.!！？?；;]+", "", city_text)
    # 去掉天气问句尾词
    city_text = re.sub(r"(?:的)?(?:实时)?天气(?:情况|怎么样|如何)?$", "", city_text)
    city_text = re.sub(r"(?:天气)?(?:怎么样|如何)$", "", city_text)
    # 去掉地点后缀（站点/商圈等），尽量收敛到城市名
    city_text = re.sub(r"(东站|西站|南站|北站|火车站|高铁站|机场|地铁站|万象城|广场|商场|小区|社区)$", "", city_text)
    # 去掉常见语气词
    city_text = re.sub(r"[吗呢呀吧嘛么]+$", "", city_text)
    # 去掉口语中的"的"尾巴（如"郑州的"）
    city_text = re.sub(r"的+$", "", city_text)
    return city_text.strip()


def is_reliable_city_name(city_name: str, strict_short_name: bool = True) -> bool:
    """
    判断城市名是否可靠。

    Args:
        city_name: 待校验城市名
        strict_short_name: 对"无行政后缀"的城市名启用长度限制（默认启用）
    """
    candidate = normalize_city_candidate(city_name)
    if not candidate:
        return False
    if len(candidate) < 2 or len(candidate) > 12:
        return False
    if candidate.endswith(CITY_INVALID_SUFFIXES):
        return False
    if candidate.endswith(("的", "玩")):
        return False
    if candidate in CITY_STOPWORDS:
        return False
    # 含噪声词（时间/疑问/动作词）的候选直接排除
    if any(noise in candidate for noise in _NON_CITY_NOISE_WORDS):
        return False
    # 以时间/疑问/代词前缀开头的短词不是城市名（如"今天是什""什么天气""是什么"）
    if any(candidate.startswith(prefix) for prefix in _NON_CITY_PREFIXES):
        return False
    if re.fullmatch(r"[\u4e00-\u9fa5]{2,12}(?:市|县|区|州|盟|旗)?", candidate) is None:
        return False

    # 无行政后缀时，默认限制为短名称，避免整句被误判为地名
    has_admin_suffix = candidate.endswith(CITY_ADMIN_SUFFIXES)
    if strict_short_name and (not has_admin_suffix) and len(candidate) > 4:
        return False
    return True


def extract_valid_city_candidate(text: str) -> str:
    """
    从文本中抽取可用城市名（生产门控版）。

    抽取优先级：
    1. 数字 location id；
    2. 明确位置表达（在/到/位于/来自/住在/去）；
    3. "XX天气"表达；
    4. 纯短词城市兜底（严格长度限制）。
    """
    raw_text = (text or "").strip()
    if not raw_text:
        return ""

    normalized_text = normalize_city_candidate(raw_text)
    if normalized_text.isdigit():
        return normalized_text

    # 明确位置表达
    # pattern1: 在/到/位于/来自/住在 + 城市（非贪婪2-4字）
    #   负向预查排除地名后缀词，防止"在郑州东站"→"郑州东站"
    #   正向预查：城市后接标点/句尾/"的"，或非汉字字符，或已知活动词
    _p_location = (
        rf"(?:在|到|位于|来自|住在)\s*([\u4e00-\u9fa5]{{2,4}}?(?:市|县|区|州|盟|旗)?)"
        rf"(?![\u4e00-\u9fa5]*(?:东站|西站|南站|北站|火车站|高铁站|机场|地铁|东路|西路|南路|北路|大道|街道|小区|广场|商场|社区|附近|周边))"
        rf"(?=工作|生活|学习|出差|旅游|上班|上学|居住|定居|[\s,，。.!！？?；;的]|[^\u4e00-\u9fa5]|$)"
    )
    explicit_patterns = (
        _p_location,
        rf"去\s*([\u4e00-\u9fa5]{{2,4}}?(?:市|县|区|州|盟|旗)?)(?=玩|旅游|出差|办事|[\s,，。.!！？?；;]|$)",
        # XX(时间词)?天气/冷热类问法
        rf"([\u4e00-\u9fa5]{{2,5}}?(?:市|县|区|州|盟|旗)?)(?:今天|明天|后天|昨天|现在|最近)?的?"
        rf"(?:实时)?(?:天气|气温|温度|冷不冷|热不热|下雨|会下雨|要下雨|冷吗|热吗|下雨吗)",
    )
    for pattern in explicit_patterns:
        matched = re.search(pattern, raw_text)
        if not matched:
            continue
        city_value = normalize_city_candidate(matched.group(1))
        if is_reliable_city_name(city_value, strict_short_name=False):
            return city_value

    # 复杂地点短语兜底：如"在郑州东站万象城附近..."，提取前缀城市"郑州"
    complex_location_match = re.search(
        r"(?:在|到|位于|来自|住在|去)\s*([\u4e00-\u9fa5]{2,4})(?:市|县|区)?[\u4e00-\u9fa5]{0,20}(?:附近|周边|后面|里面|那边)",
        raw_text,
    )
    if complex_location_match:
        city_prefix = normalize_city_candidate(complex_location_match.group(1))
        if is_reliable_city_name(city_prefix, strict_short_name=False):
            return city_prefix

    # 纯短词兜底（如"郑州"）
    if is_reliable_city_name(normalized_text, strict_short_name=True):
        return normalized_text

    return ""
