# -*- coding: utf-8 -*-
"""Agent 侧通用提示文案常量。"""

# 搜索工具循环超限提示
SEARCH_TOOL_LOOP_EXCEEDED_MESSAGE = "联网检索调用次数过多，已自动停止以避免卡住。请换个关键词重试。"

# 搜索主题跑偏拦截提示
SEARCH_TOPIC_DRIFT_BLOCK_MESSAGE = (
    "我理解你现在问的是小区/房价位置信息，不是天气。"
    "我将按该主题继续检索。若需要天气，请明确说明“查天气”。"
)

# 搜索工具失败快速收尾提示
SEARCH_TOOL_FAILURE_MESSAGE = (
    "联网检索当前不可用或超时，我已停止重复调用以避免卡住。"
    "请稍后重试，或把问题拆成更短关键词后再试。"
)

# 天气工具循环超限提示
WEATHER_TOOL_LOOP_EXCEEDED_MESSAGE = "天气工具调用次数过多，已停止当前流程。请稍后重试。"

# 云柚审核中断提示
YUNYOU_REVIEW_INTERRUPT_MESSAGE = "检测到敏感操作，请审核。"

# 云柚审核描述
YUNYOU_REVIEW_DESCRIPTION = "⚠️ 敏感业务数据操作，需审批。"

# GraphRunner 拒绝操作提示
GRAPH_RUNNER_REJECTED_MESSAGE = "已拒绝本次敏感操作。"
