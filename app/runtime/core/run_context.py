# -*- coding: utf-8 -*-
"""
运行上下文（Run Context）构建模块。

RunContext是单次运行的统一上下文对象，贯穿整个图执行过程。
它封装了会话信息、用户输入、模型配置、历史消息等关键数据。

设计要点：
1. 统一的上下文对象，避免参数传递混乱
2. 支持会话恢复，run_id中包含恢复标记
3. 包含元数据，方便监控和调试
4. 数据结构稳定，不会因业务变化而频繁修改
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from runtime.types import RunContext


def build_run_id(session_id: str, *, is_resume: bool = False) -> str:
    """
    构造统一 run_id。

    run_id 是单次运行的唯一标识，格式为：session_id:[resume:]uuid
    格式说明：
    - session_id: 会话ID，用于关联同一会话的多次运行
    - resume: 可选标记，表示这是一次恢复运行（审批恢复）
    - uuid: 唯一标识，确保每次运行都有独立的ID

    设计理由：
    1. 包含session_id方便按会话查询历史运行
    2. 包含resume标记便于区分正常运行和恢复运行
    3. 使用UUID保证全局唯一性，避免冲突

    Args:
        session_id: 会话ID，用于关联用户对话历史
        is_resume: 是否为恢复运行（如审批后恢复）

    Returns:
        格式化的run_id字符串

    Examples:
        >>> build_run_id("chat_123")
        "chat_123:a1b2c3d4e5f6g7h8"
        >>> build_run_id("chat_123", is_resume=True)
        "chat_123:resume:a1b2c3d4e5f6g7h8"
    """
    # 标准化session_id，避免空值或前后空格
    normalized_session_id = str(session_id or "").strip() or "anonymous"

    # 恢复运行在ID中加入resume标记，便于后续识别
    if is_resume:
        return f"{normalized_session_id}:resume:{uuid.uuid4().hex}"
    return f"{normalized_session_id}:{uuid.uuid4().hex}"


def build_run_context(
    *,
    session_id: str,
    user_input: str,
    model_config: Optional[Dict[str, Any]] = None,
    history_messages: Optional[List[Dict[str, Any]]] = None,
    session_context: Optional[Dict[str, Any]] = None,
    is_resume: bool = False,
    run_id: str = "",
) -> RunContext:
    """
    根据当前请求构造统一 RunContext。

    RunContext 是整个图执行过程中的核心上下文对象，包含了所有必要的信息。
    这个函数负责从HTTP请求中提取信息，构建标准的RunContext对象。

    设计理由：
    1. 统一构造入口，避免各模块各自构建导致不一致
    2. 参数校验和标准化，确保数据质量
    3. 自动生成缺失字段（如run_id），减少调用方负担
    4. 包含元数据，方便监控和调试

    参数处理说明：
    - session_id: 去除前后空格，空值保留为""
    - user_input: 转为字符串，空值保留为""
    - model_config: 深拷贝，避免修改原始配置
    - history_messages: 保留原引用，记录数量
    - session_context: 提取summary，记录槽位信息

    Args:
        session_id: 会话ID，用于关联用户对话历史
        user_input: 用户输入的文本
        model_config: 模型配置（model, temperature等）
        history_messages: 历史消息列表，用于上下文
        session_context: 会话级上下文（城市、用户画像等）
        is_resume: 是否为恢复运行（如审批后恢复）
        run_id: 可选的run_id，为空则自动生成

    Returns:
        构造好的RunContext对象，包含所有必要信息

    Examples:
        >>> ctx = build_run_context(
        ...     session_id="chat_123",
        ...     user_input="郑州今天天气如何？",
        ...     model_config={"model": "glm-4"},
        ...     history_messages=[{"user_content": "你好"}],
        ...     session_context={"context_summary": "当前城市: 郑州"}
        ... )
        >>> ctx.session_id
        "chat_123"
        >>> ctx.user_input
        "郑州今天天气如何？"
        >>> ctx.history_size
        1
        >>> ctx.context_summary
        "当前城市: 郑州"
    """
    # 标准化会话ID
    effective_session_id = str(session_id or "").strip()

    # 标准化会话上下文，避免None
    effective_context = session_context or {}

    # 标准化历史消息，避免None
    effective_history = history_messages or []

    # 解析或生成run_id
    resolved_run_id = str(run_id or "").strip() or build_run_id(
        effective_session_id,
        is_resume=is_resume,
    )

    # 构建RunContext对象
    return RunContext(
        # 核心标识
        session_id=effective_session_id,
        run_id=resolved_run_id,
        user_input=str(user_input or ""),

        # 模型和上下文配置
        model_config=dict(model_config or {}),  # 深拷贝避免污染原始配置
        history_size=len(effective_history),
        is_resume=is_resume,

        # 会话摘要，用于系统提示注入
        context_summary=str(effective_context.get("context_summary") or ""),

        # 元数据，用于监控和调试
        meta={
            "input_length": len(str(user_input or "")),          # 输入长度
            "has_context_slots": bool(effective_context.get("context_slots")),  # 是否有槽位
            "history_size": len(effective_history),             # 历史消息数
        },
    )

