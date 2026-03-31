# -*- coding: utf-8 -*-
"""
工作流事件总线（Workflow Event Bus）。

提供工作流事件的构造、解析和格式化功能。
工作流事件用于跟踪和展示Agent的执行流程，是系统可观测性的重要组成部分。

设计要点：
1. 统一的事件格式：所有工作流事件使用相同的结构
2. 时间戳标准化：使用UTC时间戳
3. 角色映射：将Agent名称映射为展示角色（supervisor/worker/system）
4. 名称本地化：将内部Agent名称转换为友好的展示标签

使用场景：
- 前端展示执行流程图和时间线
- 后端记录执行日志
- 调试和排错
- 性能分析

事件结构：
{
    "id": 事件ID（UUID）
    "session_id": 会话ID
    "run_id": 运行ID
    "phase": 阶段名称
    "title": 标题
    "summary": 摘要
    "status": 状态
    "role": 角色（supervisor/worker/system）
    "agent_name": Agent名称
    "agent_label": Agent展示标签
    "task_id": 任务ID
    "node_name": 节点名称
    "timestamp": 时间戳
    "meta": 元数据（可选）
}
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional

from harness.types import utc_now_iso


def workflow_timestamp() -> str:
    """
    返回统一 UTC 时间戳。

    设计理由：
    1. 使用UTC时间戳，避免时区问题
    2. 统一的时间戳格式，方便前端展示
    3. 使用ISO 8601格式，标准且易解析

    Returns:
        str: UTC时间戳字符串

    场景：
    - 构造工作流事件时添加时间戳
    - 日志记录时添加时间戳
    """
    return utc_now_iso()


def workflow_role_for_agent(agent_name: str) -> str:
    """
    根据 Agent 名称推断流程展示中的角色。

    角色说明：
    - supervisor: 管理者，如ChatAgent、Aggregator
    - worker: 执行者，如各专业Agent
    - system: 系统，如系统提示、工具调用

    设计理由：
    1. 前端可以根据角色展示不同的图标和颜色
    2. 帮助用户理解Agent在执行流程中的作用
    3. 简化前端逻辑，不需要复杂的判断

    Args:
        agent_name: Agent名称

    Returns:
        str: 角色名称（supervisor/worker/system）

    场景：
    - 构造工作流事件时设置role字段
    - 前端根据role决定如何展示事件
    """
    normalized = str(agent_name or "").strip()
    # 管理者类Agent
    if normalized in {"ChatAgent", "Aggregator", "chat_node", "aggregator_node"}:
        return "supervisor"
    # 执行者类Agent
    if normalized:
        return "worker"
    # 系统事件
    return "system"


def workflow_display_name(agent_name: str) -> str:
    """
    将内部 Agent 名称转换为前端展示标签。

    设计理由：
    1. 使用中文标签，更符合国风设计
    2. 标签要有特色，体现各Agent的特点
    3. 避免技术术语，用户更容易理解

    命名规则：
    - ChatAgent: "掌柜" - 负责整体对话管理
    - Aggregator: "总管汇总" - 负责汇总各Agent的结果
    - yunyou_agent: "云柚专员" - 处理云柚医疗设备
    - sql_agent: "账房先生" - 处理数据库查询
    - weather_agent: "天象司" - 处理天气信息
    - search_agent: "典籍司" - 处理联网搜索
    - medical_agent: "医馆参谋" - 处理医疗咨询
    - code_agent: "工坊司" - 处理代码执行

    Args:
        agent_name: 内部Agent名称

    Returns:
        str: 前端展示标签

    场景：
    - 构造工作流事件时设置agent_label字段
    - 前端展示Agent名称时使用标签
    """
    normalized = str(agent_name or "").strip()
    display_map = {
        "ChatAgent": "掌柜",
        "Aggregator": "总管汇总",
        "yunyou_agent": "云柚专员",
        "sql_agent": "账房先生",
        "weather_agent": "天象司",
        "search_agent": "典籍司",
        "medical_agent": "医馆参谋",
        "code_agent": "工坊司",
        "chat_node": "掌柜",
        "aggregator_node": "总管汇总",
    }
    return display_map.get(normalized, normalized or "流程节点")


def build_workflow_event(
    *,
    session_id: str,
    run_id: str,
    phase: str,
    title: str,
    summary: str = "",
    status: str = "info",
    role: str = "system",
    agent_name: str = "",
    task_id: str = "",
    node_name: str = "",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    构造统一结构化流程事件。

    设计要点：
    1. 所有事件都有统一的结构
    2. 使用UUID作为事件ID，保证唯一性
    3. 使用UTC时间戳，避免时区问题
    4. 可选字段只在有值时才添加

    参数说明：
    - session_id: 会话ID，关联用户对话
    - run_id: 运行ID，关联单次运行
    - phase: 阶段名称，如"route"、"execute"、"aggregate"
    - title: 标题，如"路由决策"、"执行任务"
    - summary: 摘要，详细描述
    - status: 状态，如"info"、"running"、"completed"、"failed"
    - role: 角色，如"supervisor"、"worker"、"system"
    - agent_name: Agent名称（可选）
    - task_id: 任务ID（可选）
    - node_name: 节点名称（可选）
    - meta: 元数据（可选），用于存储附加信息

    Returns:
        Dict[str, Any]: 结构化的工作流事件

    场景：
    - Agent开始执行时发送事件
    - 路由决策时发送事件
    - 任务完成时发送事件
    - 错误发生时发送事件
    """
    payload: Dict[str, Any] = {
        "id": uuid.uuid4().hex,  # 事件ID
        "session_id": session_id,  # 会话ID
        "run_id": run_id,  # 运行ID
        "phase": phase,  # 阶段
        "title": title,  # 标题
        "summary": summary,  # 摘要
        "status": status,  # 状态
        "role": role,  # 角色
        "timestamp": workflow_timestamp(),  # 时间戳
    }

    # 可选字段：Agent信息
    if agent_name:
        payload["agent_name"] = agent_name
        payload["agent_label"] = workflow_display_name(agent_name)

    # 可选字段：任务信息
    if task_id:
        payload["task_id"] = task_id

    # 可选字段：节点信息
    if node_name:
        payload["node_name"] = node_name

    # 可选字段：元数据
    if meta:
        payload["meta"] = meta

    return payload


def parse_workflow_event_chunk(chunk: str) -> Optional[Dict[str, Any]]:
    """
    从 workflow_event SSE 中提取 payload。

    设计要点：
    1. 只处理workflow_event类型的事件
    2. 解析SSE格式，提取data字段
    3. 返回payload部分，供后续处理

    SSE格式示例：
    event: workflow_event
    data: {"type": "workflow_event", "payload": {...}}

    Args:
        chunk: SSE事件chunk

    Returns:
        Optional[Dict[str, Any]]: 事件payload，如果解析失败则返回None

    场景：
    - GraphRunner处理SSE事件流时提取workflow_event
    - 更新运行状态快照
    - 推送给前端展示
    """
    if not str(chunk or "").startswith("event: workflow_event"):
        return None  # 不是workflow_event，忽略

    try:
        # 查找data行
        data_line = next(
            (line for line in chunk.splitlines() if line.startswith("data: ")),
            None,
        )
        if not data_line:
            return None  # 没有data行，返回None

        # 解析JSON，提取payload
        data = json.loads(data_line[6:])
    except (StopIteration, TypeError, json.JSONDecodeError):
        return None  # 解析失败，返回None

    payload = data.get("payload")
    if isinstance(payload, dict):
        return dict(payload)  # 返回payload的深拷贝

    return None

