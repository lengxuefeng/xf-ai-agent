# -*- coding: utf-8 -*-
"""
运行时记忆服务（Runtime Memory Service）。

统一管理运行时需要注入的各种记忆信息。
记忆信息包括项目级文档、会话上下文、用户输入等，是系统"记忆"机制的重要组成部分。

设计要点：
1. 结构化管理：不同来源的记忆使用MemorySnippet结构化
2. 分层注入：项目级、会话级、请求级
3. 自动组装：自动收集并组装记忆信息
4. 格式统一：统一的注入格式，便于LLM理解

使用场景：
- 系统提示注入：将项目文档注入到系统提示
- 会话上下文注入：将用户画像、城市等注入
- 请求上下文注入：将当前用户输入注入

记忆层次：
1. 项目级（project）：AGENTS.md等项目文档
2. 会话级（session）：用户画像、城市等会话信息
3. 请求级（request）：当前用户输入
"""
from __future__ import annotations

from typing import Any, Dict, List

from runtime.memory.agents_memory_loader import load_agents_memory
from runtime.memory.memory_models import MemorySnippet
from runtime.types import RunContext


class RuntimeMemoryService:
    """
    统一运行时记忆服务。

    核心职责：
    1. 收集项目级记忆（AGENTS.md等）
    2. 收集会话级记忆（用户画像、城市等）
    3. 收集请求级记忆（当前用户输入）
    4. 渲染为统一的记忆块，供系统提示注入

    设计理由：
    1. 集中管理记忆信息，避免散落各处
    2. 分层管理，不同层级有不同作用域
    3. 自动组装，减少手动拼接
    4. 格式统一，便于LLM理解

    记忆信息来源：
    - project_agents_md: 从AGENTS.md加载的项目文档
    - session_summary: 会话上下文摘要
    - session_slots: 会话槽位信息
    - current_request: 当前用户输入
    """

    def build_memory_snippets(
        self,
        *,
        run_context: RunContext,
        session_context: Dict[str, Any],
    ) -> List[MemorySnippet]:
        """
        构建记忆片段列表。

        设计要点：
        1. 从多个来源收集记忆信息
        2. 使用MemorySnippet结构化
        3. 按层级组织（project/session/request）
        4. 过滤空值，只保留有效信息

        Args:
            run_context: 运行上下文
            session_context: 会话上下文

        Returns:
            List[MemorySnippet]: 记忆片段列表

        收集流程：
        1. 收集项目级记忆（AGENTS.md）
        2. 收集会话级记忆（摘要、槽位）
        3. 收集请求级记忆（当前输入）
        4. 返回完整的记忆片段列表
        """
        snippets: List[MemorySnippet] = []

        # 1. 收集项目级记忆（AGENTS.md）
        project_memory = load_agents_memory()
        if project_memory:
            snippets.append(
                MemorySnippet(
                    key="project_agents_md",
                    source="AGENTS.md",
                    content=project_memory,
                    scope="project",
                )
            )

        # 2. 收集会话级记忆（摘要）
        context_summary = str(session_context.get("context_summary") or "").strip()
        if context_summary:
            snippets.append(
                MemorySnippet(
                    key="session_summary",
                    source="session_state",
                    content=context_summary,
                    scope="session",
                )
            )

        # 3. 收集会话级记忆（槽位）
        context_slots = session_context.get("context_slots") or {}
        if isinstance(context_slots, dict) and context_slots:
            # 过滤空值，只保留有效槽位
            slots_lines = [f"{key}: {value}" for key, value in context_slots.items() if value not in (None, "", [], {})]
            if slots_lines:
                snippets.append(
                    MemorySnippet(
                        key="session_slots",
                        source="session_state",
                        content="\n".join(slots_lines),
                        scope="session",
                    )
                )

        # 4. 收集请求级记忆（当前输入）
        snippets.append(
            MemorySnippet(
                key="current_request",
                source="run_context",
                content=f"当前用户输入：{run_context.user_input}",
                scope="request",
            )
        )

        return snippets

    def render_memory_block(self, snippets: List[MemorySnippet]) -> str:
        """
        将记忆片段渲染为统一的记忆块。

        设计要点：
        1. 过滤空内容的片段
        2. 使用统一的格式
        3. 标注来源和作用域
        4. 便于LLM理解和使用

        Args:
            snippets: 记忆片段列表

        Returns:
            str: 渲染后的记忆块字符串

        输出格式示例：
        【运行时记忆注入】
        - [project] AGENTS.md: 项目级文档内容...
        - [session] session_state: 会话摘要内容...
        - [session] session_state: city: 郑州
        - [request] run_context: 当前用户输入：...

        渲染流程：
        1. 过滤空内容
        2. 构建标题
        3. 遍历片段，添加到列表
        4. 拼接为字符串返回
        """
        # 过滤空内容
        visible = [snippet for snippet in snippets if snippet.content]
        if not visible:
            return ""

        # 构建记忆块
        lines = ["【运行时记忆注入】"]
        for snippet in visible:
            lines.append(f"- [{snippet.scope}] {snippet.source}: {snippet.content}")

        return "\n".join(lines)


# 全局唯一的运行时记忆服务实例
runtime_memory_service = RuntimeMemoryService()

