# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from common.utils.date_utils import get_agent_date_context
from common.utils.history_compressor import compress_history_messages
from config.runtime_settings import AGENT_LOOP_CONFIG
from harness.context.context_budget import ContextBudget
from harness.context.governor import runtime_context_governor
from harness.memory.memory_models import MemorySnippet
from harness.memory.memory_service import runtime_memory_service
from harness.types import RunContext
from prompts.runtime_prompts.prompt_cache_adapter import RuntimePromptBlock
from prompts.runtime_prompts.system_prompt_registry import system_prompt_registry
from tools.runtime_tools.tool_registry import runtime_tool_registry


class RuntimeContextBuilder:
    """统一上下文构建器。"""

    @staticmethod
    def _message_text(msg: BaseMessage) -> str:
        """提取消息文本内容，兼容字符串和多模态结构。"""
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part)
        if isinstance(content, dict):
            text = content.get("text") or content.get("content") or ""
            if isinstance(text, str):
                return text
        return str(content or "")

    @staticmethod
    def _extract_text_tokens(text: str) -> set[str]:
        """提取轻量关键词用于相关性筛选。"""
        if not text:
            return set()
        min_chars = AGENT_LOOP_CONFIG.context_relevance_min_token_chars
        zh_tokens = re.findall(rf"[\u4e00-\u9fa5]{{{min_chars},}}", text)
        en_tokens = re.findall(rf"[A-Za-z0-9_]{{{min_chars},}}", text.lower())
        return set(zh_tokens + en_tokens)

    @staticmethod
    def _collect_bound_tool_names(model_config: Dict[str, Any]) -> List[str]:
        requested: list[str] = []
        runtime_skills = model_config.get("runtime_skills") or []
        for skill in runtime_skills:
            if not isinstance(skill, dict):
                continue
            if skill.get("is_active") is False:
                continue
            for raw_tool_name in skill.get("bound_tools") or []:
                normalized = str(raw_tool_name or "").strip()
                if normalized:
                    requested.append(normalized)

        if requested:
            return runtime_tool_registry.normalize_bound_tools(requested)

        fallback_tools = model_config.get("allowed_builtin_tools") or []
        return runtime_tool_registry.normalize_bound_tools(fallback_tools)

    def _filter_prompt_tool_catalog(self, model_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        tool_catalog = [
            dict(item)
            for item in (model_config.get("resolved_tool_catalog") or [])
            if isinstance(item, dict)
        ]
        requested_tools = self._collect_bound_tool_names(model_config)
        if not requested_tools:
            return tool_catalog

        requested_set = set(requested_tools)
        filtered_catalog: list[dict] = []
        for item in tool_catalog:
            visible_name = str(item.get("name") or "").strip()
            if not visible_name:
                continue
            candidate_names = {visible_name}
            normalized_visible = runtime_tool_registry.normalize_bound_tools([visible_name])
            if normalized_visible:
                candidate_names.add(normalized_visible[0])
            if candidate_names & requested_set:
                filtered_catalog.append(item)
        return filtered_catalog

    def _filter_relevant_history(
        self,
        history_messages: List[Dict[str, Any]],
        user_input: str,
    ) -> Tuple[List[BaseMessage], int]:
        """按“最近窗口 + 相关性”筛选历史，降低无关上下文 token。"""
        raw_history: List[BaseMessage] = []
        for msg in history_messages or []:
            if msg.get("user_content"):
                raw_history.append(HumanMessage(content=msg["user_content"]))
            if msg.get("model_content"):
                raw_history.append(AIMessage(content=msg["model_content"], name=msg.get("name")))

        total_raw_count = len(raw_history)
        if not raw_history:
            return raw_history, total_raw_count

        max_window = max(1, AGENT_LOOP_CONFIG.context_history_messages)
        recent_history = raw_history[-max_window:]

        current_tokens = self._extract_text_tokens((user_input or "").strip())
        tail_window = max(
            1,
            min(
                AGENT_LOOP_CONFIG.context_relevance_tail_messages,
                max_window,
            ),
        )

        tail_part = recent_history[-tail_window:]
        head_part = recent_history[:-tail_window]
        filtered_history: list[BaseMessage] = []

        if current_tokens:
            for message in head_part:
                message_tokens = self._extract_text_tokens(self._message_text(message))
                if message_tokens & current_tokens:
                    filtered_history.append(message)

        for message in tail_part:
            if isinstance(message, HumanMessage):
                filtered_history.append(message)
                continue
            message_tokens = self._extract_text_tokens(self._message_text(message))
            if (not current_tokens) or (message_tokens & current_tokens):
                filtered_history.append(message)

        if not any(isinstance(message, HumanMessage) for message in filtered_history):
            for message in reversed(recent_history):
                if isinstance(message, HumanMessage):
                    filtered_history.append(message)
                    break

        return filtered_history, total_raw_count

    def build_messages(
        self,
        *,
        run_context: RunContext,
        history_messages: List[Dict[str, Any]],
        session_context: Dict[str, Any],
        max_tokens: int,
        max_chars: int,
    ) -> Tuple[List[BaseMessage], List[MemorySnippet], Dict[str, Any]]:
        memory_snippets = runtime_memory_service.build_memory_snippets(
            run_context=run_context,
            session_context=session_context,
        )
        memory_block = runtime_memory_service.render_memory_block(memory_snippets)
        prompt_tool_catalog = self._filter_prompt_tool_catalog(run_context.model_config)
        dynamic_blocks = [
            get_agent_date_context(),
            run_context.context_summary,
            (
                f"当前绑定工作目录: {run_context.model_config.get('workspace_root')}"
                if str(run_context.model_config.get("workspace_root") or "").strip()
                else ""
            ),
        ]

        filtered_history, total_raw_count = self._filter_relevant_history(
            history_messages=history_messages,
            user_input=run_context.user_input,
        )

        compressed_history = compress_history_messages(
            filtered_history,
            model=None,
            max_tokens=max_tokens,
            max_chars=max_chars,
        )
        runtime_notes = [
            "上下文装配策略: static/config/memory/dynamic 四层提示前缀 + recent history + compact boundary。",
            (
                f"历史压缩结果: raw={total_raw_count}, "
                f"filtered={len(filtered_history)}, compressed={len(compressed_history)}"
            ),
        ]
        prompt_blocks = list(system_prompt_registry.render_global_prompt_blocks(
            model_config=run_context.model_config,
            tool_catalog=prompt_tool_catalog,
            skill_prompts=run_context.model_config.get("skill_prompt_blocks") or [],
            skill_names=run_context.model_config.get("selected_skill_names") or [],
            memory_block=memory_block,
            dynamic_blocks=dynamic_blocks,
            runtime_notes=runtime_notes,
        ))

        messages: List[BaseMessage] = list(system_prompt_registry.build_runtime_messages(
            model_config=run_context.model_config,
            tool_catalog=prompt_tool_catalog,
            skill_prompts=run_context.model_config.get("skill_prompt_blocks") or [],
            skill_names=run_context.model_config.get("selected_skill_names") or [],
            memory_block=memory_block,
            dynamic_blocks=dynamic_blocks,
            runtime_notes=runtime_notes,
        ))
        messages.extend(compressed_history)
        messages.append(HumanMessage(content=run_context.user_input))

        context_layers = [
            layer.to_dict()
            for layer in runtime_context_governor.describe_layers(
                prompt_blocks=prompt_blocks,
                filtered_history=filtered_history,
                compressed_history=compressed_history,
                memory_count=len(memory_snippets),
                dynamic_blocks=dynamic_blocks,
            )
        ]
        compact_boundary = {
            "triggered": len(compressed_history) <= len(filtered_history),
            "raw_history_count": total_raw_count,
            "filtered_history_count": len(filtered_history),
            "compressed_history_count": len(compressed_history),
        }
        context_meta = {
            "history_message_count": total_raw_count,
            "filtered_history_count": len(filtered_history),
            "compressed_history_count": len(compressed_history),
            "memory_count": len(memory_snippets),
            "history_strategy": "recent+relevance+compression",
            "prompt_segments": [block.segment for block in prompt_blocks if isinstance(block, RuntimePromptBlock)],
            "compact_boundary": compact_boundary,
            "context_layers": context_layers,
            "estimated_tokens": sum(
                ContextBudget.estimate_token_budget(getattr(message, "content", ""))
                for message in messages
            ),
        }
        return messages, memory_snippets, context_meta


runtime_context_builder = RuntimeContextBuilder()
