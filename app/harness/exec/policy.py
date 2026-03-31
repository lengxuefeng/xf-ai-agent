# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
import shlex
from typing import Iterable, List, Set


@dataclass(slots=True)
class ExecPolicy:
    """命令执行策略。

    当前策略面向“本地项目工作台”场景：
    1. 允许常见开发、构建、测试、文件查看类命令；
    2. 禁止明显的空命令或未知可执行程序；
    3. 目录边界仍由工作区守卫负责。
    """

    timeout_seconds: float = 180.0
    allowed_binaries: Set[str] = field(
        default_factory=lambda: {
            "python",
            "python3",
            "pytest",
            "uv",
            "pip",
            "pip3",
            "npm",
            "npx",
            "node",
            "pnpm",
            "yarn",
            "bun",
            "git",
            "ls",
            "pwd",
            "cat",
            "echo",
            "find",
            "rg",
            "grep",
            "sort",
            "uniq",
            "wc",
            "javac",
            "java",
            "jar",
            "mvn",
            "mvnw",
            "gradle",
            "gradlew",
            "go",
            "cargo",
            "make",
            "touch",
            "mkdir",
            "cp",
            "mv",
            "rm",
            "chmod",
            "ln",
            "head",
            "tail",
            "sed",
            "tar",
            "zip",
            "unzip",
        }
    )

    def allows(self, command: Iterable[str]) -> bool:
        """判断命令是否命中白名单。"""
        argv = list(command)
        if not argv:
            return False
        binary = str(argv[0]).strip().split("/")[-1]
        return binary in self.allowed_binaries

    def parse_command_text(self, command_text: str) -> List[str]:
        """将命令文本解析为 argv，禁止空命令。"""
        argv = shlex.split(str(command_text or "").strip())
        if not argv:
            raise ValueError("命令不能为空。")
        return argv

    def describe(self) -> dict:
        """返回执行策略摘要，供健康检查和前端展示使用。"""
        return {
            "timeout_seconds": self.timeout_seconds,
            "allowed_binaries": sorted(self.allowed_binaries),
        }

    def explain_rejection(self, command: Iterable[str]) -> str:
        """生成更适合前端展示的拒绝原因。"""
        argv = list(command)
        binary = str(argv[0]).strip() if argv else ""
        binary_name = binary.split("/")[-1] if binary else "未知命令"
        examples = "python3 app.py、pytest、uv run python -m unittest、javac Main.java、java Main、npm run dev、git status"
        return (
            f"页面工位暂不支持命令 `{binary_name}`。"
            f"当前更适合运行项目开发命令，例如：{examples}。"
            "如果你只是想改文件，请先在左侧工作树中打开文件直接编辑。"
        )
