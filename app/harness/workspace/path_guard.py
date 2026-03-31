# -*- coding: utf-8 -*-
"""受控工作目录校验器。

本模块负责两类校验：
1. 校验用户指定的工作目录是否落在允许的根目录范围内；
2. 校验命令实际运行目录是否仍位于工作目录之内，避免通过相对路径或软链逃逸。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List


def _default_allowed_roots() -> List[Path]:
    """返回默认允许的工作目录根列表。"""
    project_family_root = Path(__file__).resolve().parents[4]
    return [project_family_root]


def _load_allowed_roots() -> List[Path]:
    """从环境变量或默认值构建允许的根目录集合。"""
    raw = os.getenv("TERMINAL_ALLOWED_ROOTS", "").strip()
    if not raw:
        return _default_allowed_roots()

    roots: List[Path] = []
    for segment in raw.split(os.pathsep):
        normalized = segment.strip()
        if not normalized:
            continue
        roots.append(Path(normalized).expanduser().resolve())
    return roots or _default_allowed_roots()


class WorkspacePathGuard:
    """工作目录边界守卫。"""

    def __init__(self, allowed_roots: Iterable[Path] | None = None) -> None:
        """初始化允许的顶层目录白名单。"""
        self._allowed_roots = [Path(root).expanduser().resolve() for root in (allowed_roots or _load_allowed_roots())]

    @property
    def allowed_roots(self) -> List[Path]:
        """返回当前允许的根目录白名单。"""
        return list(self._allowed_roots)

    def describe(self) -> Dict[str, List[str]]:
        """返回用于健康检查和前端展示的描述信息。"""
        return {"allowed_roots": [str(root) for root in self._allowed_roots]}

    def resolve_workspace_root(self, requested_root: str) -> Path:
        """解析并校验用户指定的工作目录。"""
        raw = str(requested_root or "").strip()
        if not raw:
            raise ValueError("工作目录不能为空。")

        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self._allowed_roots[0] / candidate

        resolved = candidate.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError("指定的工作目录不存在或不是文件夹。")

        if not any(self._is_within(resolved, root) for root in self._allowed_roots):
            raise ValueError("指定目录超出了当前允许的工作区范围。")
        return resolved

    def resolve_execution_cwd(self, workspace_root: str | Path, requested_cwd: str | None = None) -> Path:
        """解析命令实际执行目录，确保不越过工作目录。"""
        root = self.resolve_workspace_root(str(workspace_root))
        if not requested_cwd:
            return root

        candidate = Path(requested_cwd).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate

        resolved = candidate.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError("命令执行目录不存在或不是文件夹。")
        if not self._is_within(resolved, root):
            raise ValueError("命令执行目录超出了当前工作目录范围。")
        return resolved

    @staticmethod
    def _is_within(candidate: Path, root: Path) -> bool:
        """判断 candidate 是否落在 root 目录树内。"""
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            return False


workspace_path_guard = WorkspacePathGuard()
