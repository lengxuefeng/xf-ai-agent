# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from runtime.types import RunContext
from runtime.workspace.path_guard import workspace_path_guard
from runtime.workspace.models import WorkspaceArtifact


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sanitize_segment(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return normalized[:80] or fallback


class WorkspaceManager:
    """运行工作区管理器。"""

    def __init__(self, root_dir: Path | None = None) -> None:
        self._root_dir = Path(root_dir or (_project_root() / "runtime_data" / "workspaces"))
        self._bound_workspace_roots: Dict[str, str] = {}

    @property
    def root_dir(self) -> Path:
        self._root_dir.mkdir(parents=True, exist_ok=True)
        return self._root_dir

    def prepare_run_workspace(self, run_context: RunContext) -> Dict[str, Any]:
        return self.prepare_workspace_by_ids(
            session_id=run_context.session_id,
            run_id=run_context.run_id,
        )

    def prepare_workspace_by_ids(self, *, session_id: str, run_id: str) -> Dict[str, Any]:
        """为一次运行准备内部 artifact 工作区。"""
        session_dir = self.root_dir / _sanitize_segment(session_id, "session")
        run_dir = session_dir / _sanitize_segment(run_id, "run")
        artifacts_dir = run_dir / "artifacts"
        scratch_dir = run_dir / "scratch"
        logs_dir = run_dir / "logs"
        for directory in (run_dir, artifacts_dir, scratch_dir, logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

        return {
            "root": str(run_dir),
            "artifacts_dir": str(artifacts_dir),
            "scratch_dir": str(scratch_dir),
            "logs_dir": str(logs_dir),
            "relative_root": str(run_dir.relative_to(self.root_dir)),
        }

    def bind_external_workspace(self, session_id: str, workspace_root: str) -> Dict[str, Any]:
        """绑定某个聊天会话的外部工作目录。"""
        resolved_root = workspace_path_guard.resolve_workspace_root(workspace_root)
        self._bound_workspace_roots[str(session_id)] = str(resolved_root)
        return self.describe_external_workspace(session_id)

    def describe_external_workspace(self, session_id: str) -> Dict[str, Any]:
        """返回聊天会话当前绑定的外部工作目录信息。"""
        bound_root = self._bound_workspace_roots.get(str(session_id))
        if not bound_root:
            return {
                "workspace_root": "",
                "allowed_roots": workspace_path_guard.describe()["allowed_roots"],
            }

        return {
            "workspace_root": bound_root,
            "workspace_name": os.path.basename(bound_root),
            "allowed_roots": workspace_path_guard.describe()["allowed_roots"],
        }

    def get_external_workspace(self, session_id: str) -> str:
        """读取会话当前绑定的外部工作目录。"""
        return str(self._bound_workspace_roots.get(str(session_id)) or "")

    def list_workspace_directory(self, session_id: str, relative_path: str = "") -> Dict[str, Any]:
        """列出某个已绑定工作目录下的目录内容。"""
        root = self._resolve_bound_root(session_id)
        target = self._resolve_workspace_path(root, relative_path, must_exist=True)
        if not target.is_dir():
            raise ValueError("指定路径不是目录。")

        items: List[Dict[str, Any]] = []
        for item in sorted(target.iterdir(), key=lambda path: (not path.is_dir(), path.name.lower())):
            stat = item.stat()
            items.append(
                {
                    "name": item.name,
                    "path": self._relative_workspace_path(root, item),
                    "type": "directory" if item.is_dir() else "file",
                    "size_bytes": 0 if item.is_dir() else stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )

        return {
            "workspace_root": str(root),
            "path": self._relative_workspace_path(root, target),
            "items": items,
        }

    def read_workspace_file(self, session_id: str, relative_path: str) -> Dict[str, Any]:
        """读取已绑定工作目录下的文本文件。"""
        root = self._resolve_bound_root(session_id)
        target = self._resolve_workspace_path(root, relative_path, must_exist=True)
        if not target.is_file():
            raise ValueError("指定路径不是文件。")

        content = target.read_text(encoding="utf-8", errors="replace")
        stat = target.stat()
        return {
            "workspace_root": str(root),
            "path": self._relative_workspace_path(root, target),
            "name": target.name,
            "content": content,
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

    def save_workspace_file(self, session_id: str, relative_path: str, content: str) -> Dict[str, Any]:
        """将文本内容保存到已绑定工作目录内的文件。"""
        root = self._resolve_bound_root(session_id)
        target = self._resolve_workspace_path(root, relative_path, must_exist=False)
        if target.exists() and not target.is_file():
            raise ValueError("只能保存到文件路径。")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content or ""), encoding="utf-8")
        return self.read_workspace_file(session_id, relative_path)

    def create_workspace_file(self, session_id: str, relative_path: str, content: str = "") -> Dict[str, Any]:
        """创建新的工作区文本文件。"""
        root = self._resolve_bound_root(session_id)
        target = self._resolve_workspace_path(root, relative_path, must_exist=False)
        if target.exists():
            raise ValueError("目标文件已存在。")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content or ""), encoding="utf-8")
        return self.read_workspace_file(session_id, relative_path)

    def create_workspace_directory(self, session_id: str, relative_path: str) -> Dict[str, Any]:
        """创建新的工作区目录。"""
        root = self._resolve_bound_root(session_id)
        target = self._resolve_workspace_path(root, relative_path, must_exist=False)
        if target.exists():
            raise ValueError("目标目录已存在。")
        target.mkdir(parents=True, exist_ok=False)
        return {
            "workspace_root": str(root),
            "path": self._relative_workspace_path(root, target),
            "name": target.name,
            "type": "directory",
        }

    def rename_workspace_entry(self, session_id: str, relative_path: str, new_relative_path: str) -> Dict[str, Any]:
        """在工作目录内重命名或移动文件、目录。"""
        root = self._resolve_bound_root(session_id)
        source = self._resolve_workspace_path(root, relative_path, must_exist=True)
        if source == root:
            raise ValueError("不允许重命名工作目录根节点。")

        target = self._resolve_workspace_path(root, new_relative_path, must_exist=False)
        if target.exists():
            raise ValueError("目标路径已存在。")

        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        return {
            "workspace_root": str(root),
            "old_path": self._relative_workspace_path(root, source),
            "path": self._relative_workspace_path(root, target),
            "name": target.name,
            "type": "directory" if target.is_dir() else "file",
        }

    def delete_workspace_entry(self, session_id: str, relative_path: str) -> Dict[str, Any]:
        """删除工作目录内的文件或目录。"""
        root = self._resolve_bound_root(session_id)
        target = self._resolve_workspace_path(root, relative_path, must_exist=True)
        if target == root:
            raise ValueError("不允许删除工作目录根节点。")

        target_type = "directory" if target.is_dir() else "file"
        deleted_path = self._relative_workspace_path(root, target)
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

        return {
            "workspace_root": str(root),
            "path": deleted_path,
            "type": target_type,
        }

    def write_text_artifact(
        self,
        run_context: RunContext,
        *,
        name: str,
        content: str,
        category: str = "artifact",
        media_type: str = "text/plain",
    ) -> Dict[str, Any]:
        workspace_meta = self.prepare_run_workspace(run_context)
        artifacts_dir = Path(workspace_meta["artifacts_dir"])
        safe_name = _sanitize_segment(name, "artifact")
        if "." not in safe_name:
            safe_name = f"{safe_name}.txt"
        artifact_path = artifacts_dir / safe_name
        artifact_path.write_text(str(content or ""), encoding="utf-8")
        artifact = WorkspaceArtifact(
            name=safe_name,
            path=str(artifact_path),
            relative_path=str(artifact_path.relative_to(self.root_dir)),
            category=category,
            media_type=media_type,
            size_bytes=artifact_path.stat().st_size,
        )
        return artifact.to_dict()

    def write_json_artifact(
        self,
        run_context: RunContext,
        *,
        name: str,
        payload: Any,
        category: str = "artifact",
    ) -> Dict[str, Any]:
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        safe_name = name if name.endswith(".json") else f"{name}.json"
        return self.write_text_artifact(
            run_context,
            name=safe_name,
            content=text,
            category=category,
            media_type="application/json",
        )

    def list_artifacts(self, run_context: RunContext) -> List[Dict[str, Any]]:
        workspace_meta = self.prepare_run_workspace(run_context)
        return self.list_artifacts_by_ids(
            session_id=run_context.session_id,
            run_id=run_context.run_id,
            workspace_meta=workspace_meta,
        )

    def list_artifacts_by_ids(
        self,
        *,
        session_id: str,
        run_id: str,
        workspace_meta: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        workspace_meta = workspace_meta or self.prepare_workspace_by_ids(
            session_id=session_id,
            run_id=run_id,
        )
        artifacts_dir = Path(workspace_meta["artifacts_dir"])
        results: List[Dict[str, Any]] = []
        for item in sorted(artifacts_dir.glob("*")):
            if not item.is_file():
                continue
            artifact = WorkspaceArtifact(
                name=item.name,
                path=str(item),
                relative_path=str(item.relative_to(self.root_dir)),
                size_bytes=item.stat().st_size,
            )
            results.append(artifact.to_dict())
        return results

    def _resolve_bound_root(self, session_id: str) -> Path:
        """读取并校验会话当前绑定的工作目录。"""
        bound_root = self.get_external_workspace(session_id)
        if not bound_root:
            raise ValueError("当前会话尚未绑定工作目录。")
        return workspace_path_guard.resolve_workspace_root(bound_root)

    def _resolve_workspace_path(self, root: Path, relative_path: str, *, must_exist: bool) -> Path:
        """在工作目录内解析目标路径，防止通过相对路径逃逸。"""
        normalized = str(relative_path or "").strip()
        if not normalized:
            return root

        candidate = (root / normalized).expanduser()
        resolved = candidate.resolve(strict=must_exist)
        self._ensure_within_root(root, resolved)
        return resolved

    @staticmethod
    def _ensure_within_root(root: Path, candidate: Path) -> None:
        """确保目标路径仍位于工作目录以内。"""
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("目标路径超出了当前工作目录范围。") from exc

    @staticmethod
    def _relative_workspace_path(root: Path, target: Path) -> str:
        """返回相对于工作目录的标准路径表示。"""
        try:
            relative = target.relative_to(root)
        except ValueError:
            return ""
        return "" if str(relative) == "." else str(relative)


workspace_manager = WorkspaceManager()
