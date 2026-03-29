# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass(slots=True)
class WorkspaceArtifact:
    """工作区产物元数据。"""

    name: str
    path: str
    relative_path: str
    category: str = "artifact"
    media_type: str = "text/plain"
    size_bytes: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

