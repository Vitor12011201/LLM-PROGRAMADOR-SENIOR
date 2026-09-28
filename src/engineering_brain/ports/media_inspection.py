from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from engineering_brain.domain.models import MediaInspection, SourceArtifact


@dataclass(frozen=True, slots=True)
class ProbeExecution:
    tool: str
    tool_version: str
    payload: Mapping[str, object]


class MediaInspector(Protocol):
    def tool_identity(self) -> tuple[str, str]: ...

    def inspect(self, artifact_path: Path) -> ProbeExecution: ...


class MediaInspectionRepository(Protocol):
    def get_artifact(self, artifact_id: str) -> SourceArtifact: ...

    def find_media_inspection(
        self, artifact_id: str, tool: str, tool_version: str, schema_version: str
    ) -> MediaInspection | None: ...

    def create_media_inspection(self, inspection: MediaInspection) -> None: ...

    def latest_media_inspection(self, artifact_id: str) -> MediaInspection: ...
