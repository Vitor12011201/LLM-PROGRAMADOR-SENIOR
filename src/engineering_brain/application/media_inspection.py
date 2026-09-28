from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

from engineering_brain.domain.errors import ArtifactIntegrityError, MediaInspectionError
from engineering_brain.domain.models import AudioStream, MediaClassification, MediaInspection, SubtitleStream, VideoStream
from engineering_brain.application.media_normalization import canonical_probe_json, normalize_probe
from engineering_brain.ports.artifact_store import ArtifactStore
from engineering_brain.ports.media_inspection import MediaInspectionRepository, MediaInspector

MEDIA_INSPECTION_SCHEMA_VERSION = "media-inspection-v1"


class MediaInspectionService:
    def __init__(
        self,
        repository: MediaInspectionRepository,
        artifact_store: ArtifactStore,
        inspector: MediaInspector,
        *,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._repository = repository
        self._artifact_store = artifact_store
        self._inspector = inspector
        self._id_factory = id_factory or (lambda: str(uuid4()))
        self._clock = clock or (lambda: datetime.now(UTC))

    def inspect_artifact(self, artifact_id: str) -> tuple[MediaInspection, bool]:
        artifact = self._repository.get_artifact(artifact_id)
        if artifact.managed_key is None:
            raise MediaInspectionError("artifact has no managed copy and cannot be inspected")
        verification = self._artifact_store.verify(artifact.managed_key, artifact.sha256)
        if not verification.matches_expected_hash:
            raise ArtifactIntegrityError("managed artifact failed integrity verification; media inspection was not run")
        tool, tool_version = self._inspector.tool_identity()
        existing = self._repository.find_media_inspection(
            artifact.id, tool, tool_version, MEDIA_INSPECTION_SCHEMA_VERSION
        )
        if existing is not None:
            return existing, True
        execution = self._inspector.inspect(self._artifact_store.path_for_read(artifact.managed_key))
        normalized = normalize_probe(execution.payload)
        inspection = MediaInspection(
            id=self._id_factory(),
            artifact_id=artifact.id,
            tool=execution.tool,
            tool_version=execution.tool_version,
            schema_version=MEDIA_INSPECTION_SCHEMA_VERSION,
            inspected_at=self._clock(),
            status="completed",
            classification=MediaClassification(normalized["classification"]),
            format_names=normalized["format_names"],
            duration_seconds=normalized["duration_seconds"],
            bit_rate=normalized["bit_rate"],
            stream_count=normalized["stream_count"],
            video_streams=tuple(VideoStream(**stream) for stream in normalized["video_streams"]),
            audio_streams=tuple(AudioStream(**stream) for stream in normalized["audio_streams"]),
            subtitle_streams=tuple(SubtitleStream(**stream) for stream in normalized["subtitle_streams"]),
            raw_probe_json=canonical_probe_json(execution.payload),
        )
        self._repository.create_media_inspection(inspection)
        return inspection, False

    def show_latest(self, artifact_id: str) -> MediaInspection:
        return self._repository.latest_media_inspection(artifact_id)
