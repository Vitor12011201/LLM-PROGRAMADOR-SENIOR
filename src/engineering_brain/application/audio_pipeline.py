from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import tempfile
from typing import Callable
from uuid import uuid4

from engineering_brain.domain.errors import (
    AmbiguousAudioSelectionError,
    ArtifactIntegrityError,
    InvalidAudioSelectionError,
    InvalidAudioStreamError,
    NoAudioStreamError,
)
from engineering_brain.domain.models import AudioSelection, DerivedAudioArtifact
from engineering_brain.ports.artifact_store import ArtifactStore
from engineering_brain.ports.audio_extraction import AudioExtractor
from engineering_brain.ports.audio_pipeline import AudioPipelineRepository


AUDIO_DERIVATION_SCHEMA_VERSION = "audio-derivation-v1"
CANONICAL_AUDIO_CONFIG_JSON = json.dumps({"channels": 1, "codec": "pcm_s16le", "container": "wav", "sample_rate": 16000}, separators=(",", ":"), sort_keys=True)


class AudioPipelineService:
    def __init__(self, repository: AudioPipelineRepository, artifact_store: ArtifactStore, extractor: AudioExtractor, *, id_factory: Callable[[], str] | None = None, clock: Callable[[], datetime] | None = None) -> None:
        self._repository, self._artifact_store, self._extractor = repository, artifact_store, extractor
        self._id_factory = id_factory or (lambda: str(uuid4()))
        self._clock = clock or (lambda: datetime.now(UTC))

    def select_audio(self, artifact_id: str, inspection_id: str, stream_index: int | None = None) -> tuple[AudioSelection, bool]:
        inspection = self._repository.get_media_inspection(inspection_id)
        if inspection.artifact_id != artifact_id:
            raise InvalidAudioSelectionError("media inspection does not belong to source artifact")
        streams = inspection.audio_streams
        if not streams:
            raise NoAudioStreamError("media inspection has no audio streams")
        indexes = {stream.index for stream in streams if stream.index is not None}
        if stream_index is None:
            if len(indexes) != 1:
                raise AmbiguousAudioSelectionError("multiple audio streams require an explicit stream_index")
            stream_index, policy, reason = next(iter(indexes)), "ONLY_AUDIO_STREAM", "the inspection contains exactly one audio stream"
        else:
            if stream_index not in indexes:
                raise InvalidAudioStreamError(
                    f"stream index {stream_index} is not an audio stream in this inspection"
                )
            policy, reason = "EXPLICIT_STREAM_INDEX", "stream index was explicitly selected"
        existing = self._repository.find_audio_selection(artifact_id, inspection_id, stream_index)
        if existing is not None:
            return existing, True
        selection = AudioSelection(self._id_factory(), artifact_id, inspection_id, stream_index, policy, reason, self._clock())
        self._repository.create_audio_selection(selection)
        return selection, False

    def derive_audio(self, selection_id: str) -> tuple[DerivedAudioArtifact, bool]:
        selection = self._repository.get_audio_selection(selection_id)
        artifact = self._repository.get_artifact(selection.source_artifact_id)
        if artifact.managed_key is None:
            raise ArtifactIntegrityError("source artifact has no managed copy")
        verification = self._artifact_store.verify(artifact.managed_key, artifact.sha256)
        if not verification.matches_expected_hash:
            raise ArtifactIntegrityError("source artifact failed integrity verification; extraction was not run")
        extractor, version = self._extractor.tool_identity()
        existing = self._repository.find_derived_audio_artifact(selection.id, extractor, version, AUDIO_DERIVATION_SCHEMA_VERSION, CANONICAL_AUDIO_CONFIG_JSON)
        if existing is not None:
            existing_verification = self._artifact_store.verify(existing.managed_key, existing.sha256)
            if not existing_verification.matches_expected_hash:
                raise ArtifactIntegrityError(
                    "existing derived audio failed integrity verification; extraction was not run"
                )
            return existing, True
        with tempfile.TemporaryDirectory(prefix="engineering-brain-audio-") as directory:
            output = Path(directory) / "audio.wav"
            self._extractor.extract(self._artifact_store.path_for_read(artifact.managed_key), selection.stream_index, output)
            stored = self._artifact_store.ingest(output)
        derived = DerivedAudioArtifact(self._id_factory(), selection.id, stored.managed_key, stored.sha256, stored.byte_size, extractor, version, AUDIO_DERIVATION_SCHEMA_VERSION, CANONICAL_AUDIO_CONFIG_JSON, self._clock())
        self._repository.create_derived_audio_artifact(derived)
        return derived, False

    def show_audio_selection(self, selection_id: str) -> AudioSelection:
        return self._repository.get_audio_selection(selection_id)

    def show_derived_audio(self, derived_audio_artifact_id: str) -> DerivedAudioArtifact:
        return self._repository.get_derived_audio_artifact(derived_audio_artifact_id)
