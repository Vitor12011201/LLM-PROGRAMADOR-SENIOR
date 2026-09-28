from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from datetime import datetime
from enum import Enum
import json
import re
from typing import Mapping

from engineering_brain.domain.errors import ValidationError

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class Metadata:
    """A canonical, immutable JSON object used for optional structured metadata."""

    json_value: str = "{}"

    def __post_init__(self) -> None:
        try:
            decoded = json.loads(self.json_value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValidationError("metadata must be valid JSON") from exc
        if not isinstance(decoded, dict):
            raise ValidationError("metadata must be a JSON object")
        try:
            canonical = json.dumps(
                decoded, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError("metadata must contain only JSON-compatible values") from exc
        object.__setattr__(self, "json_value", canonical)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None = None) -> Metadata:
        try:
            return cls(json.dumps(dict(value or {}), ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValidationError("metadata must contain only JSON-compatible values") from exc

    def as_dict(self) -> dict[str, object]:
        return json.loads(self.json_value)


def _required(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name} must not be empty")


def _aware(timestamp: datetime, field_name: str) -> None:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValidationError(f"{field_name} must include a timezone")


def _canonical_json(value: str, field_name: str, expected_type: type[object]) -> str:
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{field_name} must be valid JSON") from exc
    if not isinstance(decoded, expected_type):
        expected_name = "object" if expected_type is dict else "array"
        raise ValidationError(f"{field_name} must be a JSON {expected_name}")
    try:
        return json.dumps(decoded, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field_name} must contain only JSON-compatible values") from exc


def _finite_decimal(value: str, field_name: str) -> Decimal:
    try:
        decimal = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValidationError(f"{field_name} must be a finite decimal string") from exc
    if not decimal.is_finite():
        raise ValidationError(f"{field_name} must be finite")
    return decimal


@dataclass(frozen=True, slots=True)
class Author:
    id: str
    name: str
    created_at: datetime
    metadata: Metadata = field(default_factory=Metadata)

    def __post_init__(self) -> None:
        _required(self.id, "author id")
        _required(self.name, "author name")
        _aware(self.created_at, "author created_at")


@dataclass(frozen=True, slots=True)
class Source:
    id: str
    kind: str
    title: str
    created_at: datetime
    author_id: str | None = None
    origin: str | None = None
    metadata: Metadata = field(default_factory=Metadata)

    def __post_init__(self) -> None:
        _required(self.id, "source id")
        _required(self.kind, "source kind")
        _required(self.title, "source title")
        if self.author_id is not None:
            _required(self.author_id, "source author_id")
        if self.origin is not None:
            _required(self.origin, "source origin")
        _aware(self.created_at, "source created_at")


@dataclass(frozen=True, slots=True)
class Material:
    id: str
    source_id: str
    kind: str
    title: str
    origin: str
    revision: str
    ingested_at: datetime
    status: str = "registered"
    metadata: Metadata = field(default_factory=Metadata)

    def __post_init__(self) -> None:
        _required(self.id, "material id")
        _required(self.source_id, "material source_id")
        _required(self.kind, "material kind")
        _required(self.title, "material title")
        _required(self.origin, "material origin")
        _required(self.revision, "material revision")
        _required(self.status, "material status")
        _aware(self.ingested_at, "material ingested_at")


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    id: str
    managed_key: str | None
    sha256: str
    byte_size: int
    preserved_at: datetime

    def __post_init__(self) -> None:
        _required(self.id, "artifact id")
        if self.managed_key is not None:
            _required(self.managed_key, "artifact managed_key")
        if not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValidationError("artifact sha256 must be a lowercase SHA-256 digest")
        if self.byte_size < 0:
            raise ValidationError("artifact byte_size must not be negative")
        _aware(self.preserved_at, "artifact preserved_at")


@dataclass(frozen=True, slots=True)
class ArtifactObservation:
    id: str
    material_id: str
    artifact_id: str
    original_location: str
    original_filename: str | None
    observed_at: datetime
    media_type: str | None = None

    def __post_init__(self) -> None:
        _required(self.id, "artifact observation id")
        _required(self.material_id, "artifact observation material_id")
        _required(self.artifact_id, "artifact observation artifact_id")
        _required(self.original_location, "artifact observation original_location")
        if self.original_filename is not None:
            _required(self.original_filename, "artifact observation original_filename")
        if self.media_type is not None:
            _required(self.media_type, "artifact observation media_type")
        _aware(self.observed_at, "artifact observation observed_at")


@dataclass(frozen=True, slots=True)
class ArtifactObservationRecord:
    observation: ArtifactObservation
    artifact: SourceArtifact


@dataclass(frozen=True, slots=True)
class ArtifactVerification:
    artifact_id: str
    managed_key: str | None
    exists: bool
    is_valid: bool
    actual_sha256: str | None
    actual_byte_size: int | None
    message: str


class MediaClassification(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    AUDIO_VIDEO = "audio_video"
    OTHER_MEDIA = "other_media"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class VideoStream:
    index: int | None
    codec: str | None
    width: int | None
    height: int | None
    frame_rate: str | None
    pixel_format: str | None
    duration_seconds: str | None
    is_default: bool
    language: str | None


@dataclass(frozen=True, slots=True)
class AudioStream:
    index: int | None
    codec: str | None
    sample_rate: int | None
    channels: int | None
    channel_layout: str | None
    duration_seconds: str | None
    bit_rate: int | None
    is_default: bool
    language: str | None


@dataclass(frozen=True, slots=True)
class SubtitleStream:
    index: int | None
    codec: str | None
    is_default: bool
    language: str | None


@dataclass(frozen=True, slots=True)
class MediaInspection:
    id: str
    artifact_id: str
    tool: str
    tool_version: str
    schema_version: str
    inspected_at: datetime
    status: str
    classification: MediaClassification
    format_names: tuple[str, ...]
    duration_seconds: str | None
    bit_rate: int | None
    stream_count: int
    video_streams: tuple[VideoStream, ...]
    audio_streams: tuple[AudioStream, ...]
    subtitle_streams: tuple[SubtitleStream, ...]
    raw_probe_json: str

    def __post_init__(self) -> None:
        for value, label in ((self.id, "inspection id"), (self.artifact_id, "inspection artifact_id"),
                             (self.tool, "inspection tool"), (self.tool_version, "inspection tool_version"),
                             (self.schema_version, "inspection schema_version"), (self.status, "inspection status")):
            _required(value, label)
        _aware(self.inspected_at, "inspection inspected_at")
        if self.stream_count < 0:
            raise ValidationError("inspection stream_count must not be negative")
        try:
            parsed = json.loads(self.raw_probe_json)
        except json.JSONDecodeError as exc:
            raise ValidationError("inspection raw_probe_json must be valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValidationError("inspection raw_probe_json must be a JSON object")


@dataclass(frozen=True, slots=True)
class MaterialRecord:
    material: Material
    source: Source
    author: Author | None
    artifact_observations: tuple[ArtifactObservationRecord, ...]


@dataclass(frozen=True, slots=True)
class AudioSelection:
    id: str
    source_artifact_id: str
    media_inspection_id: str
    stream_index: int
    policy: str
    reason: str
    selected_at: datetime

    def __post_init__(self) -> None:
        for value, label in ((self.id, "audio selection id"), (self.source_artifact_id, "audio selection source_artifact_id"), (self.media_inspection_id, "audio selection media_inspection_id"), (self.policy, "audio selection policy"), (self.reason, "audio selection reason")):
            _required(value, label)
        if self.stream_index < 0:
            raise ValidationError("audio selection stream_index must not be negative")
        _aware(self.selected_at, "audio selection selected_at")


@dataclass(frozen=True, slots=True)
class DerivedAudioArtifact:
    id: str
    audio_selection_id: str
    managed_key: str
    sha256: str
    byte_size: int
    extractor: str
    extractor_version: str
    derivation_schema_version: str
    config_json: str
    created_at: datetime

    def __post_init__(self) -> None:
        for value, label in ((self.id, "derived audio id"), (self.audio_selection_id, "derived audio selection_id"), (self.managed_key, "derived audio managed_key"), (self.extractor, "derived audio extractor"), (self.extractor_version, "derived audio extractor_version"), (self.derivation_schema_version, "derived audio schema_version")):
            _required(value, label)
        if not _SHA256_PATTERN.fullmatch(self.sha256) or self.byte_size < 0:
            raise ValidationError("derived audio content identity is invalid")
        object.__setattr__(self, "config_json", _canonical_json(self.config_json, "derived audio config_json", dict))
        _aware(self.created_at, "derived audio created_at")


class TranscriptionStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TranscriptionRun:
    id: str
    derived_audio_artifact_id: str
    engine: str
    engine_version: str
    model_id: str
    model_revision: str | None
    device: str | None
    compute_type: str | None
    config_json: str
    requested_language: str | None
    detected_language: str | None
    language_probability: str | None
    started_at: datetime
    completed_at: datetime | None
    status: TranscriptionStatus
    error_message: str | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.id, "transcription run id"),
            (self.derived_audio_artifact_id, "transcription run derived_audio_artifact_id"),
            (self.engine, "transcription run engine"),
            (self.engine_version, "transcription run engine_version"),
            (self.model_id, "transcription run model_id"),
        ):
            _required(value, label)
        object.__setattr__(self, "config_json", _canonical_json(self.config_json, "transcription run config_json", dict))
        _aware(self.started_at, "transcription run started_at")
        if self.completed_at is not None:
            _aware(self.completed_at, "transcription run completed_at")
        if self.language_probability is not None:
            probability = _finite_decimal(self.language_probability, "transcription run language_probability")
            if probability < 0 or probability > 1:
                raise ValidationError("transcription run language_probability must be between zero and one")


@dataclass(frozen=True, slots=True)
class Transcript:
    id: str
    transcription_run_id: str
    language: str | None
    full_text: str
    created_at: datetime

    def __post_init__(self) -> None:
        _required(self.id, "transcript id")
        _required(self.transcription_run_id, "transcript transcription_run_id")
        _required(self.full_text, "transcript full_text")
        _aware(self.created_at, "transcript created_at")


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    id: str
    transcript_id: str
    ordinal: int
    start_seconds: str
    end_seconds: str
    text: str
    avg_logprob: str | None = None
    no_speech_prob: str | None = None
    compression_ratio: str | None = None
    temperature: str | None = None
    words_json: str | None = None

    def __post_init__(self) -> None:
        _required(self.id, "transcript segment id")
        _required(self.transcript_id, "transcript segment transcript_id")
        if self.ordinal < 0:
            raise ValidationError("transcript segment ordinal must not be negative")
        start = _finite_decimal(self.start_seconds, "transcript segment start_seconds")
        end = _finite_decimal(self.end_seconds, "transcript segment end_seconds")
        if start < 0 or end < start:
            raise ValidationError("transcript segment timestamps are invalid")
        for value, label in (
            (self.avg_logprob, "transcript segment avg_logprob"),
            (self.no_speech_prob, "transcript segment no_speech_prob"),
            (self.compression_ratio, "transcript segment compression_ratio"),
            (self.temperature, "transcript segment temperature"),
        ):
            if value is not None:
                decimal = _finite_decimal(value, label)
                if label.endswith("no_speech_prob") and not 0 <= decimal <= 1:
                    raise ValidationError("transcript segment no_speech_prob must be between zero and one")
        if self.words_json is not None:
            object.__setattr__(self, "words_json", _canonical_json(self.words_json, "transcript segment words_json", list))
