from __future__ import annotations

from dataclasses import dataclass, field
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
    original_location: str
    managed_key: str | None
    sha256: str
    byte_size: int
    observed_at: datetime
    original_filename: str | None = None
    media_type: str | None = None

    def __post_init__(self) -> None:
        _required(self.id, "artifact id")
        _required(self.original_location, "artifact original_location")
        if self.managed_key is not None:
            _required(self.managed_key, "artifact managed_key")
        if not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValidationError("artifact sha256 must be a lowercase SHA-256 digest")
        if self.byte_size < 0:
            raise ValidationError("artifact byte_size must not be negative")
        if self.original_filename is not None:
            _required(self.original_filename, "artifact original_filename")
        if self.media_type is not None:
            _required(self.media_type, "artifact media_type")
        _aware(self.observed_at, "artifact observed_at")


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
    artifacts: tuple[SourceArtifact, ...]
