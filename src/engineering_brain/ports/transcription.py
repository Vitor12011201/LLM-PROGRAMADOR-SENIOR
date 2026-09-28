from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Protocol

from engineering_brain.domain.models import (
    DerivedAudioArtifact,
    Transcript,
    TranscriptSegment,
    TranscriptionRun,
)


@dataclass(frozen=True, slots=True)
class TranscriptionSegmentResult:
    start_seconds: str
    end_seconds: str
    text: str
    words_json: str | None = None
    ordinal: int | None = None
    avg_logprob: str | None = None
    no_speech_prob: str | None = None
    compression_ratio: str | None = None
    temperature: str | None = None


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    full_text: str
    language: str | None
    detected_language: str | None
    language_probability: str | None
    segments: tuple[TranscriptionSegmentResult, ...]


class Transcriber(Protocol):
    def identity(self) -> tuple[str, str, str, str | None, str | None, str | None]: ...

    def transcribe(self, audio_path: Path, config_json: str) -> TranscriptionResult: ...


class TranscriptionRepository(Protocol):
    def get_derived_audio_artifact(self, derived_id: str) -> DerivedAudioArtifact: ...
    def find_equivalent_completed_transcription(
        self, derived_id: str, engine: str, version: str, model_id: str,
        revision: str | None, device: str | None, compute: str | None, config: str,
    ) -> TranscriptionRun | None: ...
    def create_transcription_run(self, run: TranscriptionRun) -> None: ...
    def complete_transcription_run(
        self, run_id: str, transcript: Transcript, segments: tuple[TranscriptSegment, ...],
        detected: str | None, probability: str | None, completed_at: datetime,
    ) -> None: ...
    def mark_transcription_run_failed(
        self, run_id: str, message: str, completed_at: datetime,
    ) -> None: ...
    def get_transcription_run(self, run_id: str) -> TranscriptionRun: ...
