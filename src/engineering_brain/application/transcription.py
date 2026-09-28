from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import json
from typing import Callable
from uuid import uuid4

from engineering_brain.domain.errors import ArtifactIntegrityError, InvalidTranscriptionResultError, TranscriptionError
from engineering_brain.domain.models import Transcript, TranscriptSegment, TranscriptionRun, TranscriptionStatus
from engineering_brain.ports.artifact_store import ArtifactStore
from engineering_brain.ports.transcription import Transcriber, TranscriptionRepository, TranscriptionResult, TranscriptionSegmentResult


class TranscriptionConfig:
    def __init__(self, requested_language: str | None = None, word_timestamps: bool = True, vad_enabled: bool = False) -> None:
        self.json_value = json.dumps({"requested_language": requested_language, "vad_enabled": vad_enabled, "word_timestamps": word_timestamps}, sort_keys=True, separators=(",", ":"), allow_nan=False)


class TranscriptionService:
    def __init__(self, repository: TranscriptionRepository, artifact_store: ArtifactStore, transcriber: Transcriber, *, id_factory: Callable[[], str] | None = None, clock: Callable[[], datetime] | None = None) -> None:
        self._repository, self._store, self._transcriber = repository, artifact_store, transcriber
        self._id = id_factory or (lambda: str(uuid4()))
        self._clock = clock or (lambda: datetime.now(UTC))

    def transcribe(self, derived_id: str, config: TranscriptionConfig) -> tuple[TranscriptionRun, bool]:
        derived = self._repository.get_derived_audio_artifact(derived_id)
        verification = self._store.verify(derived.managed_key, derived.sha256)
        if not verification.matches_expected_hash:
            raise ArtifactIntegrityError("derived audio failed integrity verification; transcriber was not called")
        engine, version, model, revision, device, compute = self._transcriber.identity()
        existing = self._repository.find_equivalent_completed_transcription(derived.id, engine, version, model, revision, device, compute, config.json_value)
        if existing is not None:
            return existing, True
        run = TranscriptionRun(self._id(), derived.id, engine, version, model, revision, device, compute, config.json_value, json.loads(config.json_value)["requested_language"], None, None, self._clock(), None, TranscriptionStatus.RUNNING)
        self._repository.create_transcription_run(run)

        try:
            result = self._transcriber.transcribe(self._store.path_for_read(derived.managed_key), config.json_value)
            transcript, segments = self._validated_output(run, result)
            self._repository.complete_transcription_run(run.id, transcript, segments, result.detected_language, result.language_probability, self._clock())
        except Exception as exc:
            self._repository.mark_transcription_run_failed(run.id, str(exc), self._clock())
            if isinstance(exc, TranscriptionError):
                raise
            raise TranscriptionError(f"transcription failed: {exc}") from exc

        # Post-completion reads must never change a committed COMPLETED run to FAILED.
        return self._repository.get_transcription_run(run.id), False

    def _validated_output(self, run: TranscriptionRun, result: TranscriptionResult) -> tuple[Transcript, tuple[TranscriptSegment, ...]]:
        if not isinstance(result.full_text, str) or not result.full_text.strip():
            raise InvalidTranscriptionResultError("transcription result full_text must not be empty")
        _probability(result.language_probability, "language_probability")
        ordinals: list[int] = []
        for index, segment in enumerate(result.segments):
            ordinal = index if segment.ordinal is None else segment.ordinal
            if not isinstance(ordinal, int) or ordinal < 0:
                raise InvalidTranscriptionResultError("segment ordinal must be a non-negative integer")
            ordinals.append(ordinal)
            self._validate_segment(segment)
        if len(set(ordinals)) != len(ordinals):
            raise InvalidTranscriptionResultError("segment ordinals must be unique")

        transcript = Transcript(self._id(), run.id, result.language, result.full_text, self._clock())
        segments = tuple(TranscriptSegment(self._id(), transcript.id, ordinal, segment.start_seconds, segment.end_seconds, segment.text, avg_logprob=segment.avg_logprob, no_speech_prob=segment.no_speech_prob, compression_ratio=segment.compression_ratio, temperature=segment.temperature, words_json=segment.words_json) for ordinal, segment in zip(ordinals, result.segments))
        return transcript, segments

    def _validate_segment(self, segment: TranscriptionSegmentResult) -> None:
        start = _decimal(segment.start_seconds, "segment start_seconds")
        end = _decimal(segment.end_seconds, "segment end_seconds")
        if start < 0 or end < start:
            raise InvalidTranscriptionResultError("segment timestamps are invalid")
        for value, label in ((segment.avg_logprob, "segment avg_logprob"), (segment.no_speech_prob, "segment no_speech_prob"), (segment.compression_ratio, "segment compression_ratio"), (segment.temperature, "segment temperature")):
            if value is not None:
                decimal = _decimal(value, label)
                if label.endswith("no_speech_prob") and not 0 <= decimal <= 1:
                    raise InvalidTranscriptionResultError("segment no_speech_prob must be between zero and one")
        if segment.words_json is not None:
            self._validate_words(segment.words_json)

    def _validate_words(self, words_json: str) -> None:
        try:
            words = json.loads(words_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise InvalidTranscriptionResultError("word timestamps must be valid JSON") from exc
        if not isinstance(words, list):
            raise InvalidTranscriptionResultError("word timestamps must be a JSON array")
        for word in words:
            if not isinstance(word, dict) or not isinstance(word.get("word"), str):
                raise InvalidTranscriptionResultError("word timestamps must contain objects with a word string")
            start = _decimal(word.get("start"), "word start")
            end = _decimal(word.get("end"), "word end")
            if start < 0 or end < start:
                raise InvalidTranscriptionResultError("word timestamps are invalid")
            if "probability" in word:
                _probability(word["probability"], "word probability")


def _decimal(value: object, field_name: str) -> Decimal:
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise InvalidTranscriptionResultError(f"{field_name} must be a finite decimal") from exc
    if not decimal.is_finite():
        raise InvalidTranscriptionResultError(f"{field_name} must be finite")
    return decimal


def _probability(value: object | None, field_name: str) -> None:
    if value is None:
        return
    probability = _decimal(value, field_name)
    if probability < 0 or probability > 1:
        raise InvalidTranscriptionResultError(f"{field_name} must be between zero and one")
