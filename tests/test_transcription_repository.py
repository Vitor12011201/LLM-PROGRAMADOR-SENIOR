from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from engineering_brain.application.source_registry import SourceRegistryService
from engineering_brain.application.transcription import TranscriptionConfig, TranscriptionService
from engineering_brain.domain.errors import NotFoundError, RepositoryError, TranscriptionError
from engineering_brain.domain.models import (
    AudioSelection,
    AudioStream,
    DerivedAudioArtifact,
    MediaClassification,
    MediaInspection,
    Transcript,
    TranscriptSegment,
    TranscriptionRun,
    TranscriptionStatus,
)
from engineering_brain.infrastructure.local_artifact_store import LocalArtifactStore
from engineering_brain.infrastructure.sqlite_source_registry import SqliteSourceRegistryRepository
from tests.support.fake_transcriber import FakeTranscriber


NOW = datetime(2026, 9, 28, tzinfo=UTC)
DERIVATION_CONFIG = '{"channels":1,"codec":"pcm_s16le","container":"wav","sample_rate":16000}'


class _DuplicateSegmentCompletionRepository:
    """Test-only fault injection while retaining the real SQLite completion path."""

    def __init__(self, repository: SqliteSourceRegistryRepository) -> None:
        self._repository = repository

    def __getattr__(self, name: str):
        return getattr(self._repository, name)

    def complete_transcription_run(self, run_id, transcript, segments, detected, probability, completed_at):
        duplicate = replace(segments[0], id=f"{segments[0].id}-duplicate")
        return self._repository.complete_transcription_run(
            run_id,
            transcript,
            (segments[0], duplicate),
            detected,
            probability,
            completed_at,
        )


class TranscriptionRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = SqliteSourceRegistryRepository(self.root / "registry.sqlite")
        self.repo.initialize()
        self.store = LocalArtifactStore(self.root / "artifacts")

        registry = SourceRegistryService(self.repo, self.store)
        author = registry.register_author("Repository test author")
        source = registry.register_source(kind="local", title="Repository test source", author_id=author.id)
        material = registry.register_material(
            source_id=source.id,
            kind="recording",
            title="Repository test material",
            origin="file://repository-test",
        )
        source_file = self.root / "source.bin"
        source_file.write_bytes(b"source evidence")
        artifact = registry.attach_local_artifact(material_id=material.id, path=source_file)[0].artifact
        inspection = MediaInspection(
            "inspection",
            artifact.id,
            "ffprobe",
            "test",
            "media-inspection-v1",
            NOW,
            "completed",
            MediaClassification.AUDIO,
            (),
            None,
            None,
            1,
            (),
            (AudioStream(0, "pcm_s16le", 16000, 1, "mono", None, None, False, None),),
            (),
            "{}",
        )
        self.repo.create_media_inspection(inspection)
        selection = AudioSelection(
            "selection",
            artifact.id,
            inspection.id,
            0,
            "ONLY_AUDIO_STREAM",
            "single audio stream",
            NOW,
        )
        self.repo.create_audio_selection(selection)
        derived_file = self.root / "derived.wav"
        derived_file.write_bytes(b"canonical derived audio")
        stored = self.store.ingest(derived_file)
        self.derived = DerivedAudioArtifact(
            "derived",
            selection.id,
            stored.managed_key,
            stored.sha256,
            stored.byte_size,
            "ffmpeg",
            "test",
            "audio-derivation-v1",
            DERIVATION_CONFIG,
            NOW,
        )
        self.repo.create_derived_audio_artifact(self.derived)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _run(
        self,
        run_id: str,
        *,
        status: TranscriptionStatus = TranscriptionStatus.RUNNING,
        config: str | None = None,
    ) -> TranscriptionRun:
        return TranscriptionRun(
            run_id,
            self.derived.id,
            "fake-engine",
            "1.0",
            "fake-model",
            "revision-a",
            "cpu",
            "int8",
            config or TranscriptionConfig().json_value,
            None,
            None,
            None,
            NOW,
            None,
            status,
        )

    def _create_run(
        self,
        run_id: str,
        *,
        status: TranscriptionStatus = TranscriptionStatus.RUNNING,
        config: str | None = None,
    ) -> TranscriptionRun:
        run = self._run(run_id, config=config)
        self.repo.create_transcription_run(run)
        if status is TranscriptionStatus.FAILED:
            self.repo.mark_transcription_run_failed(run.id, "test failure", NOW)
        elif status is TranscriptionStatus.COMPLETED:
            self._complete(run)
        return self.repo.get_transcription_run(run.id)

    def _complete(
        self,
        run: TranscriptionRun,
        *,
        transcript_id: str | None = None,
        segments: tuple[TranscriptSegment, ...] = (),
    ) -> Transcript:
        transcript = Transcript(transcript_id or f"{run.id}-transcript", run.id, "pt", "texto", NOW)
        self.repo.complete_transcription_run(run.id, transcript, segments, "pt", "0.98", NOW)
        return transcript

    def _lookup(self, config: str | None = None):
        return self.repo.find_equivalent_completed_transcription(
            self.derived.id,
            "fake-engine",
            "1.0",
            "fake-model",
            "revision-a",
            "cpu",
            "int8",
            config or TranscriptionConfig().json_value,
        )

    def test_complete_transcription_rolls_back_all_result_rows_on_segment_failure(self) -> None:
        run = self._create_run("rollback")
        transcript = Transcript("rollback-transcript", run.id, "pt", "texto", NOW)
        first = TranscriptSegment("rollback-segment-1", transcript.id, 0, "0", "1", "primeiro")
        duplicate = TranscriptSegment("rollback-segment-2", transcript.id, 0, "1", "2", "segundo")

        with self.assertRaises(RepositoryError):
            self.repo.complete_transcription_run(run.id, transcript, (first, duplicate), "pt", "0.98", NOW)

        with self.assertRaises(NotFoundError):
            self.repo.get_transcript_for_run(run.id)
        self.assertEqual(self.repo._rows("SELECT * FROM transcript_segments", ()), [])
        self.assertNotEqual(self.repo.get_transcription_run(run.id).status, TranscriptionStatus.COMPLETED)

        recovered = self._create_run("after-rollback")
        self.assertEqual(recovered.status, TranscriptionStatus.RUNNING)

    def test_completion_persistence_failure_marks_run_failed(self) -> None:
        fake = FakeTranscriber()
        service = TranscriptionService(
            _DuplicateSegmentCompletionRepository(self.repo), self.store, fake
        )

        with self.assertRaises(TranscriptionError):
            service.transcribe(self.derived.id, TranscriptionConfig())

        run = self.repo._rows("SELECT * FROM transcription_runs", ())[0]
        self.assertEqual(run["status"], TranscriptionStatus.FAILED.value)
        self.assertEqual(self.repo._rows("SELECT * FROM transcripts", ()), [])
        self.assertEqual(self.repo._rows("SELECT * FROM transcript_segments", ()), [])

    def test_database_rejects_second_transcript_for_same_run(self) -> None:
        run = self._create_run("one-transcript", status=TranscriptionStatus.COMPLETED)
        first = self.repo.get_transcript_for_run(run.id)

        with self.assertRaises(sqlite3.IntegrityError):
            with self.repo._connection() as connection:
                connection.execute(
                    "INSERT INTO transcripts (id, transcription_run_id, language, full_text, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("second-transcript", run.id, "pt", "outro", NOW.isoformat()),
                )

        self.assertEqual(self.repo.get_transcript_for_run(run.id).id, first.id)

    def test_database_rejects_duplicate_segment_ordinal(self) -> None:
        run = self._create_run("unique-ordinal")
        transcript = Transcript("unique-ordinal-transcript", run.id, "pt", "texto", NOW)
        first = TranscriptSegment("unique-ordinal-first", transcript.id, 0, "0", "1", "primeiro")
        self.repo.complete_transcription_run(run.id, transcript, (first,), "pt", "0.98", NOW)

        with self.assertRaises(sqlite3.IntegrityError):
            with self.repo._connection() as connection:
                connection.execute(
                    "INSERT INTO transcript_segments "
                    "(id, transcript_id, ordinal, start_seconds, end_seconds, text) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    ("unique-ordinal-second", transcript.id, 0, "1", "2", "duplicado"),
                )

        self.assertEqual([segment.ordinal for segment in self.repo.segments_for_transcript(transcript.id)], [0])

    def test_equivalent_completed_lookup_ignores_failed_run(self) -> None:
        self._create_run("failed", status=TranscriptionStatus.FAILED)
        self.assertIsNone(self._lookup())

    def test_equivalent_completed_lookup_ignores_running_run(self) -> None:
        self._create_run("running")
        self.assertIsNone(self._lookup())

    def test_equivalent_completed_lookup_returns_completed_run(self) -> None:
        completed = self._create_run("completed", status=TranscriptionStatus.COMPLETED)
        self.assertEqual(self._lookup().id, completed.id)

    def test_equivalent_completed_lookup_uses_latest_completion_then_id(self) -> None:
        self._create_run("completed-a", status=TranscriptionStatus.COMPLETED)
        latest = self._create_run("completed-b", status=TranscriptionStatus.COMPLETED)

        self.assertEqual(self._lookup().id, latest.id)

    def test_equivalent_completed_lookup_prefers_completed_when_failed_also_exists(self) -> None:
        self._create_run("failed", status=TranscriptionStatus.FAILED)
        completed = self._create_run("completed", status=TranscriptionStatus.COMPLETED)
        self.assertEqual(self._lookup().id, completed.id)

    def test_equivalent_completed_lookup_returns_completed_when_running_also_exists(self) -> None:
        self._create_run("running")
        completed = self._create_run("completed", status=TranscriptionStatus.COMPLETED)
        self.assertEqual(self._lookup().id, completed.id)

    def test_equivalent_lookup_uses_canonical_config_identity(self) -> None:
        first_config = TranscriptionConfig().json_value
        second_config = TranscriptionConfig(
            word_timestamps=True, requested_language=None, vad_enabled=False
        ).json_value
        self.assertEqual(first_config, second_config)
        completed = self._create_run(
            "canonical", status=TranscriptionStatus.COMPLETED, config=first_config
        )
        self.assertEqual(self._lookup(second_config).id, completed.id)

    def test_equivalent_lookup_rejects_different_canonical_config(self) -> None:
        first_config = TranscriptionConfig(vad_enabled=False).json_value
        different_config = TranscriptionConfig(vad_enabled=True).json_value
        self._create_run("different-config", status=TranscriptionStatus.COMPLETED, config=first_config)
        self.assertIsNone(self._lookup(different_config))

    def test_completed_run_cannot_be_marked_failed(self) -> None:
        completed = self._create_run("completed-state", status=TranscriptionStatus.COMPLETED)

        with self.assertRaises(RepositoryError):
            self.repo.mark_transcription_run_failed(completed.id, "late failure", NOW)

        self.assertEqual(
            self.repo.get_transcription_run(completed.id).status,
            TranscriptionStatus.COMPLETED,
        )
        self.assertEqual(self.repo.get_transcript_for_run(completed.id).full_text, "texto")

    def test_failed_run_cannot_be_completed(self) -> None:
        failed = self._create_run("failed-state", status=TranscriptionStatus.FAILED)
        transcript = Transcript("failed-state-transcript", failed.id, "pt", "texto", NOW)

        with self.assertRaises(RepositoryError):
            self.repo.complete_transcription_run(failed.id, transcript, (), "pt", "0.98", NOW)

        self.assertEqual(
            self.repo.get_transcription_run(failed.id).status,
            TranscriptionStatus.FAILED,
        )
        with self.assertRaises(NotFoundError):
            self.repo.get_transcript_for_run(failed.id)

    def test_completed_run_cannot_be_completed_again(self) -> None:
        completed = self._create_run("completed-again", status=TranscriptionStatus.COMPLETED)
        replacement = Transcript("replacement-transcript", completed.id, "pt", "novo texto", NOW)

        with self.assertRaises(RepositoryError):
            self.repo.complete_transcription_run(completed.id, replacement, (), "pt", "0.98", NOW)

        self.assertEqual(
            self.repo.get_transcription_run(completed.id).status,
            TranscriptionStatus.COMPLETED,
        )
        self.assertEqual(self.repo.get_transcript_for_run(completed.id).id, "completed-again-transcript")

    def test_completion_rejects_transcript_for_different_run(self) -> None:
        run_a = self._create_run("run-a")
        run_b = self._create_run("run-b")
        transcript_for_b = Transcript("transcript-for-b", run_b.id, "pt", "texto", NOW)

        with self.assertRaises(RepositoryError):
            self.repo.complete_transcription_run(run_a.id, transcript_for_b, (), "pt", "0.98", NOW)

        self.assertEqual(self.repo.get_transcription_run(run_a.id).status, TranscriptionStatus.RUNNING)
        self.assertEqual(self.repo.get_transcription_run(run_b.id).status, TranscriptionStatus.RUNNING)
        self.assertEqual(self.repo._rows("SELECT * FROM transcripts", ()), [])
        self.assertEqual(self.repo._rows("SELECT * FROM transcript_segments", ()), [])

    def test_completion_rejects_segment_for_different_transcript(self) -> None:
        run = self._create_run("segment-run")
        transcript = Transcript("expected-transcript", run.id, "pt", "texto", NOW)
        wrong_segment = TranscriptSegment(
            "wrong-segment", "another-transcript", 0, "0", "1", "texto"
        )

        with self.assertRaises(RepositoryError):
            self.repo.complete_transcription_run(run.id, transcript, (wrong_segment,), "pt", "0.98", NOW)

        self.assertEqual(self.repo.get_transcription_run(run.id).status, TranscriptionStatus.RUNNING)
        self.assertEqual(self.repo._rows("SELECT * FROM transcripts", ()), [])
        self.assertEqual(self.repo._rows("SELECT * FROM transcript_segments", ()), [])

    def test_create_transcription_run_rejects_completed_status(self) -> None:
        with self.assertRaises(RepositoryError):
            self.repo.create_transcription_run(
                self._run("completed-at-create", status=TranscriptionStatus.COMPLETED)
            )

        self.assertEqual(self.repo._rows("SELECT * FROM transcription_runs", ()), [])

    def test_create_transcription_run_rejects_failed_status(self) -> None:
        with self.assertRaises(RepositoryError):
            self.repo.create_transcription_run(
                self._run("failed-at-create", status=TranscriptionStatus.FAILED)
            )

        self.assertEqual(self.repo._rows("SELECT * FROM transcription_runs", ()), [])
