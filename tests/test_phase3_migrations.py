from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from engineering_brain.application.source_registry import SourceRegistryService
from engineering_brain.application.transcription import TranscriptionConfig, TranscriptionService
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
from engineering_brain.infrastructure import sqlite_source_registry as sqlite_registry
from engineering_brain.infrastructure.local_artifact_store import LocalArtifactStore
from engineering_brain.infrastructure.sqlite_source_registry import (
    CURRENT_SCHEMA_VERSION,
    SqliteSourceRegistryRepository,
)
from tests.support.fake_transcriber import FakeTranscriber


V3_TIMESTAMP = "2026-09-28T12:00:00+00:00"
V3_OBSERVED_AT = "2026-09-28T12:05:00+00:00"
V3_INSPECTED_AT = "2026-09-28T12:10:00+00:00"
NOW = datetime(2026, 9, 28, 13, 0, tzinfo=UTC)
DERIVATION_CONFIG = '{"channels":1,"codec":"pcm_s16le","container":"wav","sample_rate":16000}'


class PhaseThreeMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_v3_database_migrates_to_v4_without_losing_phase21_data(self) -> None:
        database, _, stored = self._create_real_v3_database("migrate.sqlite")
        repository = SqliteSourceRegistryRepository(database)
        repository.initialize()

        self.assertEqual(self._user_version(database), 4)
        self.assertEqual(self._foreign_key_check(database), [])
        author = repository.get_author("legacy-author")
        source = repository.get_source("legacy-source")
        material = repository.get_material("legacy-material")
        artifact = repository.get_artifact("legacy-artifact")
        observation = repository.observations_for_material("legacy-material")[0].observation
        inspection = repository.latest_media_inspection("legacy-artifact")

        self.assertEqual(author.name, "Legacy Author")
        self.assertEqual(source.author_id, author.id)
        self.assertEqual(material.source_id, source.id)
        self.assertEqual(artifact.sha256, stored.sha256)
        self.assertEqual(artifact.managed_key, stored.managed_key)
        self.assertEqual(artifact.preserved_at.isoformat(), V3_TIMESTAMP)
        self.assertEqual(observation.original_location, "file:///legacy/source.mp4")
        self.assertEqual(observation.original_filename, "source.mp4")
        self.assertEqual(observation.media_type, "video/mp4")
        self.assertEqual(observation.observed_at.isoformat(), V3_OBSERVED_AT)
        self.assertEqual(inspection.tool, "ffprobe")
        self.assertEqual(inspection.tool_version, "ffprobe version n8.1.2")
        self.assertEqual(inspection.inspected_at.isoformat(), V3_INSPECTED_AT)
        self.assertEqual(inspection.audio_streams[0].codec, "aac")
        self.assertEqual(inspection.audio_streams[0].sample_rate, 48000)

    def test_v3_migrated_database_supports_phase3_pipeline(self) -> None:
        database, store, _ = self._create_real_v3_database("pipeline.sqlite")
        repository = SqliteSourceRegistryRepository(database)
        repository.initialize()

        selection = AudioSelection(
            "selection-after-migration",
            "legacy-artifact",
            "legacy-inspection",
            1,
            "EXPLICIT_STREAM_INDEX",
            "selected legacy audio stream",
            NOW,
        )
        repository.create_audio_selection(selection)
        derived_file = self.root / "derived-after-migration.wav"
        derived_file.write_bytes(b"derived audio bytes")
        stored = store.ingest(derived_file)
        derived = DerivedAudioArtifact(
            "derived-after-migration",
            selection.id,
            stored.managed_key,
            stored.sha256,
            stored.byte_size,
            "ffmpeg",
            "n8.1.2",
            "audio-derivation-v1",
            DERIVATION_CONFIG,
            NOW,
        )
        repository.create_derived_audio_artifact(derived)

        run, reused = TranscriptionService(repository, store, FakeTranscriber()).transcribe(
            derived.id, TranscriptionConfig()
        )
        transcript = repository.get_transcript_for_run(run.id)
        segments = repository.segments_for_transcript(transcript.id)

        self.assertFalse(reused)
        self.assertEqual(run.status, TranscriptionStatus.COMPLETED)
        self.assertEqual(transcript.transcription_run_id, run.id)
        self.assertEqual([segment.ordinal for segment in segments], [0, 1])
        self.assertEqual(self._foreign_key_check(database), [])

    def test_v3_to_v4_migration_preserves_foreign_key_integrity(self) -> None:
        database, _, _ = self._create_real_v3_database("fk.sqlite")
        SqliteSourceRegistryRepository(database).initialize()

        self.assertEqual(self._foreign_key_check(database), [])
        targets = self._foreign_key_targets(database)
        self.assertEqual(targets["audio_selections"], {"source_artifacts", "media_inspections"})
        self.assertEqual(targets["derived_audio_artifacts"], {"audio_selections"})
        self.assertEqual(targets["transcription_runs"], {"derived_audio_artifacts"})
        self.assertEqual(targets["transcripts"], {"transcription_runs"})
        self.assertEqual(targets["transcript_segments"], {"transcripts"})

    def test_v3_to_v4_migration_has_no_legacy_fk_targets(self) -> None:
        database, _, _ = self._create_real_v3_database("no-legacy-fks.sqlite")
        SqliteSourceRegistryRepository(database).initialize()

        for targets in self._foreign_key_targets(database).values():
            self.assertFalse(any(target.endswith("_legacy") for target in targets))

    def test_v3_to_v4_migration_creates_expected_indexes_and_constraints(self) -> None:
        database, _, _ = self._create_real_v3_database("indexes.sqlite")
        SqliteSourceRegistryRepository(database).initialize()

        with sqlite3.connect(database) as connection:
            indexes = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index'"
                )
            }
            self.assertIn("idx_audio_selections_artifact_id", indexes)
            self.assertIn("idx_derived_audio_selection_id", indexes)
            self.assertIn("idx_transcription_runs_derived_audio_id", indexes)
            self.assertTrue(self._has_unique_columns(connection, "audio_selections", ("source_artifact_id", "media_inspection_id", "stream_index")))
            self.assertTrue(self._has_unique_columns(connection, "derived_audio_artifacts", ("audio_selection_id", "extractor", "extractor_version", "derivation_schema_version", "config_json")))
            self.assertTrue(self._has_unique_columns(connection, "transcripts", ("transcription_run_id",)))
            self.assertTrue(self._has_unique_columns(connection, "transcript_segments", ("transcript_id", "ordinal")))

    def test_v3_to_v4_migration_rolls_back_on_failure(self) -> None:
        database, _, _ = self._create_real_v3_database("rollback.sqlite")

        def fail_after_first_v4_statement(connection: sqlite3.Connection) -> None:
            connection.execute(sqlite_registry._AUDIO_SELECTIONS)
            raise sqlite3.OperationalError("forced phase 3 migration failure")

        with patch.object(sqlite_registry, "_migrate_phase_three_to_four", fail_after_first_v4_statement):
            with self.assertRaisesRegex(Exception, "could not initialize SQLite registry"):
                SqliteSourceRegistryRepository(database).initialize()

        self.assertEqual(self._user_version(database), 3)
        with sqlite3.connect(database) as connection:
            self.assertIsNone(
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'audio_selections'"
                ).fetchone()
            )
            self.assertIsNotNone(
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'media_inspections'"
                ).fetchone()
            )
            self.assertEqual(
                connection.execute("SELECT id FROM source_artifacts WHERE id = 'legacy-artifact'").fetchone()[0],
                "legacy-artifact",
            )

    def test_fresh_database_initializes_directly_at_schema_v4(self) -> None:
        database = self.root / "fresh.sqlite"
        SqliteSourceRegistryRepository(database).initialize()

        self.assertEqual(self._user_version(database), CURRENT_SCHEMA_VERSION)
        self.assertEqual(self._foreign_key_check(database), [])
        with sqlite3.connect(database) as connection:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
        self.assertTrue(
            {
                "authors", "sources", "materials", "source_artifacts", "artifact_observations",
                "media_inspections", "audio_selections", "derived_audio_artifacts",
                "transcription_runs", "transcripts", "transcript_segments",
            }.issubset(tables)
        )

    def test_fresh_schema_v4_has_expected_foreign_keys(self) -> None:
        database = self.root / "fresh-fks.sqlite"
        SqliteSourceRegistryRepository(database).initialize()
        targets = self._foreign_key_targets(database)

        self.assertEqual(targets["audio_selections"], {"source_artifacts", "media_inspections"})
        self.assertEqual(targets["derived_audio_artifacts"], {"audio_selections"})
        self.assertEqual(targets["transcription_runs"], {"derived_audio_artifacts"})
        self.assertEqual(targets["transcripts"], {"transcription_runs"})
        self.assertEqual(targets["transcript_segments"], {"transcripts"})

    def test_fresh_schema_v4_enforces_transcript_and_segment_uniqueness(self) -> None:
        database, store, _ = self._create_current_pipeline_database("fresh-constraints.sqlite")
        repository = SqliteSourceRegistryRepository(database)
        run = self._create_running_run(repository, "fresh-constraint-run")
        transcript = Transcript("fresh-constraint-transcript", run.id, "pt", "texto", NOW)
        first = TranscriptSegment("fresh-constraint-segment", transcript.id, 0, "0", "1", "primeiro")
        repository.complete_transcription_run(run.id, transcript, (first,), "pt", "0.98", NOW)

        with self.assertRaises(sqlite3.IntegrityError):
            with repository._connection() as connection:
                connection.execute(
                    "INSERT INTO transcripts (id, transcription_run_id, language, full_text, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("fresh-second-transcript", run.id, "pt", "segundo", NOW.isoformat()),
                )
        with self.assertRaises(sqlite3.IntegrityError):
            with repository._connection() as connection:
                connection.execute(
                    "INSERT INTO transcript_segments "
                    "(id, transcript_id, ordinal, start_seconds, end_seconds, text) VALUES (?, ?, ?, ?, ?, ?)",
                    ("fresh-second-segment", transcript.id, 0, "1", "2", "duplicado"),
                )
        self.assertTrue(store.verify(self._derived_for(repository).managed_key, self._derived_for(repository).sha256).matches_expected_hash)

    def _create_real_v3_database(self, name: str) -> tuple[Path, LocalArtifactStore, object]:
        database = self.root / name
        store = LocalArtifactStore(self.root / f"{name}-artifacts")
        blob = self.root / f"{name}-legacy-source.mp4"
        blob.write_bytes(b"legacy phase 2.1 source bytes")
        stored = store.ingest(blob)
        with sqlite3.connect(database) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(_PHASE_21_V3_SCHEMA)
            connection.execute(
                "INSERT INTO authors (id, name, created_at, metadata_json) VALUES (?, ?, ?, ?)",
                ("legacy-author", "Legacy Author", V3_TIMESTAMP, "{}"),
            )
            connection.execute(
                "INSERT INTO sources (id, kind, title, created_at, author_id, origin, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("legacy-source", "channel", "Legacy Source", V3_TIMESTAMP, "legacy-author", "file://legacy", "{}"),
            )
            connection.execute(
                "INSERT INTO materials "
                "(id, source_id, kind, title, origin, revision, ingested_at, status, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("legacy-material", "legacy-source", "video", "Legacy Material", "file://legacy/source.mp4", "1", V3_TIMESTAMP, "registered", "{}"),
            )
            connection.execute(
                "INSERT INTO source_artifacts (id, managed_key, sha256, byte_size, preserved_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("legacy-artifact", stored.managed_key, stored.sha256, stored.byte_size, V3_TIMESTAMP),
            )
            connection.execute(
                "INSERT INTO artifact_observations "
                "(id, material_id, artifact_id, original_location, original_filename, media_type, observed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("legacy-observation", "legacy-material", "legacy-artifact", "file:///legacy/source.mp4", "source.mp4", "video/mp4", V3_OBSERVED_AT),
            )
            connection.execute(
                "INSERT INTO media_inspections "
                "(id, artifact_id, tool, tool_version, schema_version, inspected_at, status, classification, "
                "format_names_json, duration_seconds, bit_rate, stream_count, video_streams_json, "
                "audio_streams_json, subtitle_streams_json, raw_probe_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "legacy-inspection", "legacy-artifact", "ffprobe", "ffprobe version n8.1.2",
                    "media-inspection-v1", V3_INSPECTED_AT, "completed", "audio_video", '["mov","mp4"]',
                    "12.0", 128000, 2,
                    '[{"index":0,"codec":"h264","width":1920,"height":1080,"frame_rate":"30/1","pixel_format":"yuv420p","duration_seconds":"12.0","is_default":true,"language":"por"}]',
                    '[{"index":1,"codec":"aac","sample_rate":48000,"channels":2,"channel_layout":"stereo","duration_seconds":"12.0","bit_rate":128000,"is_default":true,"language":"por"}]',
                    "[]", "{}",
                ),
            )
            connection.execute("PRAGMA user_version = 3")
        return database, store, stored

    def _create_current_pipeline_database(self, name: str) -> tuple[Path, LocalArtifactStore, object]:
        database = self.root / name
        repository = SqliteSourceRegistryRepository(database)
        repository.initialize()
        store = LocalArtifactStore(self.root / f"{name}-artifacts")
        registry = SourceRegistryService(repository, store)
        author = registry.register_author("Fresh author")
        source = registry.register_source(kind="channel", title="Fresh source", author_id=author.id)
        material = registry.register_material(
            source_id=source.id,
            kind="video",
            title="Fresh material",
            origin="file://fresh/source.mp4",
        )
        source_blob = self.root / f"{name}-source.mp4"
        source_blob.write_bytes(b"fresh source bytes")
        artifact = registry.attach_local_artifact(material_id=material.id, path=source_blob)[0].artifact
        inspection = MediaInspection(
            "fresh-inspection",
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
        repository.create_media_inspection(inspection)
        selection = AudioSelection("fresh-selection", artifact.id, inspection.id, 0, "ONLY_AUDIO_STREAM", "selection", NOW)
        repository.create_audio_selection(selection)
        derived_blob = self.root / f"{name}-derived.wav"
        derived_blob.write_bytes(b"fresh derived blob")
        derived_stored = store.ingest(derived_blob)
        repository.create_derived_audio_artifact(
            DerivedAudioArtifact("fresh-derived", selection.id, derived_stored.managed_key, derived_stored.sha256, derived_stored.byte_size, "ffmpeg", "test", "audio-derivation-v1", DERIVATION_CONFIG, NOW)
        )
        return database, store, derived_stored

    def _derived_for(self, repository: SqliteSourceRegistryRepository):
        return repository.get_derived_audio_artifact("fresh-derived")

    def _create_running_run(self, repository: SqliteSourceRegistryRepository, run_id: str) -> TranscriptionRun:
        run = TranscriptionRun(run_id, "fresh-derived", "fake", "1", "model", "revision", "cpu", "int8", TranscriptionConfig().json_value, None, None, None, NOW, None, TranscriptionStatus.RUNNING)
        repository.create_transcription_run(run)
        return run

    def _user_version(self, database: Path) -> int:
        with sqlite3.connect(database) as connection:
            return connection.execute("PRAGMA user_version").fetchone()[0]

    def _foreign_key_check(self, database: Path) -> list[tuple]:
        with sqlite3.connect(database) as connection:
            return connection.execute("PRAGMA foreign_key_check").fetchall()

    def _foreign_key_targets(self, database: Path) -> dict[str, set[str]]:
        tables = (
            "audio_selections", "derived_audio_artifacts", "transcription_runs", "transcripts", "transcript_segments",
        )
        with sqlite3.connect(database) as connection:
            return {table: {row[2] for row in connection.execute(f"PRAGMA foreign_key_list({table})")} for table in tables}

    def _has_unique_columns(self, connection: sqlite3.Connection, table: str, columns: tuple[str, ...]) -> bool:
        for index in connection.execute(f"PRAGMA index_list({table})"):
            if index[2] and tuple(row[2] for row in connection.execute(f"PRAGMA index_info({index[1]})")) == columns:
                return True
        return False


_PHASE_21_V3_SCHEMA = """
CREATE TABLE authors (id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL, metadata_json TEXT NOT NULL);
CREATE TABLE sources (id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL, created_at TEXT NOT NULL, author_id TEXT REFERENCES authors(id), origin TEXT, metadata_json TEXT NOT NULL);
CREATE TABLE materials (id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id), kind TEXT NOT NULL, title TEXT NOT NULL, origin TEXT NOT NULL, revision TEXT NOT NULL, ingested_at TEXT NOT NULL, status TEXT NOT NULL, metadata_json TEXT NOT NULL);
CREATE TABLE source_artifacts (id TEXT PRIMARY KEY, managed_key TEXT, sha256 TEXT NOT NULL UNIQUE, byte_size INTEGER NOT NULL, preserved_at TEXT NOT NULL);
CREATE TABLE artifact_observations (id TEXT PRIMARY KEY, material_id TEXT NOT NULL REFERENCES materials(id), artifact_id TEXT NOT NULL REFERENCES source_artifacts(id), original_location TEXT NOT NULL, original_filename TEXT, media_type TEXT, observed_at TEXT NOT NULL, UNIQUE(material_id, artifact_id, original_location));
CREATE TABLE media_inspections (id TEXT PRIMARY KEY, artifact_id TEXT NOT NULL REFERENCES source_artifacts(id), tool TEXT NOT NULL, tool_version TEXT NOT NULL, schema_version TEXT NOT NULL, inspected_at TEXT NOT NULL, status TEXT NOT NULL, classification TEXT NOT NULL, format_names_json TEXT NOT NULL, duration_seconds TEXT, bit_rate INTEGER, stream_count INTEGER NOT NULL, video_streams_json TEXT NOT NULL, audio_streams_json TEXT NOT NULL, subtitle_streams_json TEXT NOT NULL, raw_probe_json TEXT NOT NULL, UNIQUE(artifact_id, tool, tool_version, schema_version));
CREATE INDEX idx_sources_author_id ON sources(author_id);
CREATE INDEX idx_materials_source_id ON materials(source_id);
CREATE INDEX idx_artifact_observations_material_id ON artifact_observations(material_id);
CREATE INDEX idx_media_inspections_artifact_id ON media_inspections(artifact_id);
"""
