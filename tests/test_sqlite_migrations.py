from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from engineering_brain.application.source_registry import SourceRegistryService
from engineering_brain.domain.errors import DuplicateError, RepositoryError
from engineering_brain.domain.models import Metadata, Source
from engineering_brain.infrastructure.local_artifact_store import LocalArtifactStore
from engineering_brain.infrastructure.sqlite_source_registry import (
    CURRENT_SCHEMA_VERSION,
    SqliteSourceRegistryRepository,
)


EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
TIMESTAMP = "2026-09-27T12:00:00+00:00"
LINKED_AT = "2026-09-28T13:00:00+00:00"


class SqliteMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_new_database_is_created_at_current_schema_version(self) -> None:
        database = self.root / "new.sqlite3"
        SqliteSourceRegistryRepository(database).initialize()
        self._assert_current_schema(database)

    def test_phase_one_migration_supports_new_observations_with_active_foreign_key(self) -> None:
        database = self.root / "phase-one.sqlite3"
        self._create_phase_one_database(database)

        repository = SqliteSourceRegistryRepository(database)
        repository.initialize()
        original = repository.get_artifact("artifact")
        self.assertEqual(original.sha256, EMPTY_SHA256)
        self.assertEqual(
            repository.observations_for_material("material")[0].observation.original_location,
            "file:///tmp/original.bin",
        )

        attached = self._attach_new_file(repository, "material", "phase-one-new.bin", b"new phase one data")
        self.assertEqual(attached.observation.artifact_id, attached.artifact.id)
        self._assert_current_schema(database)
        self._assert_observation_fk_targets_active_artifacts(database)

    def test_phase_two_migration_preserves_linked_at_and_active_media_index(self) -> None:
        database = self.root / "phase-two.sqlite3"
        self._create_phase_two_database(database, linked_at=LINKED_AT)

        repository = SqliteSourceRegistryRepository(database)
        repository.initialize()
        observation = repository.observations_for_material("material")[0].observation
        inspection = repository.latest_media_inspection("artifact")

        self.assertEqual(observation.observed_at.isoformat(), LINKED_AT)
        self.assertNotEqual(observation.observed_at.isoformat(), TIMESTAMP)
        self.assertEqual(inspection.id, "inspection")
        self._attach_new_file(repository, "material", "phase-two-new.bin", b"new phase two data")
        self._assert_current_schema(database)
        self._assert_observation_fk_targets_active_artifacts(database)
        with sqlite3.connect(database) as connection:
            table = connection.execute(
                "SELECT tbl_name FROM sqlite_master WHERE type = 'index' "
                "AND name = 'idx_media_inspections_artifact_id'"
            ).fetchone()[0]
        self.assertEqual(table, "media_inspections")

    def test_legacy_schema_without_linked_at_falls_back_to_artifact_observed_at(self) -> None:
        database = self.root / "phase-two-no-linked-at.sqlite3"
        self._create_phase_two_database(database, linked_at=None)

        repository = SqliteSourceRegistryRepository(database)
        repository.initialize()

        self.assertEqual(
            repository.observations_for_material("material")[0].observation.observed_at.isoformat(),
            TIMESTAMP,
        )
        self._assert_current_schema(database)

    def test_migration_rolls_back_all_ddl_and_version_when_copy_fails(self) -> None:
        database = self.root / "broken-phase-two.sqlite3"
        self._create_phase_two_database(database, linked_at=LINKED_AT, invalid_material_reference=True)

        with self.assertRaises(RepositoryError):
            SqliteSourceRegistryRepository(database).initialize()

        with sqlite3.connect(database) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 0)
            self.assertIsNotNone(
                connection.execute("SELECT 1 FROM sqlite_master WHERE name = 'source_artifacts'").fetchone()
            )
            self.assertIsNone(
                connection.execute("SELECT 1 FROM sqlite_master WHERE name = 'source_artifacts_legacy'").fetchone()
            )

    def test_existing_artifact_without_managed_key_is_upgraded_on_reattach(self) -> None:
        database = self.root / "legacy-artifact.sqlite3"
        self._create_phase_one_database(database)
        repository = SqliteSourceRegistryRepository(database)
        repository.initialize()
        service = SourceRegistryService(repository, LocalArtifactStore(self.root / "artifacts"))
        empty_file = self.root / "empty.bin"
        empty_file.write_bytes(b"")

        record, reused = service.attach_local_artifact(material_id="material", path=empty_file)
        artifact = repository.get_artifact("artifact")

        self.assertTrue(reused)
        self.assertEqual(record.artifact.id, "artifact")
        self.assertIsNotNone(artifact.managed_key)
        self.assertTrue(service.verify_artifact("artifact").is_valid)

    def test_newer_schema_is_rejected(self) -> None:
        database = self.root / "future.sqlite3"
        with sqlite3.connect(database) as connection:
            connection.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION + 1}")

        with self.assertRaisesRegex(RepositoryError, "newer than supported"):
            SqliteSourceRegistryRepository(database).initialize()

    def test_invalid_foreign_key_is_not_reported_as_duplicate(self) -> None:
        database = self.root / "current.sqlite3"
        repository = SqliteSourceRegistryRepository(database)
        repository.initialize()
        invalid_source = Source(
            "source", "collection", "Invalid", datetime.now(UTC), "missing-author", None, Metadata.from_mapping()
        )

        with self.assertRaises(RepositoryError) as captured:
            repository.create_source(invalid_source)
        self.assertNotIsInstance(captured.exception, DuplicateError)

    def _create_phase_one_database(self, database: Path) -> None:
        with sqlite3.connect(database) as connection:
            connection.executescript(_COMMON_SCHEMA + """
                CREATE TABLE source_artifacts (id TEXT PRIMARY KEY, storage_uri TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE, byte_size INTEGER NOT NULL, observed_at TEXT NOT NULL, media_type TEXT);
                CREATE TABLE material_artifacts (material_id TEXT NOT NULL REFERENCES materials(id), artifact_id TEXT NOT NULL REFERENCES source_artifacts(id), linked_at TEXT NOT NULL, PRIMARY KEY(material_id, artifact_id));
            """)
            _insert_common_records(connection)
            connection.execute("INSERT INTO source_artifacts VALUES ('artifact', 'file:///tmp/original.bin', ?, 0, ?, 'application/octet-stream')", (EMPTY_SHA256, TIMESTAMP))
            connection.execute("INSERT INTO material_artifacts VALUES ('material', 'artifact', ?)", (LINKED_AT,))

    def _create_phase_two_database(
        self, database: Path, *, linked_at: str | None, invalid_material_reference: bool = False
    ) -> None:
        with sqlite3.connect(database) as connection:
            association = (
                "CREATE TABLE material_artifacts (material_id TEXT NOT NULL REFERENCES materials(id), "
                "artifact_id TEXT NOT NULL REFERENCES source_artifacts(id), linked_at TEXT NOT NULL, "
                "PRIMARY KEY(material_id, artifact_id));"
                if linked_at is not None
                else "CREATE TABLE material_artifacts (material_id TEXT NOT NULL REFERENCES materials(id), artifact_id TEXT NOT NULL REFERENCES source_artifacts(id), PRIMARY KEY(material_id, artifact_id));"
            )
            connection.executescript(_COMMON_SCHEMA + """
                CREATE TABLE source_artifacts (id TEXT PRIMARY KEY, storage_uri TEXT NOT NULL, original_location TEXT NOT NULL, managed_key TEXT, original_filename TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE, byte_size INTEGER NOT NULL, observed_at TEXT NOT NULL, media_type TEXT);
            """ + association + """
                CREATE TABLE media_inspections (id TEXT PRIMARY KEY, artifact_id TEXT NOT NULL REFERENCES source_artifacts(id), tool TEXT NOT NULL, tool_version TEXT NOT NULL, schema_version TEXT NOT NULL, inspected_at TEXT NOT NULL, status TEXT NOT NULL, classification TEXT NOT NULL, format_names_json TEXT NOT NULL, duration_seconds TEXT, bit_rate INTEGER, stream_count INTEGER NOT NULL, video_streams_json TEXT NOT NULL, audio_streams_json TEXT NOT NULL, subtitle_streams_json TEXT NOT NULL, raw_probe_json TEXT NOT NULL, UNIQUE(artifact_id, tool, tool_version, schema_version));
                CREATE INDEX idx_media_inspections_artifact_id ON media_inspections(artifact_id);
            """)
            _insert_common_records(connection)
            connection.execute("INSERT INTO source_artifacts VALUES ('artifact', 'file:///tmp/original.mp4', 'file:///tmp/original.mp4', 'sha256/e3/' || ?, 'original.mp4', ?, 0, ?, 'video/mp4')", (EMPTY_SHA256, EMPTY_SHA256, TIMESTAMP))
            material_id = "missing-material" if invalid_material_reference else "material"
            if linked_at is None:
                connection.execute("PRAGMA foreign_keys = OFF")
                connection.execute("INSERT INTO material_artifacts VALUES (?, 'artifact')", (material_id,))
            else:
                if invalid_material_reference:
                    connection.execute("PRAGMA foreign_keys = OFF")
                connection.execute("INSERT INTO material_artifacts VALUES (?, 'artifact', ?)", (material_id, linked_at))
            connection.execute("INSERT INTO media_inspections VALUES ('inspection', 'artifact', 'ffprobe', 'ffprobe version test', 'media-inspection-v1', ?, 'complete', 'video', '[\"mp4\"]', NULL, NULL, 1, '[]', '[]', '[]', '{}')", (TIMESTAMP,))

    def _attach_new_file(
        self, repository: SqliteSourceRegistryRepository, material_id: str, name: str, content: bytes
    ) -> object:
        path = self.root / name
        path.write_bytes(content)
        return SourceRegistryService(repository, LocalArtifactStore(self.root / "managed")).attach_local_artifact(material_id=material_id, path=path)[0]

    def _assert_current_schema(self, database: Path) -> None:
        with sqlite3.connect(database) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], CURRENT_SCHEMA_VERSION)
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])

    def _assert_observation_fk_targets_active_artifacts(self, database: Path) -> None:
        with sqlite3.connect(database) as connection:
            references = {row[2] for row in connection.execute("PRAGMA foreign_key_list(artifact_observations)")}
        self.assertIn("source_artifacts", references)
        self.assertNotIn("source_artifacts_legacy", references)


_COMMON_SCHEMA = """
    CREATE TABLE authors (id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL, metadata_json TEXT NOT NULL);
    CREATE TABLE sources (id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL, created_at TEXT NOT NULL, author_id TEXT REFERENCES authors(id), origin TEXT, metadata_json TEXT NOT NULL);
    CREATE TABLE materials (id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id), kind TEXT NOT NULL, title TEXT NOT NULL, origin TEXT NOT NULL, revision TEXT NOT NULL, ingested_at TEXT NOT NULL, status TEXT NOT NULL, metadata_json TEXT NOT NULL);
"""


def _insert_common_records(connection: sqlite3.Connection) -> None:
    connection.execute("INSERT INTO authors VALUES ('author', 'Author', ?, '{}')", (TIMESTAMP,))
    connection.execute("INSERT INTO sources VALUES ('source', 'collection', 'Source', ?, 'author', NULL, '{}')", (TIMESTAMP,))
    connection.execute("INSERT INTO materials VALUES ('material', 'source', 'video', 'Material', 'file://source', '1', ?, 'active', '{}')", (TIMESTAMP,))
