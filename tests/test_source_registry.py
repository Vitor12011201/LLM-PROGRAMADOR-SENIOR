from __future__ import annotations

from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from engineering_brain.application.source_registry import SourceRegistryService
from engineering_brain.domain.errors import ArtifactFileNotFoundError, NotFoundError
from engineering_brain.infrastructure.local_artifact_store import LocalArtifactStore
from engineering_brain.infrastructure.sqlite_source_registry import SqliteSourceRegistryRepository


class SourceRegistryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.database_path = self.root / "registry.sqlite3"
        self.repository = SqliteSourceRegistryRepository(self.database_path)
        self.repository.initialize()
        self.artifact_store = LocalArtifactStore(self.root / "managed-artifacts")
        self.service = SourceRegistryService(self.repository, self.artifact_store)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _register_material(self, title: str = "Material") -> str:
        author = self.service.register_author("Generic author", {"organization": "Example"})
        source = self.service.register_source(
            kind="documentation",
            title="Generic documentation",
            author_id=author.id,
            origin="https://example.test/docs",
        )
        material = self.service.register_material(
            source_id=source.id,
            kind="article",
            title=title,
            origin=f"https://example.test/docs/{title.lower()}",
            revision="2026-09",
        )
        return material.id

    def test_register_persist_and_read_full_provenance_chain(self) -> None:
        material_id = self._register_material()
        artifact_path = self.root / "original.txt"
        artifact_path.write_bytes(b"evidence remains separate from interpretation")

        observation_record, reused = self.service.attach_local_artifact(material_id=material_id, path=artifact_path)
        reloaded = SourceRegistryService(
            SqliteSourceRegistryRepository(self.database_path), self.artifact_store
        ).show_material(material_id)

        self.assertFalse(reused)
        self.assertEqual(reloaded.material.id, material_id)
        self.assertEqual(reloaded.source.kind, "documentation")
        self.assertEqual(reloaded.author.name, "Generic author")
        self.assertEqual(reloaded.artifact_observations, (observation_record,))
        artifact = observation_record.artifact
        observation = observation_record.observation
        self.assertEqual(artifact.sha256, "93ad299209d6bc19f33ddbd4dbb70c187a86ef2a6c8ba35aebc153a8553a295d")
        self.assertEqual(observation.original_location, artifact_path.absolute().as_uri())
        self.assertEqual(observation.original_filename, "original.txt")
        self.assertTrue(self.artifact_store.managed_path(artifact.managed_key).is_file())

    def test_same_material_content_and_location_is_idempotent(self) -> None:
        material_id = self._register_material()
        artifact_path = self.root / "original.txt"
        artifact_path.write_text("same bytes", encoding="utf-8")
        first, _ = self.service.attach_local_artifact(material_id=material_id, path=artifact_path)
        second, reused = self.service.attach_local_artifact(material_id=material_id, path=artifact_path)

        self.assertTrue(reused)
        self.assertEqual(first.observation.id, second.observation.id)

    def test_same_bytes_can_be_explicitly_reused_by_another_logical_material(self) -> None:
        first_material_id = self._register_material("First")
        source = self.repository.get_source(self.repository.get_material(first_material_id).source_id)
        second_material = self.service.register_material(
            source_id=source.id,
            kind="article",
            title="Second publication",
            origin="https://example.test/docs/second",
            revision="1",
        )
        artifact_path = self.root / "shared.txt"
        artifact_path.write_text("shared exact evidence", encoding="utf-8")
        first_artifact, first_reused = self.service.attach_local_artifact(
            material_id=first_material_id, path=artifact_path
        )
        second_artifact, second_reused = self.service.attach_local_artifact(
            material_id=second_material.id, path=artifact_path
        )

        self.assertFalse(first_reused)
        self.assertTrue(second_reused)
        self.assertEqual(first_artifact.artifact.id, second_artifact.artifact.id)

    def test_same_content_for_different_materials_preserves_both_observations(self) -> None:
        first_material_id = self._register_material("First")
        source = self.repository.get_source(self.repository.get_material(first_material_id).source_id)
        second_material = self.service.register_material(
            source_id=source.id, kind="video", title="Second", origin="file://second"
        )
        first_path = self.root / "first" / "video.mp4"
        second_path = self.root / "second" / "renamed.mp4"
        first_path.parent.mkdir()
        second_path.parent.mkdir()
        first_path.write_bytes(b"one preserved binary")
        second_path.write_bytes(b"one preserved binary")

        first, _ = self.service.attach_local_artifact(material_id=first_material_id, path=first_path)
        second, reused = self.service.attach_local_artifact(material_id=second_material.id, path=second_path)

        self.assertTrue(reused)
        self.assertEqual(first.artifact.id, second.artifact.id)
        self.assertEqual(first.artifact.managed_key, second.artifact.managed_key)
        self.assertEqual(first.observation.original_location, first_path.absolute().as_uri())
        self.assertEqual(second.observation.original_location, second_path.absolute().as_uri())
        self.assertEqual(first.observation.original_filename, "video.mp4")
        self.assertEqual(second.observation.original_filename, "renamed.mp4")

    def test_same_content_and_material_from_two_locations_creates_two_observations(self) -> None:
        material_id = self._register_material()
        first_path = self.root / "first.bin"
        second_path = self.root / "second.bin"
        first_path.write_bytes(b"same bytes, two observations")
        second_path.write_bytes(b"same bytes, two observations")

        first, _ = self.service.attach_local_artifact(material_id=material_id, path=first_path)
        second, reused = self.service.attach_local_artifact(material_id=material_id, path=second_path)

        self.assertTrue(reused)
        self.assertEqual(first.artifact.id, second.artifact.id)
        observations = self.service.show_material(material_id).artifact_observations
        self.assertEqual(len(observations), 2)
        self.assertEqual({item.observation.original_location for item in observations}, {first_path.absolute().as_uri(), second_path.absolute().as_uri()})

    def test_managed_copy_survives_removal_of_the_external_original(self) -> None:
        material_id = self._register_material()
        original = self.root / "external.pdf"
        original.write_bytes(b"preserved source evidence")
        artifact_record, _ = self.service.attach_local_artifact(material_id=material_id, path=original)
        artifact = artifact_record.artifact

        original.unlink()
        verification = self.service.verify_artifact(artifact.id)

        self.assertFalse(original.exists())
        self.assertTrue(self.artifact_store.managed_path(artifact.managed_key).is_file())
        self.assertTrue(verification.exists)
        self.assertTrue(verification.is_valid)

    def test_identical_external_files_share_one_managed_blob(self) -> None:
        first_material_id = self._register_material("First")
        source = self.repository.get_source(self.repository.get_material(first_material_id).source_id)
        second_material = self.service.register_material(
            source_id=source.id, kind="article", title="Second", origin="https://example.test/second"
        )
        first_path = self.root / "first.bin"
        second_path = self.root / "renamed-copy.bin"
        first_path.write_bytes(b"identical binary content")
        second_path.write_bytes(b"identical binary content")

        first_artifact, _ = self.service.attach_local_artifact(material_id=first_material_id, path=first_path)
        second_artifact, reused = self.service.attach_local_artifact(material_id=second_material.id, path=second_path)

        self.assertTrue(reused)
        self.assertEqual(first_artifact.artifact.id, second_artifact.artifact.id)
        self.assertEqual(first_artifact.artifact.managed_key, second_artifact.artifact.managed_key)
        blobs = list((self.root / "managed-artifacts" / "sha256").glob("*/*"))
        self.assertEqual(blobs, [self.artifact_store.managed_path(first_artifact.artifact.managed_key)])

    def test_different_contents_produce_different_hashes_and_managed_blobs(self) -> None:
        first_material_id = self._register_material("First")
        source = self.repository.get_source(self.repository.get_material(first_material_id).source_id)
        second_material = self.service.register_material(
            source_id=source.id, kind="article", title="Second", origin="https://example.test/second"
        )
        first_path = self.root / "first.bin"
        second_path = self.root / "second.bin"
        first_path.write_bytes(b"first content")
        second_path.write_bytes(b"second content")

        first_artifact, _ = self.service.attach_local_artifact(material_id=first_material_id, path=first_path)
        second_artifact, _ = self.service.attach_local_artifact(material_id=second_material.id, path=second_path)

        self.assertNotEqual(first_artifact.artifact.sha256, second_artifact.artifact.sha256)
        self.assertNotEqual(first_artifact.artifact.managed_key, second_artifact.artifact.managed_key)
        self.assertTrue(self.artifact_store.managed_path(first_artifact.artifact.managed_key).is_file())
        self.assertTrue(self.artifact_store.managed_path(second_artifact.artifact.managed_key).is_file())

    def test_integrity_check_detects_a_corrupted_managed_blob(self) -> None:
        material_id = self._register_material()
        original = self.root / "evidence.txt"
        original.write_bytes(b"original content")
        artifact_record, _ = self.service.attach_local_artifact(material_id=material_id, path=original)
        artifact = artifact_record.artifact
        managed_path = self.artifact_store.managed_path(artifact.managed_key)

        managed_path.chmod(0o644)
        managed_path.write_bytes(b"corrupted content")
        verification = self.service.verify_artifact(artifact.id)

        self.assertTrue(verification.exists)
        self.assertFalse(verification.is_valid)
        self.assertNotEqual(verification.actual_sha256, artifact.sha256)

    def test_phase_one_artifact_row_migrates_without_losing_content_identity(self) -> None:
        legacy_database = self.root / "legacy.sqlite3"
        with sqlite3.connect(legacy_database) as connection:
            connection.execute(
                """CREATE TABLE source_artifacts (
                    id TEXT PRIMARY KEY, storage_uri TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE,
                    byte_size INTEGER NOT NULL, observed_at TEXT NOT NULL, media_type TEXT
                )"""
            )
            connection.execute(
                """INSERT INTO source_artifacts
                   (id, storage_uri, sha256, byte_size, observed_at, media_type)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    "legacy-artifact",
                    "file:///tmp/legacy-source.txt",
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    0,
                    "2026-09-27T12:00:00+00:00",
                    "text/plain",
                ),
            )

        migrated_repository = SqliteSourceRegistryRepository(legacy_database)
        migrated_repository.initialize()
        artifact = migrated_repository.get_artifact("legacy-artifact")

        self.assertIsNone(artifact.managed_key)
        self.assertEqual(artifact.sha256, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")

    def test_invalid_references_and_missing_files_produce_expected_errors(self) -> None:
        with self.assertRaises(NotFoundError):
            self.service.register_source(kind="repository", title="Missing author", author_id="missing")
        with self.assertRaises(NotFoundError):
            self.service.register_material(
                source_id="missing", kind="article", title="Missing source", origin="https://example.test/missing"
            )
        material_id = self._register_material()
        with self.assertRaises(ArtifactFileNotFoundError):
            self.service.attach_local_artifact(material_id=material_id, path=self.root / "absent.txt")

    def test_list_materials_is_ordered_and_reads_persisted_records(self) -> None:
        first_id = self._register_material("First")
        source = self.repository.get_source(self.repository.get_material(first_id).source_id)
        second = self.service.register_material(
            source_id=source.id, kind="article", title="Second", origin="https://example.test/second"
        )
        listed = self.service.list_materials()
        self.assertEqual({item.id for item in listed}, {first_id, second.id})
