from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from engineering_brain.application.audio_pipeline import (
    AUDIO_DERIVATION_SCHEMA_VERSION,
    CANONICAL_AUDIO_CONFIG_JSON,
    AudioPipelineService,
)
from engineering_brain.application.media_inspection import MediaInspectionService
from engineering_brain.application.source_registry import SourceRegistryService
from engineering_brain.domain.errors import (
    AmbiguousAudioSelectionError,
    ArtifactIntegrityError,
    AudioExtractionError,
    InvalidAudioSelectionError,
    InvalidAudioStreamError,
    NoAudioStreamError,
    RepositoryError,
)
from engineering_brain.domain.models import (
    AudioStream,
    DerivedAudioArtifact,
    MediaClassification,
    MediaInspection,
)
from engineering_brain.infrastructure.ffmpeg_audio_extractor import FfmpegAudioExtractor
from engineering_brain.infrastructure.ffprobe_media_inspector import FfprobeMediaInspector
from engineering_brain.infrastructure.local_artifact_store import LocalArtifactStore
from engineering_brain.infrastructure.sqlite_source_registry import SqliteSourceRegistryRepository


class FakeExtractor:
    def __init__(self, version="ffmpeg fixture", *, output_bytes=b"RIFFfixture deterministic wav", fail_after_output=False):
        self.version, self.output_bytes, self.fail_after_output = version, output_bytes, fail_after_output
        self.calls = 0
        self.source_paths: list[Path] = []
        self.destination_paths: list[Path] = []

    def tool_identity(self): return "ffmpeg", self.version

    def extract(self, source_path, stream_index, destination_path):
        self.calls += 1; self.source_paths.append(source_path); self.destination_paths.append(destination_path)
        destination_path.write_bytes(self.output_bytes)
        if self.fail_after_output: raise AudioExtractionError("forced extraction failure")


class TrackingTemporaryDirectory:
    def __init__(self, root): self.root, self.path = root, None
    def __enter__(self):
        self.path = Path(tempfile.mkdtemp(prefix="engineering-brain-audio-test-", dir=self.root))
        return str(self.path)
    def __exit__(self, exc_type, exc, traceback): shutil.rmtree(self.path, ignore_errors=True)


class AudioPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(); self.root = Path(self.temp.name)
        self.repository = SqliteSourceRegistryRepository(self.root / "db.sqlite3"); self.repository.initialize()
        self.store = LocalArtifactStore(self.root / "artifacts")
        self.registry = SourceRegistryService(self.repository, self.store)
        author = self.registry.register_author("Author")
        source = self.registry.register_source(kind="video", title="Source", author_id=author.id)
        self.material = self.registry.register_material(source_id=source.id, kind="video", title="Material", origin="file://source")
        original = self.root / "source.bin"; original.write_bytes(b"source")
        self.artifact = self.registry.attach_local_artifact(material_id=self.material.id, path=original)[0].artifact
        self._inspection_number = 0

    def tearDown(self): self.temp.cleanup()

    def _inspection(self, streams, artifact_id=None):
        self._inspection_number += 1
        inspection = MediaInspection(
            f"inspection-{self._inspection_number}", artifact_id or self.artifact.id, "fixture", str(self._inspection_number), "v1",
            datetime.now(UTC), "completed", MediaClassification.AUDIO, (), None, None, len(streams), (), streams, (), "{}",
        )
        self.repository.create_media_inspection(inspection)
        return inspection

    def _single_audio_selection(self, extractor=None):
        service = AudioPipelineService(self.repository, self.store, extractor or FakeExtractor())
        inspection = self._inspection((AudioStream(1, "aac", 48000, 2, None, None, None, False, None),))
        return service, service.select_audio(self.artifact.id, inspection.id)[0]

    def _derived_count(self): return len(self.repository._rows("SELECT * FROM derived_audio_artifacts", ()))

    def test_audio_selection_rules_and_idempotency(self):
        service = AudioPipelineService(self.repository, self.store, FakeExtractor())
        no_audio = self._inspection(())
        with self.assertRaises(NoAudioStreamError): service.select_audio(self.artifact.id, no_audio.id)
        one = self._inspection((AudioStream(3, "aac", 48000, 2, None, None, None, False, None),))
        selection, reused = service.select_audio(self.artifact.id, one.id); again, reused_again = service.select_audio(self.artifact.id, one.id)
        self.assertFalse(reused); self.assertTrue(reused_again); self.assertEqual(selection.id, again.id); self.assertEqual(selection.stream_index, 3)
        many = self._inspection((AudioStream(1, "aac", None, None, None, None, None, False, None), AudioStream(2, "aac", None, None, None, None, None, False, None)))
        with self.assertRaises(AmbiguousAudioSelectionError): service.select_audio(self.artifact.id, many.id)
        self.assertEqual(service.select_audio(self.artifact.id, many.id, 2)[0].stream_index, 2)
        with self.assertRaises(InvalidAudioStreamError): service.select_audio(self.artifact.id, many.id, 9)

    def test_video_stream_index_is_rejected_for_audio_selection(self):
        service = AudioPipelineService(self.repository, self.store, FakeExtractor())
        inspection = self._inspection((AudioStream(2, "aac", None, None, None, None, None, False, None),))
        with self.assertRaises(InvalidAudioStreamError): service.select_audio(self.artifact.id, inspection.id, stream_index=0)

    def test_media_inspection_for_different_artifact_is_rejected(self):
        other_file = self.root / "other.bin"; other_file.write_bytes(b"other source")
        other = self.registry.attach_local_artifact(material_id=self.material.id, path=other_file)[0].artifact
        inspection = self._inspection((AudioStream(1, "aac", None, None, None, None, None, False, None),), artifact_id=other.id)
        with self.assertRaisesRegex(InvalidAudioSelectionError, "does not belong"):
            AudioPipelineService(self.repository, self.store, FakeExtractor()).select_audio(self.artifact.id, inspection.id)

    def test_missing_existing_derived_artifact_is_rejected_before_reextraction(self):
        extractor = FakeExtractor(); service, selection = self._single_audio_selection(extractor)
        derived, _ = service.derive_audio(selection.id)
        self.store.managed_path(derived.managed_key).unlink()
        with self.assertRaises(ArtifactIntegrityError): service.derive_audio(selection.id)
        self.assertEqual(extractor.calls, 1); self.assertEqual(self._derived_count(), 1)

    def test_corrupted_existing_derived_artifact_is_rejected_before_reextraction(self):
        extractor = FakeExtractor(); service, selection = self._single_audio_selection(extractor)
        derived, _ = service.derive_audio(selection.id)
        blob = self.store.managed_path(derived.managed_key); blob.chmod(0o644); blob.write_bytes(b"corrupted derived")
        with self.assertRaises(ArtifactIntegrityError): service.derive_audio(selection.id)
        self.assertEqual(extractor.calls, 1); self.assertEqual(self._derived_count(), 1)

    def test_source_integrity_is_verified_before_ffmpeg_execution(self):
        extractor = FakeExtractor(); service, selection = self._single_audio_selection(extractor)
        blob = self.store.managed_path(self.artifact.managed_key); blob.chmod(0o644); blob.write_bytes(b"corrupted source")
        with self.assertRaises(ArtifactIntegrityError): service.derive_audio(selection.id)
        self.assertEqual(extractor.calls, 0); self.assertEqual(self._derived_count(), 0)

    def test_missing_source_blob_is_rejected_before_ffmpeg_execution(self):
        extractor = FakeExtractor(); service, selection = self._single_audio_selection(extractor)
        self.store.managed_path(self.artifact.managed_key).unlink()
        with self.assertRaises(ArtifactIntegrityError): service.derive_audio(selection.id)
        self.assertEqual(extractor.calls, 0); self.assertEqual(self._derived_count(), 0)

    def test_derived_audio_output_is_stored_in_content_addressed_store(self):
        output = b"canonical wav payload"; extractor = FakeExtractor(output_bytes=output); service, selection = self._single_audio_selection(extractor)
        derived, reused = service.derive_audio(selection.id)
        self.assertFalse(reused); self.assertTrue(self.store.managed_path(derived.managed_key).is_file())
        self.assertEqual(derived.sha256, hashlib.sha256(output).hexdigest()); self.assertEqual(derived.byte_size, len(output))
        self.assertEqual(derived.managed_key, f"sha256/{derived.sha256[:2]}/{derived.sha256}")
        self.assertEqual(extractor.source_paths, [self.store.path_for_read(self.artifact.managed_key)])

    def test_derived_audio_is_not_registered_as_source_artifact(self):
        service, selection = self._single_audio_selection(); source_count = len(self.repository._rows("SELECT * FROM source_artifacts", ()))
        derived, _ = service.derive_audio(selection.id)
        self.assertEqual(len(self.repository._rows("SELECT * FROM source_artifacts", ())), source_count); self.assertEqual(self._derived_count(), 1)
        self.assertNotIn(derived.id, {row["id"] for row in self.repository._rows("SELECT * FROM source_artifacts", ())})

    def test_derived_audio_persists_extractor_provenance(self):
        extractor = FakeExtractor(version="ffmpeg n-test"); service, selection = self._single_audio_selection(extractor)
        derived, _ = service.derive_audio(selection.id)
        self.assertEqual(derived.audio_selection_id, selection.id); self.assertEqual(derived.extractor, "ffmpeg"); self.assertEqual(derived.extractor_version, "ffmpeg n-test")
        self.assertEqual(derived.derivation_schema_version, AUDIO_DERIVATION_SCHEMA_VERSION); self.assertEqual(derived.config_json, CANONICAL_AUDIO_CONFIG_JSON)
        self.assertEqual(json.loads(derived.config_json), {"channels": 1, "codec": "pcm_s16le", "container": "wav", "sample_rate": 16000})
        self.assertTrue(self.store.verify(derived.managed_key, derived.sha256).matches_expected_hash)

    def test_identical_derivation_reuses_existing_artifact(self):
        extractor = FakeExtractor(); service, selection = self._single_audio_selection(extractor)
        first, reused = service.derive_audio(selection.id); second, reused_second = service.derive_audio(selection.id)
        self.assertFalse(reused); self.assertTrue(reused_second); self.assertEqual(first.id, second.id); self.assertEqual(extractor.calls, 1); self.assertEqual(self._derived_count(), 1)

    def test_different_derivation_config_creates_new_artifact(self):
        service, selection = self._single_audio_selection(); first, _ = service.derive_audio(selection.id)
        alternate_config = json.dumps({"channels": 1, "codec": "pcm_s16le", "container": "wav", "sample_rate": 8000}, sort_keys=True, separators=(",", ":"))
        alternative = DerivedAudioArtifact("derived-other-config", selection.id, first.managed_key, first.sha256, first.byte_size, first.extractor, first.extractor_version, first.derivation_schema_version, alternate_config, datetime.now(UTC))
        self.repository.create_derived_audio_artifact(alternative)
        self.assertNotEqual(first.id, alternative.id); self.assertNotEqual(first.config_json, alternative.config_json)
        self.assertEqual(self.repository.find_derived_audio_artifact(selection.id, first.extractor, first.extractor_version, first.derivation_schema_version, alternate_config).id, alternative.id)

    def test_different_extractor_version_creates_new_derivation(self):
        first_extractor = FakeExtractor(version="ffmpeg n1"); first_service, selection = self._single_audio_selection(first_extractor)
        first, _ = first_service.derive_audio(selection.id); second_extractor = FakeExtractor(version="ffmpeg n2")
        second, reused = AudioPipelineService(self.repository, self.store, second_extractor).derive_audio(selection.id)
        self.assertFalse(reused); self.assertNotEqual(first.id, second.id); self.assertEqual(first_extractor.calls, 1); self.assertEqual(second_extractor.calls, 1)

    def test_same_output_bytes_can_share_cas_without_losing_derivation_provenance(self):
        first_extractor = FakeExtractor(version="ffmpeg n1", output_bytes=b"same wav"); first_service, first_selection = self._single_audio_selection(first_extractor)
        first, _ = first_service.derive_audio(first_selection.id)
        other_file = self.root / "other-source.bin"; other_file.write_bytes(b"other source")
        other_artifact = self.registry.attach_local_artifact(material_id=self.material.id, path=other_file)[0].artifact
        inspection = self._inspection((AudioStream(4, "aac", None, None, None, None, None, False, None),), artifact_id=other_artifact.id)
        second_extractor = FakeExtractor(version="ffmpeg n1", output_bytes=b"same wav"); second_service = AudioPipelineService(self.repository, self.store, second_extractor)
        second_selection, _ = second_service.select_audio(other_artifact.id, inspection.id); second, reused = second_service.derive_audio(second_selection.id)
        self.assertFalse(reused); self.assertEqual(first.sha256, second.sha256); self.assertEqual(first.managed_key, second.managed_key); self.assertNotEqual(first.id, second.id); self.assertNotEqual(first.audio_selection_id, second.audio_selection_id)

    def test_temporary_audio_file_is_cleaned_after_success(self):
        service, selection = self._single_audio_selection(); tracker = TrackingTemporaryDirectory(self.root)
        with patch("engineering_brain.application.audio_pipeline.tempfile.TemporaryDirectory", return_value=tracker): derived, _ = service.derive_audio(selection.id)
        self.assertIsNotNone(tracker.path); self.assertFalse(tracker.path.exists()); self.assertTrue(self.store.managed_path(derived.managed_key).exists())

    def test_temporary_audio_file_is_cleaned_after_extractor_failure(self):
        extractor = FakeExtractor(fail_after_output=True); service, selection = self._single_audio_selection(extractor); tracker = TrackingTemporaryDirectory(self.root)
        with patch("engineering_brain.application.audio_pipeline.tempfile.TemporaryDirectory", return_value=tracker):
            with self.assertRaises(AudioExtractionError): service.derive_audio(selection.id)
        self.assertIsNotNone(tracker.path); self.assertFalse(tracker.path.exists()); self.assertEqual(self._derived_count(), 0)

    def test_temporary_audio_file_is_cleaned_after_persistence_failure(self):
        service, selection = self._single_audio_selection(); tracker = TrackingTemporaryDirectory(self.root)
        with patch("engineering_brain.application.audio_pipeline.tempfile.TemporaryDirectory", return_value=tracker):
            with patch.object(self.repository, "create_derived_audio_artifact", side_effect=RepositoryError("forced persistence failure")):
                with self.assertRaisesRegex(RepositoryError, "forced persistence failure"): service.derive_audio(selection.id)
        self.assertIsNotNone(tracker.path); self.assertFalse(tracker.path.exists()); self.assertEqual(self._derived_count(), 0)

    @unittest.skipUnless(
        shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None,
        "ffmpeg and ffprobe are required for the real media integration test",
    )
    def test_real_ffmpeg_extracts_explicit_second_audio_stream_to_canonical_wav(self):
        media = self.root / "multi-track.mp4"
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=2", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1", "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000:duration=2", "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0", "-c:v", "mpeg4", "-c:a", "aac", str(media)], check=True, capture_output=True, text=True)
        media_artifact = self.registry.attach_local_artifact(material_id=self.material.id, path=media)[0].artifact
        inspector = FfprobeMediaInspector(); inspection, _ = MediaInspectionService(self.repository, self.store, inspector).inspect_artifact(media_artifact.id)
        self.assertEqual(len(inspection.audio_streams), 2); selected_global_index = inspection.audio_streams[1].index; self.assertIsNotNone(selected_global_index)
        extractor = FfmpegAudioExtractor(); service = AudioPipelineService(self.repository, self.store, extractor)
        selection, _ = service.select_audio(media_artifact.id, inspection.id, stream_index=selected_global_index); derived, _ = service.derive_audio(selection.id)
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(self.store.managed_path(derived.managed_key))], check=True, capture_output=True, text=True)
        payload = json.loads(probe.stdout); streams = payload["streams"]
        self.assertEqual(len(streams), 1); self.assertEqual(streams[0]["codec_type"], "audio"); self.assertEqual(streams[0]["codec_name"], "pcm_s16le"); self.assertEqual(streams[0]["sample_rate"], "16000"); self.assertEqual(streams[0]["channels"], 1)
        duration = float(payload["format"]["duration"]); self.assertGreater(duration, 1.5); self.assertLess(duration, 2.3)
        self.assertTrue(extractor.tool_identity()[1]); self.assertTrue(inspector.tool_identity()[1])
