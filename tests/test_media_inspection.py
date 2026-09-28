from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from engineering_brain.application.media_inspection import MediaInspectionService
from engineering_brain.application.source_registry import SourceRegistryService
from engineering_brain.domain.errors import (
    ArtifactIntegrityError,
    MediaInspectorExecutionError,
    MediaInspectorOutputError,
    MediaInspectorTimeoutError,
)
from engineering_brain.application.media_normalization import normalize_probe
from engineering_brain.infrastructure.ffprobe_media_inspector import FfprobeMediaInspector
from engineering_brain.infrastructure.local_artifact_store import LocalArtifactStore
from engineering_brain.infrastructure.sqlite_source_registry import SqliteSourceRegistryRepository
from engineering_brain.ports.media_inspection import ProbeExecution


class FakeInspector:
    def __init__(self, payload: dict[str, object], version: str = "ffprobe version fixture") -> None:
        self.payload = payload
        self.version = version
        self.calls = 0

    def tool_identity(self) -> tuple[str, str]:
        return "ffprobe", self.version

    def inspect(self, artifact_path: Path) -> ProbeExecution:
        self.calls += 1
        return ProbeExecution("ffprobe", self.version, self.payload)


class MediaInspectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.repository = SqliteSourceRegistryRepository(self.root / "registry.sqlite3")
        self.repository.initialize()
        self.store = LocalArtifactStore(self.root / "artifacts")
        self.registry = SourceRegistryService(self.repository, self.store)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _artifact_id(self) -> str:
        author = self.registry.register_author("Author")
        source = self.registry.register_source(kind="collection", title="Source", author_id=author.id)
        material = self.registry.register_material(
            source_id=source.id, kind="video", title="Material", origin="file://external"
        )
        external = self.root / "external.bin"
        external.write_bytes(b"managed media candidate")
        artifact, _ = self.registry.attach_local_artifact(material_id=material.id, path=external)
        return artifact.id

    def test_parser_normalizes_audio_video_multiple_tracks_and_subtitles(self) -> None:
        normalized = normalize_probe(_audio_video_payload())
        self.assertEqual(normalized["classification"], "audio_video")
        self.assertEqual(normalized["format_names"], ("matroska", "webm"))
        self.assertEqual(len(normalized["video_streams"]), 1)
        self.assertEqual(normalized["video_streams"][0]["frame_rate"], "30000/1001")
        self.assertEqual(len(normalized["audio_streams"]), 2)
        self.assertEqual(normalized["audio_streams"][1]["language"], "eng")
        self.assertEqual(len(normalized["subtitle_streams"]), 1)

    def test_parser_classifies_video_only_and_audio_only(self) -> None:
        video_only = normalize_probe({"streams": [_video_stream()], "format": {}})
        audio_only = normalize_probe({"streams": [_audio_stream(0)], "format": {}})
        self.assertEqual(video_only["classification"], "video")
        self.assertEqual(audio_only["classification"], "audio")

    def test_parser_handles_missing_optional_metadata(self) -> None:
        normalized = normalize_probe({"streams": [{"codec_type": "audio"}], "format": {}})
        audio = normalized["audio_streams"][0]
        self.assertIsNone(audio["codec"])
        self.assertIsNone(audio["sample_rate"])
        self.assertFalse(audio["is_default"])

    def test_inspection_persists_provenance_and_is_idempotent(self) -> None:
        artifact_id = self._artifact_id()
        inspector = FakeInspector(_audio_video_payload())
        service = MediaInspectionService(self.repository, self.store, inspector)

        inspection, reused = service.inspect_artifact(artifact_id)
        second, reused_second = service.inspect_artifact(artifact_id)
        reloaded = service.show_latest(artifact_id)

        self.assertFalse(reused)
        self.assertTrue(reused_second)
        self.assertEqual(inspector.calls, 1)
        self.assertEqual(inspection.id, second.id)
        self.assertEqual(reloaded.artifact_id, artifact_id)
        self.assertEqual(reloaded.tool, "ffprobe")
        self.assertEqual(reloaded.tool_version, "ffprobe version fixture")
        self.assertEqual(reloaded.schema_version, "media-inspection-v1")
        self.assertIn('"streams"', reloaded.raw_probe_json)

    def test_corrupted_artifact_is_rejected_before_probe(self) -> None:
        artifact_id = self._artifact_id()
        artifact = self.repository.get_artifact(artifact_id)
        path = self.store.managed_path(artifact.managed_key)
        path.chmod(0o644)
        path.write_bytes(b"corrupt")
        inspector = FakeInspector(_audio_video_payload())

        with self.assertRaises(ArtifactIntegrityError):
            MediaInspectionService(self.repository, self.store, inspector).inspect_artifact(artifact_id)
        self.assertEqual(inspector.calls, 0)


class FfprobeAdapterTests(unittest.TestCase):
    def test_invalid_json_is_reported_clearly(self) -> None:
        completed = [
            CompletedProcess(["ffprobe", "-version"], 0, "ffprobe version test\n", ""),
            CompletedProcess(["ffprobe"], 0, "not-json", ""),
        ]
        with patch("engineering_brain.infrastructure.ffprobe_media_inspector.subprocess.run", side_effect=completed):
            with self.assertRaises(MediaInspectorOutputError):
                FfprobeMediaInspector().inspect(Path("artifact"))

    def test_nonzero_exit_is_reported_clearly(self) -> None:
        completed = [
            CompletedProcess(["ffprobe", "-version"], 0, "ffprobe version test\n", ""),
            CompletedProcess(["ffprobe"], 1, "", "bad media"),
        ]
        with patch("engineering_brain.infrastructure.ffprobe_media_inspector.subprocess.run", side_effect=completed):
            with self.assertRaisesRegex(MediaInspectorExecutionError, "bad media"):
                FfprobeMediaInspector().inspect(Path("artifact"))

    def test_timeout_is_reported_clearly(self) -> None:
        import subprocess

        with patch(
            "engineering_brain.infrastructure.ffprobe_media_inspector.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["ffprobe", "-version"], 30),
        ):
            with self.assertRaises(MediaInspectorTimeoutError):
                FfprobeMediaInspector().inspect(Path("artifact"))


def _video_stream() -> dict[str, object]:
    return {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080,
            "avg_frame_rate": "30000/1001", "pix_fmt": "yuv420p", "duration": "12.500",
            "disposition": {"default": 1}, "tags": {"language": "und"}}


def _audio_stream(index: int) -> dict[str, object]:
    return {"index": index, "codec_type": "audio", "codec_name": "aac", "sample_rate": "48000",
            "channels": 2, "channel_layout": "stereo", "duration": "12.500", "bit_rate": "128000",
            "disposition": {"default": index == 1}, "tags": {"language": "por" if index == 1 else "eng"}}


def _audio_video_payload() -> dict[str, object]:
    return {"format": {"format_name": "matroska,webm", "duration": "12.500", "bit_rate": "250000"},
            "streams": [_video_stream(), _audio_stream(1), _audio_stream(2),
                        {"index": 3, "codec_type": "subtitle", "codec_name": "ass",
                         "disposition": {"default": 0}, "tags": {"language": "por"}}]}
