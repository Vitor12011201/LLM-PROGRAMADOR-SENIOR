from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import UTC, datetime
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from engineering_brain.cli import main
from engineering_brain.domain.errors import AudioExtractionError
from engineering_brain.domain.models import AudioStream, MediaClassification, MediaInspection
from engineering_brain.infrastructure.sqlite_source_registry import SqliteSourceRegistryRepository


class CliExtractor:
    def __init__(self, *, failure: bool = False) -> None:
        self.failure = failure

    def tool_identity(self) -> tuple[str, str]:
        return "ffmpeg", "cli-test"

    def extract(self, source_path: Path, stream_index: int, destination_path: Path) -> None:
        if self.failure:
            raise AudioExtractionError("forced ffmpeg failure")
        destination_path.write_bytes(b"CLI canonical WAV")


class CommandLineFlowTests(unittest.TestCase):
    def test_cli_registers_and_shows_a_material(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            environment = {"ENGINEERING_BRAIN_DATA_DIR": str(root / "data")}
            with patch.dict(os.environ, environment, clear=False):
                author = self._run("author", "add", "--name", "CLI author")
                source = self._run(
                    "source", "add", "--kind", "collection", "--title", "CLI source", "--author-id", author["id"]
                )
                material = self._run(
                    "material", "add", "--source-id", source["id"], "--kind", "article", "--title", "CLI material",
                    "--origin", "https://example.test/cli",
                )
                artifact_path = root / "evidence.txt"
                artifact_path.write_text("CLI evidence", encoding="utf-8")
                artifact = self._run(
                    "material", "attach-artifact", "--material-id", material["id"], "--path", str(artifact_path)
                )
                record = self._run("material", "show", "--id", material["id"])
                verification = self._run("artifact", "verify", "--id", artifact["artifact"]["id"])

            self.assertFalse(artifact["reused_existing_artifact"])
            self.assertEqual(artifact["observation"]["original_location"], artifact_path.absolute().as_uri())
            self.assertIsNotNone(artifact["artifact"]["managed_artifact"])
            self.assertEqual(record["author"]["id"], author["id"])
            self.assertEqual(record["source"]["id"], source["id"])
            self.assertEqual(record["material"]["id"], material["id"])
            self.assertEqual(len(record["artifacts"]), 1)
            self.assertTrue(verification["is_valid"])

    def test_cli_audio_select_dispatches_selection(self) -> None:
        with self._audio_workspace((3,)) as (_, artifact_id, inspection_id):
            selection = self._run(
                "audio", "select", "--artifact-id", artifact_id, "--inspection-id", inspection_id
            )

        self.assertEqual(selection["source_artifact_id"], artifact_id)
        self.assertEqual(selection["media_inspection_id"], inspection_id)
        self.assertEqual(selection["stream_index"], 3)
        self.assertEqual(selection["policy"], "ONLY_AUDIO_STREAM")
        self.assertFalse(selection["reused_existing_selection"])

    def test_cli_audio_selection_show_returns_stable_json(self) -> None:
        with self._audio_workspace((1,)) as (_, artifact_id, inspection_id):
            selected = self._run(
                "audio", "select", "--artifact-id", artifact_id, "--inspection-id", inspection_id
            )
            shown = self._run("audio", "selection-show", "--id", selected["id"])

        self.assertEqual(shown, {key: value for key, value in selected.items() if key != "reused_existing_selection"})

    def test_cli_audio_derive_dispatches_derivation(self) -> None:
        with self._audio_workspace((1,)) as (_, artifact_id, inspection_id):
            selected = self._run(
                "audio", "select", "--artifact-id", artifact_id, "--inspection-id", inspection_id
            )
            with patch("engineering_brain.cli.FfmpegAudioExtractor", return_value=CliExtractor()):
                derived = self._run("audio", "derive", "--selection-id", selected["id"])

        self.assertEqual(derived["audio_selection_id"], selected["id"])
        self.assertEqual(derived["extractor"], "ffmpeg")
        self.assertEqual(derived["config"], {
            "channels": 1, "codec": "pcm_s16le", "container": "wav", "sample_rate": 16000,
        })
        self.assertFalse(derived["reused_existing_derivation"])

    def test_cli_audio_derived_show_returns_stable_json(self) -> None:
        with self._audio_workspace((1,)) as (_, artifact_id, inspection_id):
            selected = self._run(
                "audio", "select", "--artifact-id", artifact_id, "--inspection-id", inspection_id
            )
            with patch("engineering_brain.cli.FfmpegAudioExtractor", return_value=CliExtractor()):
                derived = self._run("audio", "derive", "--selection-id", selected["id"])
            shown = self._run("audio", "derived-show", "--id", derived["id"])

        self.assertEqual(shown, {key: value for key, value in derived.items() if key != "reused_existing_derivation"})

    def test_cli_audio_pipeline_end_to_end(self) -> None:
        with self._audio_workspace((1,)) as (_, artifact_id, inspection_id):
            selected = self._run(
                "audio", "select", "--artifact-id", artifact_id, "--inspection-id", inspection_id
            )
            with patch("engineering_brain.cli.FfmpegAudioExtractor", return_value=CliExtractor()):
                derived = self._run("audio", "derive", "--selection-id", selected["id"])
            shown = self._run("audio", "derived-show", "--id", derived["id"])

        self.assertEqual(shown["id"], derived["id"])
        self.assertEqual(shown["audio_selection_id"], selected["id"])
        self.assertTrue(shown["managed_artifact"].startswith("sha256/"))

    def test_cli_audio_errors_are_reported_clearly(self) -> None:
        with self._audio_workspace((1, 2)) as (_, artifact_id, inspection_id):
            ambiguous = self._error_run(
                "audio", "select", "--artifact-id", artifact_id, "--inspection-id", inspection_id
            )
            invalid = self._error_run(
                "audio", "select", "--artifact-id", artifact_id, "--inspection-id", inspection_id,
                "--stream-index", "99",
            )
            selected = self._run(
                "audio", "select", "--artifact-id", artifact_id, "--inspection-id", inspection_id,
                "--stream-index", "2",
            )
            with patch("engineering_brain.cli.FfmpegAudioExtractor", return_value=CliExtractor(failure=True)):
                extraction = self._error_run("audio", "derive", "--selection-id", selected["id"])
            missing = self._error_run("audio", "derived-show", "--id", "missing-derived")

        self.assertIn("multiple audio streams", ambiguous)
        self.assertIn("not an audio stream", invalid)
        self.assertIn("forced ffmpeg failure", extraction)
        self.assertIn("derived audio artifact not found", missing)

    def test_cli_audio_inspection_artifact_mismatch_is_reported_without_traceback(self) -> None:
        with self._audio_workspace((1,)) as (root, artifact_id, _):
            repository = SqliteSourceRegistryRepository(root / "data" / "engineering_brain.sqlite3")
            material_id = repository._rows("SELECT id FROM materials", ())[0]["id"]
            other_path = root / "other-source.bin"
            other_path.write_bytes(b"other source evidence")
            other_attached = self._run(
                "material", "attach-artifact", "--material-id", material_id, "--path", str(other_path)
            )
            other_artifact_id = other_attached["artifact"]["id"]
            repository.create_media_inspection(
                MediaInspection(
                    "other-inspection", other_artifact_id, "fixture", "1", "media-inspection-v1",
                    datetime.now(UTC), "completed", MediaClassification.AUDIO, (), None, None, 1,
                    (), (AudioStream(1, "aac", 48000, 2, "stereo", None, None, False, None),), (), "{}",
                )
            )

            error = self._error_run(
                "audio", "select", "--artifact-id", artifact_id,
                "--inspection-id", "other-inspection",
            )

        self.assertIn("does not belong", error)
        self.assertNotIn("Traceback", error)

    @contextmanager
    def _audio_workspace(self, stream_indexes: tuple[int, ...]):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            environment = {"ENGINEERING_BRAIN_DATA_DIR": str(root / "data")}
            with patch.dict(os.environ, environment, clear=False):
                author = self._run("author", "add", "--name", "CLI audio author")
                source = self._run(
                    "source", "add", "--kind", "video", "--title", "CLI audio source",
                    "--author-id", author["id"],
                )
                material = self._run(
                    "material", "add", "--source-id", source["id"], "--kind", "video",
                    "--title", "CLI audio material", "--origin", "file://cli/audio",
                )
                source_path = root / "source.bin"
                source_path.write_bytes(b"CLI source evidence")
                attached = self._run(
                    "material", "attach-artifact", "--material-id", material["id"], "--path", str(source_path)
                )
                artifact_id = attached["artifact"]["id"]
                repository = SqliteSourceRegistryRepository(root / "data" / "engineering_brain.sqlite3")
                streams = tuple(
                    AudioStream(index, "aac", 48000, 2, "stereo", None, None, False, None)
                    for index in stream_indexes
                )
                inspection = MediaInspection(
                    "cli-inspection", artifact_id, "fixture", "1", "media-inspection-v1",
                    datetime.now(UTC), "completed", MediaClassification.AUDIO, (), None, None,
                    len(streams), (), streams, (), "{}",
                )
                repository.create_media_inspection(inspection)
                yield root, artifact_id, inspection.id

    def _run(self, *arguments: str) -> dict[str, object]:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(arguments)
        self.assertEqual(exit_code, 0)
        return json.loads(output.getvalue())

    def _error_run(self, *arguments: str) -> str:
        error = io.StringIO()
        with redirect_stderr(error):
            with self.assertRaises(SystemExit) as captured:
                main(arguments)
        self.assertEqual(captured.exception.code, 2)
        return error.getvalue()
