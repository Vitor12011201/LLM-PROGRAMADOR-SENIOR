from __future__ import annotations

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from engineering_brain.domain.errors import AudioExtractionError
from engineering_brain.infrastructure.ffmpeg_audio_extractor import FfmpegAudioExtractor


def _version_result() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["ffmpeg", "-version"], 0, "ffmpeg version n-test\n", "")


class FfmpegAudioExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mp4"
        self.source.write_bytes(b"source")
        self.destination = self.root / "audio.wav"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_ffmpeg_tool_identity_is_cached(self) -> None:
        with patch("engineering_brain.infrastructure.ffmpeg_audio_extractor.subprocess.run", return_value=_version_result()) as run:
            extractor = FfmpegAudioExtractor()
            self.assertEqual(extractor.tool_identity(), ("ffmpeg", "ffmpeg version n-test"))
            self.assertEqual(extractor.tool_identity(), ("ffmpeg", "ffmpeg version n-test"))
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], ["ffmpeg", "-version"])

    def test_ffmpeg_unavailable_raises_explicit_error(self) -> None:
        with patch("engineering_brain.infrastructure.ffmpeg_audio_extractor.subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(AudioExtractionError, "unavailable"):
                FfmpegAudioExtractor().tool_identity()

    def test_ffmpeg_version_timeout_raises_explicit_error(self) -> None:
        with patch(
            "engineering_brain.infrastructure.ffmpeg_audio_extractor.subprocess.run",
            side_effect=subprocess.TimeoutExpired("ffmpeg", 5),
        ):
            with self.assertRaisesRegex(AudioExtractionError, "version check timed out"):
                FfmpegAudioExtractor(timeout_seconds=5).tool_identity()

    def test_ffmpeg_empty_version_output_is_rejected(self) -> None:
        empty = subprocess.CompletedProcess(["ffmpeg", "-version"], 0, "\n", "")
        with patch("engineering_brain.infrastructure.ffmpeg_audio_extractor.subprocess.run", return_value=empty):
            with self.assertRaisesRegex(AudioExtractionError, "version check failed"):
                FfmpegAudioExtractor().tool_identity()

    def test_ffmpeg_extraction_timeout_raises_explicit_error(self) -> None:
        with patch(
            "engineering_brain.infrastructure.ffmpeg_audio_extractor.subprocess.run",
            side_effect=[_version_result(), subprocess.TimeoutExpired("ffmpeg", 5)],
        ):
            with self.assertRaisesRegex(AudioExtractionError, "extraction timed out"):
                FfmpegAudioExtractor(timeout_seconds=5).extract(self.source, 2, self.destination)
        self.assertFalse(self.destination.exists())

    def test_ffmpeg_nonzero_exit_raises_execution_error(self) -> None:
        failed = subprocess.CompletedProcess(["ffmpeg"], 1, "", "invalid selected stream")
        with patch(
            "engineering_brain.infrastructure.ffmpeg_audio_extractor.subprocess.run",
            side_effect=[_version_result(), failed],
        ):
            with self.assertRaisesRegex(AudioExtractionError, "invalid selected stream"):
                FfmpegAudioExtractor().extract(self.source, 2, self.destination)

    def test_ffmpeg_success_without_output_is_rejected(self) -> None:
        succeeded = subprocess.CompletedProcess(["ffmpeg"], 0, "", "")
        with patch(
            "engineering_brain.infrastructure.ffmpeg_audio_extractor.subprocess.run",
            side_effect=[_version_result(), succeeded],
        ):
            with self.assertRaisesRegex(AudioExtractionError, "without producing audio output"):
                FfmpegAudioExtractor().extract(self.source, 2, self.destination)

    def test_ffmpeg_command_explicitly_maps_selected_stream(self) -> None:
        calls: list[tuple[list[str], dict[str, object]]] = []

        def run(command, **kwargs):
            calls.append((command, kwargs))
            if command[1] == "-version":
                return _version_result()
            Path(command[-1]).write_bytes(b"wav")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch("engineering_brain.infrastructure.ffmpeg_audio_extractor.subprocess.run", side_effect=run):
            FfmpegAudioExtractor().extract(self.source, 2, self.destination)

        command, _ = calls[-1]
        pairs = dict(zip(command, command[1:]))
        self.assertEqual(pairs["-map"], "0:2")
        self.assertEqual(pairs["-loglevel"], "error")
        self.assertEqual(pairs["-map_metadata"], "-1")
        for argument in ("-hide_banner", "-nostdin", "-vn", "-sn", "-dn"):
            self.assertIn(argument, command)
        self.assertEqual(pairs["-ac"], "1")
        self.assertEqual(pairs["-ar"], "16000")
        self.assertEqual(pairs["-c:a"], "pcm_s16le")

    def test_ffmpeg_subprocess_does_not_use_shell(self) -> None:
        calls: list[dict[str, object]] = []

        def run(command, **kwargs):
            calls.append(kwargs)
            if command[1] == "-version":
                return _version_result()
            Path(command[-1]).write_bytes(b"wav")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch("engineering_brain.infrastructure.ffmpeg_audio_extractor.subprocess.run", side_effect=run):
            FfmpegAudioExtractor().extract(self.source, 2, self.destination)

        self.assertTrue(all(kwargs.get("shell") is not True for kwargs in calls))
