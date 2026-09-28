from __future__ import annotations

from pathlib import Path
import subprocess

from engineering_brain.domain.errors import AudioExtractionError


class FfmpegAudioExtractor:
    def __init__(self, executable: str = "ffmpeg", timeout_seconds: int = 120) -> None:
        self._executable = executable
        self._timeout_seconds = timeout_seconds
        self._identity: tuple[str, str] | None = None

    def tool_identity(self) -> tuple[str, str]:
        if self._identity is None:
            self._identity = ("ffmpeg", self._version())
        return self._identity

    def extract(self, source_path: Path, stream_index: int, destination_path: Path) -> None:
        self.tool_identity()
        command = [self._executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(source_path), "-map", f"0:{stream_index}", "-vn", "-sn", "-dn", "-map_metadata", "-1", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination_path)]
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=self._timeout_seconds)
        except FileNotFoundError as exc:
            raise AudioExtractionError(f"ffmpeg is unavailable: {self._executable}") from exc
        except subprocess.TimeoutExpired as exc:
            raise AudioExtractionError(f"ffmpeg extraction timed out after {self._timeout_seconds} seconds") from exc
        if result.returncode != 0:
            raise AudioExtractionError(f"ffmpeg extraction failed: {result.stderr.strip() or result.returncode}")
        if not destination_path.is_file() or destination_path.stat().st_size == 0:
            raise AudioExtractionError("ffmpeg completed without producing audio output")

    def _version(self) -> str:
        try:
            result = subprocess.run([self._executable, "-version"], check=False, capture_output=True, text=True, timeout=self._timeout_seconds)
        except FileNotFoundError as exc:
            raise AudioExtractionError(f"ffmpeg is unavailable: {self._executable}") from exc
        except subprocess.TimeoutExpired as exc:
            raise AudioExtractionError(f"ffmpeg version check timed out after {self._timeout_seconds} seconds") from exc
        first_line = result.stdout.splitlines()[0].strip() if result.stdout.splitlines() else ""
        if result.returncode != 0 or not first_line:
            raise AudioExtractionError("ffmpeg version check failed")
        return first_line
