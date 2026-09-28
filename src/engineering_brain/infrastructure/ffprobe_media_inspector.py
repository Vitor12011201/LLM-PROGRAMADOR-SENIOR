from __future__ import annotations

import json
from pathlib import Path
import subprocess

from engineering_brain.domain.errors import (
    MediaInspectorExecutionError,
    MediaInspectorOutputError,
    MediaInspectorTimeoutError,
    MediaInspectorUnavailableError,
)
from engineering_brain.ports.media_inspection import ProbeExecution


class FfprobeMediaInspector:
    """Subprocess adapter only: execution and JSON validation, never normalization."""

    def __init__(self, executable: str = "ffprobe", timeout_seconds: int = 30) -> None:
        self._executable = executable
        self._timeout_seconds = timeout_seconds
        self._identity: tuple[str, str] | None = None

    def tool_identity(self) -> tuple[str, str]:
        if self._identity is None:
            self._identity = ("ffprobe", self._version())
        return self._identity

    def inspect(self, artifact_path: Path) -> ProbeExecution:
        tool, version = self.tool_identity()
        try:
            result = subprocess.run(
                [self._executable, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(artifact_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise MediaInspectorUnavailableError(f"ffprobe is unavailable: {self._executable}") from exc
        except subprocess.TimeoutExpired as exc:
            raise MediaInspectorTimeoutError(f"ffprobe timed out after {self._timeout_seconds} seconds") from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit code {result.returncode}"
            raise MediaInspectorExecutionError(f"ffprobe failed: {detail}")
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise MediaInspectorOutputError("ffprobe returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise MediaInspectorOutputError("ffprobe JSON root must be an object")
        return ProbeExecution(tool, version, payload)

    def _version(self) -> str:
        try:
            result = subprocess.run(
                [self._executable, "-version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise MediaInspectorUnavailableError(f"ffprobe is unavailable: {self._executable}") from exc
        except subprocess.TimeoutExpired as exc:
            raise MediaInspectorTimeoutError(
                f"ffprobe version check timed out after {self._timeout_seconds} seconds"
            ) from exc
        if result.returncode != 0:
            raise MediaInspectorExecutionError("ffprobe version check failed")
        line = result.stdout.splitlines()[0].strip() if result.stdout else ""
        if not line: raise MediaInspectorOutputError("ffprobe did not report a version")
        return line
