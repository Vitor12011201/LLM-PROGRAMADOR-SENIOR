from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Mapping

from engineering_brain.domain.errors import (
    MediaInspectorExecutionError,
    MediaInspectorOutputError,
    MediaInspectorTimeoutError,
    MediaInspectorUnavailableError,
)
from engineering_brain.ports.media_inspection import ProbeExecution


class FfprobeMediaInspector:
    """Small subprocess adapter for deterministic FFmpeg metadata inspection."""

    def __init__(self, executable: str = "ffprobe", timeout_seconds: int = 30) -> None:
        self._executable = executable
        self._timeout_seconds = timeout_seconds

    def inspect(self, artifact_path: Path) -> ProbeExecution:
        tool, version = self.tool_identity()
        command = [
            self._executable,
            "-v", "error",
            "-show_format",
            "-show_streams",
            "-of", "json",
            str(artifact_path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise MediaInspectorUnavailableError(f"ffprobe is unavailable: {self._executable}") from exc
        except subprocess.TimeoutExpired as exc:
            raise MediaInspectorTimeoutError(f"ffprobe timed out after {self._timeout_seconds} seconds") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"exit code {completed.returncode}"
            raise MediaInspectorExecutionError(f"ffprobe failed: {detail}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise MediaInspectorOutputError("ffprobe returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise MediaInspectorOutputError("ffprobe JSON root must be an object")
        return ProbeExecution(tool=tool, tool_version=version, payload=payload)

    def tool_identity(self) -> tuple[str, str]:
        return "ffprobe", self._version()

    def _version(self) -> str:
        try:
            completed = subprocess.run(
                [self._executable, "-version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise MediaInspectorUnavailableError(f"ffprobe is unavailable: {self._executable}") from exc
        except subprocess.TimeoutExpired as exc:
            raise MediaInspectorTimeoutError(f"ffprobe version check timed out after {self._timeout_seconds} seconds") from exc
        if completed.returncode != 0:
            raise MediaInspectorExecutionError("ffprobe version check failed")
        first_line = completed.stdout.splitlines()[0].strip() if completed.stdout else ""
        if not first_line:
            raise MediaInspectorOutputError("ffprobe did not report a version")
        return first_line


def normalize_probe(payload: Mapping[str, object]) -> dict[str, object]:
    """Normalize only fields needed by later media phases; retain raw JSON separately."""
    streams = payload.get("streams")
    if not isinstance(streams, list):
        streams = []
    format_data = payload.get("format")
    if not isinstance(format_data, dict):
        format_data = {}

    videos: list[dict[str, object]] = []
    audios: list[dict[str, object]] = []
    subtitles: list[dict[str, object]] = []
    for item in streams:
        if not isinstance(item, dict):
            continue
        stream_type = _text(item.get("codec_type"))
        common = {
            "index": _integer(item.get("index")),
            "codec": _text(item.get("codec_name")),
            "duration_seconds": _duration(item.get("duration")),
            "is_default": _truthy(_mapping(item.get("disposition")).get("default")),
            "language": _text(_mapping(item.get("tags")).get("language")),
        }
        if stream_type == "video":
            videos.append({
                **common,
                "width": _integer(item.get("width")),
                "height": _integer(item.get("height")),
                "frame_rate": _frame_rate(item),
                "pixel_format": _text(item.get("pix_fmt")),
            })
        elif stream_type == "audio":
            audios.append({
                **common,
                "sample_rate": _integer(item.get("sample_rate")),
                "channels": _integer(item.get("channels")),
                "channel_layout": _text(item.get("channel_layout")),
                "bit_rate": _integer(item.get("bit_rate")),
            })
        elif stream_type == "subtitle":
            subtitles.append({key: common[key] for key in ("index", "codec", "is_default", "language")})

    if videos and audios:
        classification = "audio_video"
    elif videos:
        classification = "video"
    elif audios:
        classification = "audio"
    elif streams:
        classification = "other_media"
    else:
        classification = "unsupported"
    format_names = tuple(part.strip() for part in (_text(format_data.get("format_name")) or "").split(",") if part.strip())
    return {
        "classification": classification,
        "format_names": format_names,
        "duration_seconds": _duration(format_data.get("duration")),
        "bit_rate": _integer(format_data.get("bit_rate")),
        "stream_count": len(streams),
        "video_streams": tuple(videos),
        "audio_streams": tuple(audios),
        "subtitle_streams": tuple(subtitles),
    }


def canonical_probe_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _integer(value: object) -> int | None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _duration(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        from decimal import Decimal, InvalidOperation
        parsed = Decimal(value)
    except (InvalidOperation, ValueError):
        return None
    return value if parsed >= 0 else None


def _truthy(value: object) -> bool:
    return value in (1, "1", True)


def _frame_rate(stream: Mapping[str, object]) -> str | None:
    for field in ("avg_frame_rate", "r_frame_rate"):
        value = _text(stream.get(field))
        if value and value != "0/0":
            return value
    return None
