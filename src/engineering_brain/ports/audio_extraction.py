from __future__ import annotations

from pathlib import Path
from typing import Protocol


class AudioExtractor(Protocol):
    def tool_identity(self) -> tuple[str, str]: ...

    def extract(self, source_path: Path, stream_index: int, destination_path: Path) -> None: ...
