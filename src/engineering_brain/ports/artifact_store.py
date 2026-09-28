from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    original_location: str
    original_filename: str
    managed_key: str
    sha256: str
    byte_size: int
    reused_existing_blob: bool


@dataclass(frozen=True, slots=True)
class StoreVerification:
    exists: bool
    matches_expected_hash: bool
    actual_sha256: str | None
    actual_byte_size: int | None


class ArtifactStore(Protocol):
    """Storage boundary for preserved source bytes."""

    def ingest(self, source_path: Path) -> StoredArtifact: ...

    def verify(self, managed_key: str, expected_sha256: str) -> StoreVerification: ...

    def path_for_read(self, managed_key: str) -> Path: ...
