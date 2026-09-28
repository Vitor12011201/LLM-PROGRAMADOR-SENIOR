from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    database_path: Path
    artifact_dir: Path

    @classmethod
    def from_environment(cls) -> Settings:
        data_dir = Path(os.environ.get("ENGINEERING_BRAIN_DATA_DIR", "data")).expanduser()
        database_path = Path(
            os.environ.get("ENGINEERING_BRAIN_DATABASE_PATH", str(data_dir / "engineering_brain.sqlite3"))
        ).expanduser()
        artifact_dir = Path(
            os.environ.get("ENGINEERING_BRAIN_ARTIFACT_DIR", str(data_dir / "artifacts"))
        ).expanduser()
        return cls(data_dir=data_dir, database_path=database_path, artifact_dir=artifact_dir)
