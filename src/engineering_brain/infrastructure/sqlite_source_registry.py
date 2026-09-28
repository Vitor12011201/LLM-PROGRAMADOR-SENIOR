from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Iterator

from engineering_brain.domain.errors import DuplicateError, NotFoundError, RepositoryError
from engineering_brain.domain.models import (
    AudioStream,
    MediaClassification,
    MediaInspection,
    Author,
    Material,
    Metadata,
    Source,
    SourceArtifact,
    SubtitleStream,
    VideoStream,
)


class SqliteSourceRegistryRepository:
    """SQLite adapter; the domain knows neither SQL nor SQLite."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    def initialize(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        schema = """
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS authors (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL, metadata_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sources (
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL, created_at TEXT NOT NULL,
            author_id TEXT REFERENCES authors(id) ON DELETE RESTRICT, origin TEXT, metadata_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS materials (
            id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE RESTRICT,
            kind TEXT NOT NULL, title TEXT NOT NULL, origin TEXT NOT NULL, revision TEXT NOT NULL,
            ingested_at TEXT NOT NULL, status TEXT NOT NULL, metadata_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS source_artifacts (
            id TEXT PRIMARY KEY,
            storage_uri TEXT NOT NULL,
            original_location TEXT NOT NULL,
            managed_key TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            sha256 TEXT NOT NULL UNIQUE,
            byte_size INTEGER NOT NULL,
            observed_at TEXT NOT NULL,
            media_type TEXT
        );
        CREATE TABLE IF NOT EXISTS material_artifacts (
            material_id TEXT NOT NULL REFERENCES materials(id) ON DELETE RESTRICT,
            artifact_id TEXT NOT NULL REFERENCES source_artifacts(id) ON DELETE RESTRICT,
            linked_at TEXT NOT NULL,
            PRIMARY KEY (material_id, artifact_id)
        );
        CREATE TABLE IF NOT EXISTS media_inspections (
            id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL REFERENCES source_artifacts(id) ON DELETE RESTRICT,
            tool TEXT NOT NULL,
            tool_version TEXT NOT NULL,
            schema_version TEXT NOT NULL,
            inspected_at TEXT NOT NULL,
            status TEXT NOT NULL,
            classification TEXT NOT NULL,
            format_names_json TEXT NOT NULL,
            duration_seconds TEXT,
            bit_rate INTEGER,
            stream_count INTEGER NOT NULL,
            video_streams_json TEXT NOT NULL,
            audio_streams_json TEXT NOT NULL,
            subtitle_streams_json TEXT NOT NULL,
            raw_probe_json TEXT NOT NULL,
            UNIQUE (artifact_id, tool, tool_version, schema_version)
        );
        CREATE INDEX IF NOT EXISTS idx_sources_author_id ON sources(author_id);
        CREATE INDEX IF NOT EXISTS idx_materials_source_id ON materials(source_id);
        CREATE INDEX IF NOT EXISTS idx_media_inspections_artifact_id ON media_inspections(artifact_id);
        """
        try:
            with self._connection() as connection:
                connection.executescript(schema)
                self._migrate_artifact_schema(connection)
        except sqlite3.Error as exc:
            raise RepositoryError(f"could not initialize SQLite registry at {self._database_path}: {exc}") from exc

    def create_author(self, author: Author) -> None:
        self._insert(
            "INSERT INTO authors (id, name, created_at, metadata_json) VALUES (?, ?, ?, ?)",
            (author.id, author.name, _serialize_datetime(author.created_at), author.metadata.json_value),
            "author",
        )

    def get_author(self, author_id: str) -> Author:
        row = self._one("SELECT * FROM authors WHERE id = ?", (author_id,), "author", author_id)
        return Author(row["id"], row["name"], _parse_datetime(row["created_at"]), Metadata(row["metadata_json"]))

    def create_source(self, source: Source) -> None:
        self._insert(
            """INSERT INTO sources (id, kind, title, created_at, author_id, origin, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                source.id, source.kind, source.title, _serialize_datetime(source.created_at), source.author_id,
                source.origin, source.metadata.json_value,
            ),
            "source",
        )

    def get_source(self, source_id: str) -> Source:
        row = self._one("SELECT * FROM sources WHERE id = ?", (source_id,), "source", source_id)
        return Source(
            row["id"], row["kind"], row["title"], _parse_datetime(row["created_at"]), row["author_id"],
            row["origin"], Metadata(row["metadata_json"]),
        )

    def create_material(self, material: Material) -> None:
        self._insert(
            """INSERT INTO materials
               (id, source_id, kind, title, origin, revision, ingested_at, status, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                material.id, material.source_id, material.kind, material.title, material.origin, material.revision,
                _serialize_datetime(material.ingested_at), material.status, material.metadata.json_value,
            ),
            "material",
        )

    def get_material(self, material_id: str) -> Material:
        row = self._one("SELECT * FROM materials WHERE id = ?", (material_id,), "material", material_id)
        return _material_from_row(row)

    def list_materials(self) -> list[Material]:
        try:
            with self._connection() as connection:
                rows = connection.execute("SELECT * FROM materials ORDER BY ingested_at, id").fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError(f"could not list materials: {exc}") from exc
        return [_material_from_row(row) for row in rows]

    def get_artifact(self, artifact_id: str) -> SourceArtifact:
        row = self._one("SELECT * FROM source_artifacts WHERE id = ?", (artifact_id,), "artifact", artifact_id)
        return _artifact_from_row(row)

    def find_media_inspection(
        self, artifact_id: str, tool: str, tool_version: str, schema_version: str
    ) -> MediaInspection | None:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    """SELECT * FROM media_inspections
                       WHERE artifact_id = ? AND tool = ? AND tool_version = ? AND schema_version = ?""",
                    (artifact_id, tool, tool_version, schema_version),
                ).fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError(f"could not load media inspection: {exc}") from exc
        return _inspection_from_row(row) if row is not None else None

    def create_media_inspection(self, inspection: MediaInspection) -> None:
        self._insert(
            """INSERT INTO media_inspections
               (id, artifact_id, tool, tool_version, schema_version, inspected_at, status, classification,
                format_names_json, duration_seconds, bit_rate, stream_count, video_streams_json,
                audio_streams_json, subtitle_streams_json, raw_probe_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                inspection.id, inspection.artifact_id, inspection.tool, inspection.tool_version,
                inspection.schema_version, _serialize_datetime(inspection.inspected_at), inspection.status,
                inspection.classification.value, _json_value(inspection.format_names), inspection.duration_seconds,
                inspection.bit_rate, inspection.stream_count, _json_value(inspection.video_streams),
                _json_value(inspection.audio_streams), _json_value(inspection.subtitle_streams), inspection.raw_probe_json,
            ),
            "media inspection",
        )

    def latest_media_inspection(self, artifact_id: str) -> MediaInspection:
        row = self._one(
            "SELECT * FROM media_inspections WHERE artifact_id = ? ORDER BY inspected_at DESC, id DESC LIMIT 1",
            (artifact_id,),
            "media inspection",
            artifact_id,
        )
        return _inspection_from_row(row)

    def attach_artifact(self, material_id: str, artifact: SourceArtifact) -> tuple[SourceArtifact, bool]:
        """Persist one content-addressed artifact, then link it to this material atomically."""
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                existing_row = connection.execute(
                    "SELECT * FROM source_artifacts WHERE sha256 = ?", (artifact.sha256,)
                ).fetchone()
                if existing_row is None:
                    chosen = artifact
                    connection.execute(
                        """INSERT INTO source_artifacts
                           (id, storage_uri, original_location, managed_key, original_filename,
                            sha256, byte_size, observed_at, media_type)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            artifact.id,
                            _managed_uri(artifact.managed_key),
                            artifact.original_location,
                            artifact.managed_key,
                            artifact.original_filename,
                            artifact.sha256,
                            artifact.byte_size,
                            _serialize_datetime(artifact.observed_at),
                            artifact.media_type,
                        ),
                    )
                    reused = False
                else:
                    chosen = _artifact_from_row(existing_row)
                    if chosen.managed_key is None:
                        connection.execute(
                            """UPDATE source_artifacts
                               SET storage_uri = ?, managed_key = ?, original_filename = COALESCE(original_filename, ?)
                               WHERE id = ?""",
                            (_managed_uri(artifact.managed_key), artifact.managed_key, artifact.original_filename, chosen.id),
                        )
                        chosen = SourceArtifact(
                            id=chosen.id,
                            original_location=chosen.original_location,
                            managed_key=artifact.managed_key,
                            sha256=chosen.sha256,
                            byte_size=chosen.byte_size,
                            observed_at=chosen.observed_at,
                            original_filename=chosen.original_filename or artifact.original_filename,
                            media_type=chosen.media_type,
                        )
                    reused = True
                already_linked = connection.execute(
                    "SELECT 1 FROM material_artifacts WHERE material_id = ? AND artifact_id = ?",
                    (material_id, chosen.id),
                ).fetchone()
                if already_linked is not None:
                    raise DuplicateError(
                        f"artifact with SHA-256 {chosen.sha256} is already linked to material {material_id}"
                    )
                connection.execute(
                    "INSERT INTO material_artifacts (material_id, artifact_id, linked_at) VALUES (?, ?, ?)",
                    (material_id, chosen.id, _serialize_datetime(artifact.observed_at)),
                )
                connection.commit()
                return chosen, reused
        except sqlite3.IntegrityError as exc:
            raise RepositoryError(f"could not attach artifact: {exc}") from exc
        except sqlite3.Error as exc:
            raise RepositoryError(f"could not attach artifact: {exc}") from exc

    def artifacts_for_material(self, material_id: str) -> list[SourceArtifact]:
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    """SELECT a.* FROM source_artifacts a
                       JOIN material_artifacts ma ON ma.artifact_id = a.id
                       WHERE ma.material_id = ? ORDER BY ma.linked_at, a.id""",
                    (material_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError(f"could not load material artifacts: {exc}") from exc
        return [_artifact_from_row(row) for row in rows]

    def _insert(self, statement: str, values: tuple[object, ...], record_name: str) -> None:
        try:
            with self._connection() as connection:
                connection.execute(statement, values)
        except sqlite3.IntegrityError as exc:
            if "UNIQUE" in str(exc).upper() or "PRIMARY KEY" in str(exc).upper():
                raise DuplicateError(f"{record_name} already exists") from exc
            raise RepositoryError(f"could not store {record_name}: invalid reference") from exc
        except sqlite3.Error as exc:
            raise RepositoryError(f"could not store {record_name}: {exc}") from exc

    def _one(self, statement: str, values: tuple[object, ...], kind: str, record_id: str) -> sqlite3.Row:
        try:
            with self._connection() as connection:
                row = connection.execute(statement, values).fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError(f"could not load {kind}: {exc}") from exc
        if row is None:
            raise NotFoundError(f"{kind} not found: {record_id}")
        return row

    @staticmethod
    def _migrate_artifact_schema(connection: sqlite3.Connection) -> None:
        """Safely retain Phase 1 rows, which only referenced external files."""
        columns = {row[1] for row in connection.execute("PRAGMA table_info(source_artifacts)")}
        if "original_location" not in columns:
            connection.execute("ALTER TABLE source_artifacts ADD COLUMN original_location TEXT")
            connection.execute("UPDATE source_artifacts SET original_location = storage_uri")
        if "managed_key" not in columns:
            connection.execute("ALTER TABLE source_artifacts ADD COLUMN managed_key TEXT")
        if "original_filename" not in columns:
            connection.execute("ALTER TABLE source_artifacts ADD COLUMN original_filename TEXT")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()


def _serialize_datetime(value: datetime) -> str:
    return value.isoformat()


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _material_from_row(row: sqlite3.Row) -> Material:
    return Material(
        row["id"], row["source_id"], row["kind"], row["title"], row["origin"], row["revision"],
        _parse_datetime(row["ingested_at"]), row["status"], Metadata(row["metadata_json"]),
    )


def _artifact_from_row(row: sqlite3.Row) -> SourceArtifact:
    return SourceArtifact(
        id=row["id"],
        original_location=row["original_location"] or row["storage_uri"],
        managed_key=row["managed_key"],
        sha256=row["sha256"],
        byte_size=row["byte_size"],
        observed_at=_parse_datetime(row["observed_at"]),
        original_filename=row["original_filename"],
        media_type=row["media_type"],
    )


def _managed_uri(managed_key: str | None) -> str:
    if managed_key is None:
        return ""
    return f"artifact://{managed_key}"


def _json_value(value: object) -> str:
    import json
    from dataclasses import asdict

    if isinstance(value, tuple):
        value = [asdict(item) if hasattr(item, "__dataclass_fields__") else item for item in value]
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _inspection_from_row(row: sqlite3.Row) -> MediaInspection:
    import json

    videos = tuple(VideoStream(**item) for item in json.loads(row["video_streams_json"]))
    audios = tuple(AudioStream(**item) for item in json.loads(row["audio_streams_json"]))
    subtitles = tuple(SubtitleStream(**item) for item in json.loads(row["subtitle_streams_json"]))
    return MediaInspection(
        id=row["id"], artifact_id=row["artifact_id"], tool=row["tool"], tool_version=row["tool_version"],
        schema_version=row["schema_version"], inspected_at=_parse_datetime(row["inspected_at"]), status=row["status"],
        classification=MediaClassification(row["classification"]), format_names=tuple(json.loads(row["format_names_json"])),
        duration_seconds=row["duration_seconds"], bit_rate=row["bit_rate"], stream_count=row["stream_count"],
        video_streams=videos, audio_streams=audios, subtitle_streams=subtitles, raw_probe_json=row["raw_probe_json"],
    )
