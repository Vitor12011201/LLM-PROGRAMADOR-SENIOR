from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Iterator

from engineering_brain.domain.errors import DuplicateError, NotFoundError, RegistryError, RepositoryError
from engineering_brain.domain.models import (
    ArtifactObservation,
    ArtifactObservationRecord,
    AudioStream,
    Author,
    MediaClassification,
    MediaInspection,
    Material,
    Metadata,
    Source,
    SourceArtifact,
    SubtitleStream,
    VideoStream,
)


CURRENT_SCHEMA_VERSION = 3


class SqliteSourceRegistryRepository:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    def initialize(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._connection() as connection:
                declared_version = connection.execute("PRAGMA user_version").fetchone()[0]
                if declared_version > CURRENT_SCHEMA_VERSION:
                    raise RepositoryError(
                        f"SQLite schema version {declared_version} is newer than supported version "
                        f"{CURRENT_SCHEMA_VERSION}"
                    )
                version = declared_version or _infer_schema_version(connection)
                while version < CURRENT_SCHEMA_VERSION:
                    next_version = CURRENT_SCHEMA_VERSION if version == 0 else version + 1
                    connection.execute("BEGIN IMMEDIATE")
                    try:
                        if version == 0:
                            _create_current_schema(connection)
                        elif version == 1:
                            _migrate_phase_one_to_two(connection)
                        elif version == 2:
                            _migrate_phase_two_to_three(connection)
                        else:
                            raise RepositoryError(f"unsupported SQLite schema version: {version}")
                        connection.execute(f"PRAGMA user_version = {next_version}")
                        connection.commit()
                    except BaseException:
                        connection.rollback()
                        raise
                    version = next_version
        except RepositoryError:
            raise
        except sqlite3.Error as exc:
            raise RepositoryError(
                f"could not initialize SQLite registry at {self._database_path}: {exc}"
            ) from exc

    def create_author(self, author: Author) -> None:
        self._insert(
            "INSERT INTO authors (id, name, created_at, metadata_json) VALUES (?, ?, ?, ?)",
            (author.id, author.name, _datetime(author.created_at), author.metadata.json_value),
            "author",
        )

    def get_author(self, author_id: str) -> Author:
        row = self._one("SELECT * FROM authors WHERE id = ?", (author_id,), "author", author_id)
        return Author(
            row["id"], row["name"], _parse_datetime(row["created_at"]), Metadata(row["metadata_json"])
        )

    def create_source(self, source: Source) -> None:
        self._insert(
            "INSERT INTO sources (id, kind, title, created_at, author_id, origin, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                source.id,
                source.kind,
                source.title,
                _datetime(source.created_at),
                source.author_id,
                source.origin,
                source.metadata.json_value,
            ),
            "source",
        )

    def get_source(self, source_id: str) -> Source:
        row = self._one("SELECT * FROM sources WHERE id = ?", (source_id,), "source", source_id)
        return Source(
            row["id"],
            row["kind"],
            row["title"],
            _parse_datetime(row["created_at"]),
            row["author_id"],
            row["origin"],
            Metadata(row["metadata_json"]),
        )

    def create_material(self, material: Material) -> None:
        self._insert(
            "INSERT INTO materials "
            "(id, source_id, kind, title, origin, revision, ingested_at, status, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                material.id,
                material.source_id,
                material.kind,
                material.title,
                material.origin,
                material.revision,
                _datetime(material.ingested_at),
                material.status,
                material.metadata.json_value,
            ),
            "material",
        )

    def get_material(self, material_id: str) -> Material:
        return _material(
            self._one("SELECT * FROM materials WHERE id = ?", (material_id,), "material", material_id)
        )

    def list_materials(self) -> list[Material]:
        return [
            _material(row)
            for row in self._rows("SELECT * FROM materials ORDER BY ingested_at, id", ())
        ]

    def get_artifact(self, artifact_id: str) -> SourceArtifact:
        return _artifact(
            self._one("SELECT * FROM source_artifacts WHERE id = ?", (artifact_id,), "artifact", artifact_id)
        )

    def attach_artifact(
        self, artifact: SourceArtifact, observation: ArtifactObservation
    ) -> tuple[ArtifactObservationRecord, bool]:
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM source_artifacts WHERE sha256 = ?", (artifact.sha256,)
                ).fetchone()
                if row is None:
                    selected, reused = artifact, False
                    connection.execute(
                        "INSERT INTO source_artifacts "
                        "(id, managed_key, sha256, byte_size, preserved_at) VALUES (?, ?, ?, ?, ?)",
                        (
                            artifact.id,
                            artifact.managed_key,
                            artifact.sha256,
                            artifact.byte_size,
                            _datetime(artifact.preserved_at),
                        ),
                    )
                else:
                    selected, reused = _artifact(row), True
                    if selected.managed_key is None and artifact.managed_key is not None:
                        connection.execute(
                            "UPDATE source_artifacts SET managed_key = ? WHERE id = ?",
                            (artifact.managed_key, selected.id),
                        )
                        selected = SourceArtifact(
                            selected.id,
                            artifact.managed_key,
                            selected.sha256,
                            selected.byte_size,
                            selected.preserved_at,
                        )

                row = connection.execute(
                    "SELECT * FROM artifact_observations "
                    "WHERE material_id = ? AND artifact_id = ? AND original_location = ?",
                    (observation.material_id, selected.id, observation.original_location),
                ).fetchone()
                if row is not None:
                    connection.commit()
                    return ArtifactObservationRecord(_observation(row), selected), True

                persisted = ArtifactObservation(
                    observation.id,
                    observation.material_id,
                    selected.id,
                    observation.original_location,
                    observation.original_filename,
                    observation.observed_at,
                    observation.media_type,
                )
                connection.execute(
                    "INSERT INTO artifact_observations "
                    "(id, material_id, artifact_id, original_location, original_filename, media_type, observed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        persisted.id,
                        persisted.material_id,
                        persisted.artifact_id,
                        persisted.original_location,
                        persisted.original_filename,
                        persisted.media_type,
                        _datetime(persisted.observed_at),
                    ),
                )
                connection.commit()
                return ArtifactObservationRecord(persisted, selected), reused
        except sqlite3.IntegrityError as exc:
            raise _integrity_error("attach artifact", exc) from exc
        except sqlite3.Error as exc:
            raise RepositoryError(f"could not attach artifact: {exc}") from exc

    def observations_for_material(self, material_id: str) -> list[ArtifactObservationRecord]:
        rows = self._rows(
            "SELECT o.*, a.managed_key, a.sha256, a.byte_size, a.preserved_at "
            "FROM artifact_observations o JOIN source_artifacts a ON a.id = o.artifact_id "
            "WHERE o.material_id = ? ORDER BY o.observed_at, o.id",
            (material_id,),
        )
        return [
            ArtifactObservationRecord(
                _observation(row),
                SourceArtifact(
                    row["artifact_id"],
                    row["managed_key"],
                    row["sha256"],
                    row["byte_size"],
                    _parse_datetime(row["preserved_at"]),
                ),
            )
            for row in rows
        ]

    def find_media_inspection(
        self, artifact_id: str, tool: str, version: str, schema: str
    ) -> MediaInspection | None:
        rows = self._rows(
            "SELECT * FROM media_inspections "
            "WHERE artifact_id = ? AND tool = ? AND tool_version = ? AND schema_version = ?",
            (artifact_id, tool, version, schema),
        )
        return _inspection(rows[0]) if rows else None

    def create_media_inspection(self, inspection: MediaInspection) -> None:
        self._insert(
            "INSERT INTO media_inspections "
            "(id, artifact_id, tool, tool_version, schema_version, inspected_at, status, classification, "
            "format_names_json, duration_seconds, bit_rate, stream_count, video_streams_json, "
            "audio_streams_json, subtitle_streams_json, raw_probe_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                inspection.id,
                inspection.artifact_id,
                inspection.tool,
                inspection.tool_version,
                inspection.schema_version,
                _datetime(inspection.inspected_at),
                inspection.status,
                inspection.classification.value,
                _json(inspection.format_names),
                inspection.duration_seconds,
                inspection.bit_rate,
                inspection.stream_count,
                _json(inspection.video_streams),
                _json(inspection.audio_streams),
                _json(inspection.subtitle_streams),
                inspection.raw_probe_json,
            ),
            "media inspection",
        )

    def latest_media_inspection(self, artifact_id: str) -> MediaInspection:
        return _inspection(
            self._one(
                "SELECT * FROM media_inspections WHERE artifact_id = ? "
                "ORDER BY inspected_at DESC, id DESC LIMIT 1",
                (artifact_id,),
                "media inspection",
                artifact_id,
            )
        )

    def _insert(self, query: str, values: tuple[object, ...], kind: str) -> None:
        try:
            with self._connection() as connection:
                connection.execute(query, values)
        except sqlite3.IntegrityError as exc:
            raise _integrity_error(f"store {kind}", exc) from exc
        except sqlite3.Error as exc:
            raise RepositoryError(f"could not store {kind}: {exc}") from exc

    def _one(
        self, query: str, values: tuple[object, ...], kind: str, identifier: str
    ) -> sqlite3.Row:
        rows = self._rows(query, values)
        if not rows:
            raise NotFoundError(f"{kind} not found: {identifier}")
        return rows[0]

    def _rows(self, query: str, values: tuple[object, ...]) -> list[sqlite3.Row]:
        try:
            with self._connection() as connection:
                return connection.execute(query, values).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError(f"could not load records: {exc}") from exc

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


def _integrity_error(operation: str, exc: sqlite3.IntegrityError) -> RegistryError:
    message = str(exc).upper()
    if "UNIQUE" in message or "PRIMARY KEY" in message:
        return DuplicateError(f"could not {operation}: duplicate record")
    return RepositoryError(f"could not {operation}: integrity constraint failed: {exc}")


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone() is not None


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def _infer_schema_version(connection: sqlite3.Connection) -> int:
    if not _table_exists(connection, "source_artifacts"):
        return 0
    return 2 if "managed_key" in _columns(connection, "source_artifacts") else 1


def _create_current_schema(connection: sqlite3.Connection) -> None:
    for statement in _CURRENT_SCHEMA_STATEMENTS:
        connection.execute(statement)


def _migrate_phase_one_to_two(connection: sqlite3.Connection) -> None:
    columns = _columns(connection, "source_artifacts")
    for name in ("original_location", "managed_key", "original_filename"):
        if name not in columns:
            connection.execute(f"ALTER TABLE source_artifacts ADD COLUMN {name} TEXT")
    if "media_type" not in _columns(connection, "source_artifacts"):
        connection.execute("ALTER TABLE source_artifacts ADD COLUMN media_type TEXT")
    connection.execute(
        "UPDATE source_artifacts SET original_location = COALESCE(original_location, storage_uri)"
    )
    _create_phase_two_schema(connection)


def _migrate_phase_two_to_three(connection: sqlite3.Connection) -> None:
    if "preserved_at" in _columns(connection, "source_artifacts"):
        _create_current_schema(connection)
        return

    connection.execute("ALTER TABLE source_artifacts RENAME TO source_artifacts_legacy")
    _create_v3_artifact_schema(connection)
    connection.execute(
        "INSERT INTO source_artifacts (id, managed_key, sha256, byte_size, preserved_at) "
        "SELECT id, managed_key, sha256, byte_size, observed_at FROM source_artifacts_legacy"
    )

    if _table_exists(connection, "material_artifacts"):
        connection.execute("ALTER TABLE material_artifacts RENAME TO material_artifacts_legacy")
        timestamp = "ma.linked_at" if "linked_at" in _columns(connection, "material_artifacts_legacy") else "a.observed_at"
        connection.execute(
            "INSERT INTO artifact_observations "
            "(id, material_id, artifact_id, original_location, original_filename, media_type, observed_at) "
            "SELECT printf('legacy:%d:%s:%d:%s', length(ma.material_id), hex(ma.material_id), "
            "length(ma.artifact_id), hex(ma.artifact_id)), ma.material_id, ma.artifact_id, "
            "COALESCE(a.original_location, a.storage_uri), a.original_filename, a.media_type, "
            f"{timestamp} FROM material_artifacts_legacy ma "
            "JOIN source_artifacts_legacy a ON a.id = ma.artifact_id"
        )

    if _table_exists(connection, "media_inspections"):
        connection.execute("DROP INDEX IF EXISTS idx_media_inspections_artifact_id")
        connection.execute("ALTER TABLE media_inspections RENAME TO media_inspections_legacy")
        _create_v3_media_inspections_schema(connection)
        connection.execute(
            "INSERT INTO media_inspections "
            "(id, artifact_id, tool, tool_version, schema_version, inspected_at, status, classification, "
            "format_names_json, duration_seconds, bit_rate, stream_count, video_streams_json, "
            "audio_streams_json, subtitle_streams_json, raw_probe_json) "
            "SELECT id, artifact_id, tool, tool_version, schema_version, inspected_at, status, classification, "
            "format_names_json, duration_seconds, bit_rate, stream_count, video_streams_json, "
            "audio_streams_json, subtitle_streams_json, raw_probe_json FROM media_inspections_legacy"
        )
    else:
        _create_v3_media_inspections_schema(connection)
    _create_current_indexes(connection)


def _create_phase_two_schema(connection: sqlite3.Connection) -> None:
    for statement in _PHASE_TWO_STATEMENTS:
        connection.execute(statement)


def _create_v3_artifact_schema(connection: sqlite3.Connection) -> None:
    connection.execute(_SOURCE_ARTIFACTS_V3)
    connection.execute(_ARTIFACT_OBSERVATIONS_V3)


def _create_v3_media_inspections_schema(connection: sqlite3.Connection) -> None:
    connection.execute(_MEDIA_INSPECTIONS_V3)
    connection.execute(_MEDIA_INSPECTIONS_INDEX)


def _create_current_indexes(connection: sqlite3.Connection) -> None:
    for statement in _CURRENT_INDEX_STATEMENTS:
        connection.execute(statement)


_AUTHORS = "CREATE TABLE IF NOT EXISTS authors (id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL, metadata_json TEXT NOT NULL)"
_SOURCES = "CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL, created_at TEXT NOT NULL, author_id TEXT REFERENCES authors(id), origin TEXT, metadata_json TEXT NOT NULL)"
_MATERIALS = "CREATE TABLE IF NOT EXISTS materials (id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id), kind TEXT NOT NULL, title TEXT NOT NULL, origin TEXT NOT NULL, revision TEXT NOT NULL, ingested_at TEXT NOT NULL, status TEXT NOT NULL, metadata_json TEXT NOT NULL)"
_SOURCE_ARTIFACTS_V3 = "CREATE TABLE IF NOT EXISTS source_artifacts (id TEXT PRIMARY KEY, managed_key TEXT, sha256 TEXT NOT NULL UNIQUE, byte_size INTEGER NOT NULL, preserved_at TEXT NOT NULL)"
_ARTIFACT_OBSERVATIONS_V3 = "CREATE TABLE IF NOT EXISTS artifact_observations (id TEXT PRIMARY KEY, material_id TEXT NOT NULL REFERENCES materials(id), artifact_id TEXT NOT NULL REFERENCES source_artifacts(id), original_location TEXT NOT NULL, original_filename TEXT, media_type TEXT, observed_at TEXT NOT NULL, UNIQUE(material_id, artifact_id, original_location))"
_MEDIA_INSPECTIONS_V3 = "CREATE TABLE IF NOT EXISTS media_inspections (id TEXT PRIMARY KEY, artifact_id TEXT NOT NULL REFERENCES source_artifacts(id), tool TEXT NOT NULL, tool_version TEXT NOT NULL, schema_version TEXT NOT NULL, inspected_at TEXT NOT NULL, status TEXT NOT NULL, classification TEXT NOT NULL, format_names_json TEXT NOT NULL, duration_seconds TEXT, bit_rate INTEGER, stream_count INTEGER NOT NULL, video_streams_json TEXT NOT NULL, audio_streams_json TEXT NOT NULL, subtitle_streams_json TEXT NOT NULL, raw_probe_json TEXT NOT NULL, UNIQUE(artifact_id, tool, tool_version, schema_version))"
_MEDIA_INSPECTIONS_INDEX = "CREATE INDEX IF NOT EXISTS idx_media_inspections_artifact_id ON media_inspections(artifact_id)"
_CURRENT_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS idx_sources_author_id ON sources(author_id)",
    "CREATE INDEX IF NOT EXISTS idx_materials_source_id ON materials(source_id)",
    "CREATE INDEX IF NOT EXISTS idx_artifact_observations_material_id ON artifact_observations(material_id)",
    _MEDIA_INSPECTIONS_INDEX,
)
_CURRENT_SCHEMA_STATEMENTS = (_AUTHORS, _SOURCES, _MATERIALS, _SOURCE_ARTIFACTS_V3, _ARTIFACT_OBSERVATIONS_V3, _MEDIA_INSPECTIONS_V3, *_CURRENT_INDEX_STATEMENTS)
_PHASE_TWO_STATEMENTS = (_AUTHORS, _SOURCES, _MATERIALS, _MEDIA_INSPECTIONS_V3, "CREATE INDEX IF NOT EXISTS idx_sources_author_id ON sources(author_id)", "CREATE INDEX IF NOT EXISTS idx_materials_source_id ON materials(source_id)", _MEDIA_INSPECTIONS_INDEX)


def _datetime(value: datetime) -> str:
    return value.isoformat()


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _json(value: object) -> str:
    if isinstance(value, tuple):
        value = [asdict(item) if hasattr(item, "__dataclass_fields__") else item for item in value]
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _material(row: sqlite3.Row) -> Material:
    return Material(row["id"], row["source_id"], row["kind"], row["title"], row["origin"], row["revision"], _parse_datetime(row["ingested_at"]), row["status"], Metadata(row["metadata_json"]))


def _artifact(row: sqlite3.Row) -> SourceArtifact:
    return SourceArtifact(row["id"], row["managed_key"], row["sha256"], row["byte_size"], _parse_datetime(row["preserved_at"]))


def _observation(row: sqlite3.Row) -> ArtifactObservation:
    return ArtifactObservation(row["id"], row["material_id"], row["artifact_id"], row["original_location"], row["original_filename"], _parse_datetime(row["observed_at"]), row["media_type"])


def _inspection(row: sqlite3.Row) -> MediaInspection:
    return MediaInspection(row["id"], row["artifact_id"], row["tool"], row["tool_version"], row["schema_version"], _parse_datetime(row["inspected_at"]), row["status"], MediaClassification(row["classification"]), tuple(json.loads(row["format_names_json"])), row["duration_seconds"], row["bit_rate"], row["stream_count"], tuple(VideoStream(**item) for item in json.loads(row["video_streams_json"])), tuple(AudioStream(**item) for item in json.loads(row["audio_streams_json"])), tuple(SubtitleStream(**item) for item in json.loads(row["subtitle_streams_json"])), row["raw_probe_json"])
