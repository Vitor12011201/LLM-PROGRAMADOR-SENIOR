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
    TranscriptionRun, Transcript, TranscriptSegment, TranscriptionStatus,
)


CURRENT_SCHEMA_VERSION = 4


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
                        elif version == 3:
                            _migrate_phase_three_to_four(connection)
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

    def get_media_inspection(self, inspection_id: str) -> MediaInspection:
        return _inspection(self._one("SELECT * FROM media_inspections WHERE id = ?", (inspection_id,), "media inspection", inspection_id))

    def create_audio_selection(self, selection):
        self._insert("INSERT INTO audio_selections (id, source_artifact_id, media_inspection_id, stream_index, policy, reason, selected_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (selection.id, selection.source_artifact_id, selection.media_inspection_id, selection.stream_index, selection.policy, selection.reason, _datetime(selection.selected_at)), "audio selection")

    def find_audio_selection(self, artifact_id: str, inspection_id: str, stream_index: int):
        rows = self._rows("SELECT * FROM audio_selections WHERE source_artifact_id = ? AND media_inspection_id = ? AND stream_index = ?", (artifact_id, inspection_id, stream_index))
        return _audio_selection(rows[0]) if rows else None

    def get_audio_selection(self, selection_id: str):
        return _audio_selection(self._one("SELECT * FROM audio_selections WHERE id = ?", (selection_id,), "audio selection", selection_id))

    def create_derived_audio_artifact(self, derived):
        self._insert("INSERT INTO derived_audio_artifacts (id, audio_selection_id, managed_key, sha256, byte_size, extractor, extractor_version, derivation_schema_version, config_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (derived.id, derived.audio_selection_id, derived.managed_key, derived.sha256, derived.byte_size, derived.extractor, derived.extractor_version, derived.derivation_schema_version, derived.config_json, _datetime(derived.created_at)), "derived audio artifact")

    def find_derived_audio_artifact(self, selection_id: str, extractor: str, version: str, schema: str, config_json: str):
        rows = self._rows("SELECT * FROM derived_audio_artifacts WHERE audio_selection_id = ? AND extractor = ? AND extractor_version = ? AND derivation_schema_version = ? AND config_json = ?", (selection_id, extractor, version, schema, config_json))
        return _derived_audio(rows[0]) if rows else None

    def get_derived_audio_artifact(self, derived_id: str):
        return _derived_audio(self._one("SELECT * FROM derived_audio_artifacts WHERE id = ?", (derived_id,), "derived audio artifact", derived_id))

    def create_transcription_run(self, run: TranscriptionRun) -> None:
        if run.status is not TranscriptionStatus.RUNNING:
            raise RepositoryError("new transcription runs must start in the running state")
        self._insert("INSERT INTO transcription_runs (id, derived_audio_artifact_id, engine, engine_version, model_id, model_revision, device, compute_type, config_json, requested_language, detected_language, language_probability, started_at, completed_at, status, error_message) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (run.id, run.derived_audio_artifact_id, run.engine, run.engine_version, run.model_id, run.model_revision, run.device, run.compute_type, run.config_json, run.requested_language, run.detected_language, run.language_probability, _datetime(run.started_at), _datetime(run.completed_at) if run.completed_at else None, run.status.value, run.error_message), "transcription run")

    def find_equivalent_completed_transcription(self, derived_id, engine, version, model_id, revision, device, compute, config):
        rows = self._rows("SELECT * FROM transcription_runs WHERE derived_audio_artifact_id=? AND engine=? AND engine_version=? AND model_id=? AND model_revision IS ? AND device IS ? AND compute_type IS ? AND config_json=? AND status='completed' ORDER BY completed_at DESC, id DESC LIMIT 1", (derived_id, engine, version, model_id, revision, device, compute, config))
        return _run(rows[0]) if rows else None

    def get_transcription_run(self, run_id): return _run(self._one("SELECT * FROM transcription_runs WHERE id=?", (run_id,), "transcription run", run_id))
    def get_transcript_for_run(self, run_id): return _transcript(self._one("SELECT * FROM transcripts WHERE transcription_run_id=?", (run_id,), "transcript", run_id))
    def segments_for_transcript(self, transcript_id): return [_segment(row) for row in self._rows("SELECT * FROM transcript_segments WHERE transcript_id=? ORDER BY ordinal", (transcript_id,))]
    def mark_transcription_run_failed(self, run_id, message, completed_at):
        with self._connection() as c:
            result = c.execute("UPDATE transcription_runs SET status='failed', error_message=?, completed_at=? WHERE id=? AND status='running'", (message, _datetime(completed_at), run_id))
            if result.rowcount != 1:
                raise RepositoryError(f"could not mark transcription run failed from running state: {run_id}")
    def complete_transcription_run(self, run_id, transcript: Transcript, segments: tuple[TranscriptSegment, ...], detected, probability, completed_at):
        if transcript.transcription_run_id != run_id:
            raise RepositoryError("transcript does not belong to the transcription run being completed")
        if any(segment.transcript_id != transcript.id for segment in segments):
            raise RepositoryError("all transcript segments must belong to the transcript being persisted")
        try:
            with self._connection() as c:
                c.execute("BEGIN IMMEDIATE")
                result = c.execute("UPDATE transcription_runs SET status='completed', detected_language=?, language_probability=?, completed_at=? WHERE id=? AND status='running'", (detected, probability, _datetime(completed_at), run_id))
                if result.rowcount != 1:
                    raise RepositoryError(f"could not complete transcription run from running state: {run_id}")
                c.execute("INSERT INTO transcripts (id, transcription_run_id, language, full_text, created_at) VALUES (?, ?, ?, ?, ?)", (transcript.id, transcript.transcription_run_id, transcript.language, transcript.full_text, _datetime(transcript.created_at)))
                for item in segments:
                    c.execute("INSERT INTO transcript_segments (id, transcript_id, ordinal, start_seconds, end_seconds, text, avg_logprob, no_speech_prob, compression_ratio, temperature, words_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (item.id, item.transcript_id, item.ordinal, item.start_seconds, item.end_seconds, item.text, item.avg_logprob, item.no_speech_prob, item.compression_ratio, item.temperature, item.words_json))
        except sqlite3.Error as exc: raise RepositoryError(f"could not persist transcription result: {exc}") from exc

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


def _migrate_phase_three_to_four(connection: sqlite3.Connection) -> None:
    for statement in _PHASE_FOUR_STATEMENTS:
        connection.execute(statement)


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
_AUDIO_SELECTIONS = "CREATE TABLE IF NOT EXISTS audio_selections (id TEXT PRIMARY KEY, source_artifact_id TEXT NOT NULL REFERENCES source_artifacts(id), media_inspection_id TEXT NOT NULL REFERENCES media_inspections(id), stream_index INTEGER NOT NULL, policy TEXT NOT NULL, reason TEXT NOT NULL, selected_at TEXT NOT NULL, UNIQUE(source_artifact_id, media_inspection_id, stream_index))"
_DERIVED_AUDIO = "CREATE TABLE IF NOT EXISTS derived_audio_artifacts (id TEXT PRIMARY KEY, audio_selection_id TEXT NOT NULL REFERENCES audio_selections(id), managed_key TEXT NOT NULL, sha256 TEXT NOT NULL, byte_size INTEGER NOT NULL, extractor TEXT NOT NULL, extractor_version TEXT NOT NULL, derivation_schema_version TEXT NOT NULL, config_json TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE(audio_selection_id, extractor, extractor_version, derivation_schema_version, config_json))"
_TRANSCRIPTION_RUNS = "CREATE TABLE IF NOT EXISTS transcription_runs (id TEXT PRIMARY KEY, derived_audio_artifact_id TEXT NOT NULL REFERENCES derived_audio_artifacts(id), engine TEXT NOT NULL, engine_version TEXT NOT NULL, model_id TEXT NOT NULL, model_revision TEXT, device TEXT, compute_type TEXT, config_json TEXT NOT NULL, requested_language TEXT, detected_language TEXT, language_probability TEXT, started_at TEXT NOT NULL, completed_at TEXT, status TEXT NOT NULL, error_message TEXT)"
_TRANSCRIPTS = "CREATE TABLE IF NOT EXISTS transcripts (id TEXT PRIMARY KEY, transcription_run_id TEXT NOT NULL UNIQUE REFERENCES transcription_runs(id), language TEXT, full_text TEXT NOT NULL, created_at TEXT NOT NULL)"
_TRANSCRIPT_SEGMENTS = "CREATE TABLE IF NOT EXISTS transcript_segments (id TEXT PRIMARY KEY, transcript_id TEXT NOT NULL REFERENCES transcripts(id), ordinal INTEGER NOT NULL, start_seconds TEXT NOT NULL, end_seconds TEXT NOT NULL, text TEXT NOT NULL, avg_logprob TEXT, no_speech_prob TEXT, compression_ratio TEXT, temperature TEXT, words_json TEXT, UNIQUE(transcript_id, ordinal))"
_PHASE_FOUR_STATEMENTS = (_AUDIO_SELECTIONS, _DERIVED_AUDIO, _TRANSCRIPTION_RUNS, _TRANSCRIPTS, _TRANSCRIPT_SEGMENTS, "CREATE INDEX IF NOT EXISTS idx_audio_selections_artifact_id ON audio_selections(source_artifact_id)", "CREATE INDEX IF NOT EXISTS idx_derived_audio_selection_id ON derived_audio_artifacts(audio_selection_id)", "CREATE INDEX IF NOT EXISTS idx_transcription_runs_derived_audio_id ON transcription_runs(derived_audio_artifact_id)")
_CURRENT_SCHEMA_STATEMENTS = (*_CURRENT_SCHEMA_STATEMENTS, *_PHASE_FOUR_STATEMENTS)


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


def _audio_selection(row):
    from engineering_brain.domain.models import AudioSelection
    return AudioSelection(row["id"], row["source_artifact_id"], row["media_inspection_id"], row["stream_index"], row["policy"], row["reason"], _parse_datetime(row["selected_at"]))


def _derived_audio(row):
    from engineering_brain.domain.models import DerivedAudioArtifact
    return DerivedAudioArtifact(row["id"], row["audio_selection_id"], row["managed_key"], row["sha256"], row["byte_size"], row["extractor"], row["extractor_version"], row["derivation_schema_version"], row["config_json"], _parse_datetime(row["created_at"]))


def _run(row):
    return TranscriptionRun(row["id"], row["derived_audio_artifact_id"], row["engine"], row["engine_version"], row["model_id"], row["model_revision"], row["device"], row["compute_type"], row["config_json"], row["requested_language"], row["detected_language"], row["language_probability"], _parse_datetime(row["started_at"]), _parse_datetime(row["completed_at"]) if row["completed_at"] else None, TranscriptionStatus(row["status"]), row["error_message"])


def _transcript(row): return Transcript(row["id"], row["transcription_run_id"], row["language"], row["full_text"], _parse_datetime(row["created_at"]))
def _segment(row): return TranscriptSegment(row["id"], row["transcript_id"], row["ordinal"], row["start_seconds"], row["end_seconds"], row["text"], row["avg_logprob"], row["no_speech_prob"], row["compression_ratio"], row["temperature"], row["words_json"])
