from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from engineering_brain.application.source_registry import SourceRegistryService
from engineering_brain.application.media_inspection import MediaInspectionService
from engineering_brain.config import Settings
from engineering_brain.domain.errors import RegistryError, ValidationError
from engineering_brain.domain.models import (
    ArtifactVerification, AudioStream, Author, MediaInspection, Material, MaterialRecord, Source, SourceArtifact,
    SubtitleStream, VideoStream,
)
from engineering_brain.infrastructure.ffprobe_media_inspector import FfprobeMediaInspector
from engineering_brain.infrastructure.local_artifact_store import LocalArtifactStore
from engineering_brain.infrastructure.sqlite_source_registry import SqliteSourceRegistryRepository


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    settings = Settings.from_environment()
    repository = SqliteSourceRegistryRepository(settings.database_path)
    try:
        repository.initialize()
        artifact_store = LocalArtifactStore(settings.artifact_dir)
        service = SourceRegistryService(repository, artifact_store)
        media_service = MediaInspectionService(repository, artifact_store, FfprobeMediaInspector())
        result = _dispatch(arguments, service, media_service)
    except RegistryError as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="engineering-brain", description="Engineering Brain Source Registry")
    top = parser.add_subparsers(dest="entity", required=True)

    author = top.add_parser("author")
    author_commands = author.add_subparsers(dest="action", required=True)
    author_add = author_commands.add_parser("add")
    author_add.add_argument("--name", required=True)
    author_add.add_argument("--metadata")

    source = top.add_parser("source")
    source_commands = source.add_subparsers(dest="action", required=True)
    source_add = source_commands.add_parser("add")
    source_add.add_argument("--kind", required=True)
    source_add.add_argument("--title", required=True)
    source_add.add_argument("--author-id")
    source_add.add_argument("--origin")
    source_add.add_argument("--metadata")

    material = top.add_parser("material")
    material_commands = material.add_subparsers(dest="action", required=True)
    material_add = material_commands.add_parser("add")
    material_add.add_argument("--source-id", required=True)
    material_add.add_argument("--kind", required=True)
    material_add.add_argument("--title", required=True)
    material_add.add_argument("--origin", required=True)
    material_add.add_argument("--revision", default="1")
    material_add.add_argument("--metadata")
    material_list = material_commands.add_parser("list")
    material_show = material_commands.add_parser("show")
    material_show.add_argument("--id", required=True)
    attach_artifact = material_commands.add_parser("attach-artifact")
    attach_artifact.add_argument("--material-id", required=True)
    attach_artifact.add_argument("--path", required=True)
    attach_artifact.add_argument("--media-type")

    artifact = top.add_parser("artifact")
    artifact_commands = artifact.add_subparsers(dest="action", required=True)
    artifact_verify = artifact_commands.add_parser("verify")
    artifact_verify.add_argument("--id", required=True)

    media = top.add_parser("media")
    media_commands = media.add_subparsers(dest="action", required=True)
    media_inspect = media_commands.add_parser("inspect")
    media_inspect.add_argument("--artifact-id", required=True)
    media_show = media_commands.add_parser("show")
    media_show.add_argument("--artifact-id", required=True)
    return parser


def _dispatch(
    arguments: argparse.Namespace, service: SourceRegistryService, media_service: MediaInspectionService
) -> dict[str, object]:
    if arguments.entity == "author":
        return _author_to_dict(service.register_author(arguments.name, _metadata(arguments.metadata)))
    if arguments.entity == "source":
        return _source_to_dict(
            service.register_source(
                kind=arguments.kind, title=arguments.title, author_id=arguments.author_id,
                origin=arguments.origin, metadata=_metadata(arguments.metadata),
            )
        )
    if arguments.entity == "artifact":
        return _verification_to_dict(service.verify_artifact(arguments.id))
    if arguments.entity == "media":
        if arguments.action == "inspect":
            inspection, reused = media_service.inspect_artifact(arguments.artifact_id)
            result = _inspection_to_dict(inspection)
            result["reused_existing_inspection"] = reused
            return result
        return _inspection_to_dict(media_service.show_latest(arguments.artifact_id))
    if arguments.action == "add":
        return _material_to_dict(
            service.register_material(
                source_id=arguments.source_id, kind=arguments.kind, title=arguments.title,
                origin=arguments.origin, revision=arguments.revision, metadata=_metadata(arguments.metadata),
            )
        )
    if arguments.action == "list":
        return {"materials": [_material_to_dict(item) for item in service.list_materials()]}
    if arguments.action == "show":
        return _record_to_dict(service.show_material(arguments.id))
    artifact, reused = service.attach_local_artifact(
        material_id=arguments.material_id, path=Path(arguments.path), media_type=arguments.media_type
    )
    result = _artifact_to_dict(artifact)
    result["reused_existing_artifact"] = reused
    return result


def _metadata(raw_value: str | None) -> dict[str, object] | None:
    if raw_value is None:
        return None
    try:
        decoded = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValidationError("--metadata must be a JSON object") from exc
    if not isinstance(decoded, dict):
        raise ValidationError("--metadata must be a JSON object")
    return decoded


def _author_to_dict(author: Author) -> dict[str, object]:
    return {"id": author.id, "name": author.name, "created_at": author.created_at.isoformat(), "metadata": author.metadata.as_dict()}


def _source_to_dict(source: Source) -> dict[str, object]:
    return {
        "id": source.id, "kind": source.kind, "title": source.title, "created_at": source.created_at.isoformat(),
        "author_id": source.author_id, "origin": source.origin, "metadata": source.metadata.as_dict(),
    }


def _material_to_dict(material: Material) -> dict[str, object]:
    return {
        "id": material.id, "source_id": material.source_id, "kind": material.kind, "title": material.title,
        "origin": material.origin, "revision": material.revision, "ingested_at": material.ingested_at.isoformat(),
        "status": material.status, "metadata": material.metadata.as_dict(),
    }


def _artifact_to_dict(artifact: SourceArtifact) -> dict[str, object]:
    return {
        "id": artifact.id,
        "original_location": artifact.original_location,
        "original_filename": artifact.original_filename,
        "managed_artifact": artifact.managed_key,
        "sha256": artifact.sha256,
        "byte_size": artifact.byte_size,
        "observed_at": artifact.observed_at.isoformat(),
        "media_type": artifact.media_type,
    }


def _verification_to_dict(verification: ArtifactVerification) -> dict[str, object]:
    return {
        "artifact_id": verification.artifact_id,
        "managed_artifact": verification.managed_key,
        "exists": verification.exists,
        "is_valid": verification.is_valid,
        "actual_sha256": verification.actual_sha256,
        "actual_byte_size": verification.actual_byte_size,
        "message": verification.message,
    }


def _inspection_to_dict(inspection: MediaInspection) -> dict[str, object]:
    return {
        "id": inspection.id,
        "artifact_id": inspection.artifact_id,
        "tool": inspection.tool,
        "tool_version": inspection.tool_version,
        "schema_version": inspection.schema_version,
        "inspected_at": inspection.inspected_at.isoformat(),
        "status": inspection.status,
        "classification": inspection.classification.value,
        "formats": list(inspection.format_names),
        "duration_seconds": inspection.duration_seconds,
        "bit_rate": inspection.bit_rate,
        "stream_count": inspection.stream_count,
        "has_video": bool(inspection.video_streams),
        "has_audio": bool(inspection.audio_streams),
        "has_subtitles": bool(inspection.subtitle_streams),
        "video_streams": [_video_stream_to_dict(stream) for stream in inspection.video_streams],
        "audio_streams": [_audio_stream_to_dict(stream) for stream in inspection.audio_streams],
        "subtitle_streams": [_subtitle_stream_to_dict(stream) for stream in inspection.subtitle_streams],
    }


def _video_stream_to_dict(stream: VideoStream) -> dict[str, object]:
    return {"index": stream.index, "codec": stream.codec, "width": stream.width, "height": stream.height,
            "frame_rate": stream.frame_rate, "pixel_format": stream.pixel_format,
            "duration_seconds": stream.duration_seconds, "is_default": stream.is_default, "language": stream.language}


def _audio_stream_to_dict(stream: AudioStream) -> dict[str, object]:
    return {"index": stream.index, "codec": stream.codec, "sample_rate": stream.sample_rate,
            "channels": stream.channels, "channel_layout": stream.channel_layout,
            "duration_seconds": stream.duration_seconds, "bit_rate": stream.bit_rate,
            "is_default": stream.is_default, "language": stream.language}


def _subtitle_stream_to_dict(stream: SubtitleStream) -> dict[str, object]:
    return {"index": stream.index, "codec": stream.codec, "is_default": stream.is_default,
            "language": stream.language}


def _record_to_dict(record: MaterialRecord) -> dict[str, object]:
    return {
        "material": _material_to_dict(record.material),
        "source": _source_to_dict(record.source),
        "author": _author_to_dict(record.author) if record.author else None,
        "artifacts": [_artifact_to_dict(item) for item in record.artifacts],
    }
