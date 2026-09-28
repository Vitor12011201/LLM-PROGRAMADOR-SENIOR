from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping
from uuid import uuid4

from engineering_brain.domain.models import (
    ArtifactVerification,
    Author,
    Material,
    MaterialRecord,
    Metadata,
    Source,
    SourceArtifact,
)
from engineering_brain.ports.artifact_store import ArtifactStore
from engineering_brain.ports.source_registry import SourceRegistryRepository


class SourceRegistryService:
    """Application boundary for append-only source evidence registration."""

    def __init__(
        self,
        repository: SourceRegistryRepository,
        artifact_store: ArtifactStore,
        *,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._repository = repository
        self._artifact_store = artifact_store
        self._id_factory = id_factory or (lambda: str(uuid4()))
        self._clock = clock or (lambda: datetime.now(UTC))

    def register_author(self, name: str, metadata: Mapping[str, object] | None = None) -> Author:
        author = Author(self._id_factory(), name, self._clock(), Metadata.from_mapping(metadata))
        self._repository.create_author(author)
        return author

    def register_source(
        self,
        *,
        kind: str,
        title: str,
        author_id: str | None = None,
        origin: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Source:
        if author_id is not None:
            self._repository.get_author(author_id)
        source = Source(
            self._id_factory(), kind, title, self._clock(), author_id, origin, Metadata.from_mapping(metadata)
        )
        self._repository.create_source(source)
        return source

    def register_material(
        self,
        *,
        source_id: str,
        kind: str,
        title: str,
        origin: str,
        revision: str = "1",
        metadata: Mapping[str, object] | None = None,
    ) -> Material:
        self._repository.get_source(source_id)
        material = Material(
            self._id_factory(),
            source_id,
            kind,
            title,
            origin,
            revision,
            self._clock(),
            metadata=Metadata.from_mapping(metadata),
        )
        self._repository.create_material(material)
        return material

    def attach_local_artifact(
        self, *, material_id: str, path: Path, media_type: str | None = None
    ) -> tuple[SourceArtifact, bool]:
        self._repository.get_material(material_id)
        stored = self._artifact_store.ingest(path)
        candidate = SourceArtifact(
            id=self._id_factory(),
            original_location=stored.original_location,
            managed_key=stored.managed_key,
            sha256=stored.sha256,
            byte_size=stored.byte_size,
            observed_at=self._clock(),
            original_filename=stored.original_filename,
            media_type=media_type,
        )
        return self._repository.attach_artifact(material_id, candidate)

    def verify_artifact(self, artifact_id: str) -> ArtifactVerification:
        artifact = self._repository.get_artifact(artifact_id)
        if artifact.managed_key is None:
            return ArtifactVerification(
                artifact_id=artifact.id,
                managed_key=None,
                exists=False,
                is_valid=False,
                actual_sha256=None,
                actual_byte_size=None,
                message="artifact was registered before managed storage was available",
            )
        verification = self._artifact_store.verify(artifact.managed_key, artifact.sha256)
        if not verification.exists:
            message = "managed artifact is missing"
        elif verification.matches_expected_hash:
            message = "managed artifact matches its registered SHA-256"
        else:
            message = "managed artifact SHA-256 does not match its registered value"
        return ArtifactVerification(
            artifact_id=artifact.id,
            managed_key=artifact.managed_key,
            exists=verification.exists,
            is_valid=verification.matches_expected_hash,
            actual_sha256=verification.actual_sha256,
            actual_byte_size=verification.actual_byte_size,
            message=message,
        )

    def show_material(self, material_id: str) -> MaterialRecord:
        material = self._repository.get_material(material_id)
        source = self._repository.get_source(material.source_id)
        author = self._repository.get_author(source.author_id) if source.author_id else None
        return MaterialRecord(material, source, author, tuple(self._repository.artifacts_for_material(material_id)))

    def list_materials(self) -> list[Material]:
        return self._repository.list_materials()
