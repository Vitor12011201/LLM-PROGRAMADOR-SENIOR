from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

from engineering_brain.domain.errors import ArtifactFileNotFoundError, ArtifactIntegrityError, ArtifactStoreError
from engineering_brain.ports.artifact_store import ArtifactStore, StoreVerification, StoredArtifact


class LocalArtifactStore(ArtifactStore):
    """Content-addressed filesystem storage owned by Engineering Brain."""

    def __init__(self, root_directory: Path) -> None:
        self._root_directory = root_directory.expanduser()

    def ingest(self, source_path: Path) -> StoredArtifact:
        original_path = source_path.expanduser().absolute()
        if not original_path.is_file():
            raise ArtifactFileNotFoundError(f"local artifact does not exist or is not a file: {source_path}")

        temporary_directory = self._root_directory / ".tmp"
        try:
            temporary_directory.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(prefix="ingest-", dir=temporary_directory)
        except OSError as exc:
            raise ArtifactStoreError(f"could not prepare managed artifact storage: {exc}") from exc

        temporary_path = Path(temporary_name)
        try:
            sha256, byte_size = _copy_and_hash(original_path, descriptor)
            if _sha256_file(temporary_path) != sha256:
                raise ArtifactIntegrityError("copied artifact failed SHA-256 verification before storage")

            managed_key = _managed_key(sha256)
            destination = self.managed_path(managed_key)
            destination.parent.mkdir(parents=True, exist_ok=True)
            reused = self._publish_or_verify_existing(temporary_path, destination, sha256)
            return StoredArtifact(
                original_location=original_path.as_uri(),
                original_filename=original_path.name,
                managed_key=managed_key,
                sha256=sha256,
                byte_size=byte_size,
                reused_existing_blob=reused,
            )
        except OSError as exc:
            raise ArtifactStoreError(f"could not ingest local artifact {source_path}: {exc}") from exc
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    def verify(self, managed_key: str, expected_sha256: str) -> StoreVerification:
        path = self.managed_path(managed_key)
        if not path.is_file():
            return StoreVerification(False, False, None, None)
        try:
            actual_sha256 = _sha256_file(path)
            return StoreVerification(
                exists=True,
                matches_expected_hash=actual_sha256 == expected_sha256,
                actual_sha256=actual_sha256,
                actual_byte_size=path.stat().st_size,
            )
        except OSError as exc:
            raise ArtifactStoreError(f"could not verify managed artifact {managed_key}: {exc}") from exc

    def path_for_read(self, managed_key: str) -> Path:
        return self.managed_path(managed_key)

    def managed_path(self, managed_key: str) -> Path:
        parts = managed_key.split("/")
        if len(parts) != 3 or parts[0] != "sha256" or len(parts[1]) != 2 or len(parts[2]) != 64:
            raise ArtifactStoreError(f"invalid managed artifact key: {managed_key}")
        if any(character not in "0123456789abcdef" for character in parts[1] + parts[2]):
            raise ArtifactStoreError(f"invalid managed artifact key: {managed_key}")
        if parts[2][:2] != parts[1]:
            raise ArtifactStoreError(f"managed artifact key does not match its hash prefix: {managed_key}")
        return self._root_directory.joinpath(*parts)

    def _publish_or_verify_existing(self, temporary_path: Path, destination: Path, sha256: str) -> bool:
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            verification = self.verify(_managed_key(sha256), sha256)
            if not verification.matches_expected_hash:
                raise ArtifactIntegrityError(
                    f"managed artifact already exists but does not match SHA-256 {sha256}"
                )
            return True
        try:
            destination.chmod(0o444)
        except OSError as exc:
            raise ArtifactStoreError(f"could not make managed artifact read-only: {exc}") from exc
        return False


def _copy_and_hash(source_path: Path, destination_descriptor: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_size = 0
    try:
        with source_path.open("rb") as source, os.fdopen(destination_descriptor, "wb") as destination:
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)
                digest.update(chunk)
                byte_size += len(chunk)
            destination.flush()
            os.fsync(destination.fileno())
    except OSError:
        try:
            os.close(destination_descriptor)
        except OSError:
            pass
        raise
    return digest.hexdigest(), byte_size


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _managed_key(sha256: str) -> str:
    return f"sha256/{sha256[:2]}/{sha256}"
