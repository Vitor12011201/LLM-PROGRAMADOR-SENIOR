class RegistryError(Exception):
    """Base class for expected source-registry failures."""


class ValidationError(RegistryError):
    """Input does not satisfy a domain invariant."""


class NotFoundError(RegistryError):
    """A referenced registry record does not exist."""


class DuplicateError(RegistryError):
    """An attempted registration would duplicate an existing relation."""


class ArtifactFileNotFoundError(RegistryError):
    """The requested local artifact cannot be read as a file."""


class ArtifactStoreError(RegistryError):
    """The managed artifact store could not complete an operation."""


class ArtifactIntegrityError(ArtifactStoreError):
    """A managed artifact does not match its expected content identity."""


class MediaInspectionError(RegistryError):
    """A media inspection could not be completed safely."""


class MediaInspectorUnavailableError(MediaInspectionError):
    """The configured inspection tool is unavailable."""


class MediaInspectorTimeoutError(MediaInspectionError):
    """The inspection tool did not finish before its timeout."""


class MediaInspectorExecutionError(MediaInspectionError):
    """The inspection tool exited unsuccessfully."""


class MediaInspectorOutputError(MediaInspectionError):
    """The inspection tool returned invalid structured output."""


class RepositoryError(RegistryError):
    """The persistence adapter could not complete an operation."""
