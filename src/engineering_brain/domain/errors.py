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


class AudioSelectionError(RegistryError):
    """An inspected artifact cannot yield the requested audio selection."""


class NoAudioStreamError(AudioSelectionError):
    """The inspected media contains no audio stream."""


class AmbiguousAudioSelectionError(AudioSelectionError):
    """More than one audio stream requires an explicit selection."""


class InvalidAudioSelectionError(AudioSelectionError):
    """An inspection cannot be used with the requested source artifact."""


class InvalidAudioStreamError(AudioSelectionError):
    """The explicitly selected stream is not an audio stream."""


class AudioExtractionError(RegistryError):
    """Deterministic audio extraction could not complete safely."""


class TranscriptionError(RegistryError):
    """A transcription engine could not complete a run."""


class InvalidTranscriptionResultError(TranscriptionError):
    """A transcriber returned inconsistent structured output."""
