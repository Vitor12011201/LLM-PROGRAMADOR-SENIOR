from datetime import UTC, datetime
import unittest

from engineering_brain.domain.errors import ValidationError
from engineering_brain.domain.models import (
    DerivedAudioArtifact,
    Material,
    Metadata,
    SourceArtifact,
    TranscriptSegment,
    TranscriptionRun,
    TranscriptionStatus,
)


class DomainModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)

    def test_material_requires_provenance_fields(self) -> None:
        with self.assertRaisesRegex(ValidationError, "material origin"):
            Material("material-1", "source-1", "article", "Title", "", "1", self.now)

    def test_metadata_is_canonical_and_does_not_expose_internal_mutability(self) -> None:
        metadata = Metadata.from_mapping({"z": [1, 2], "a": {"flag": True}})
        self.assertEqual(metadata.json_value, '{"a":{"flag":true},"z":[1,2]}')
        copy = metadata.as_dict()
        copy["a"]["flag"] = False
        self.assertEqual(metadata.as_dict()["a"]["flag"], True)

    def test_artifact_requires_valid_sha256(self) -> None:
        with self.assertRaisesRegex(ValidationError, "sha256"):
            SourceArtifact("artifact-1", "sha256/ba/bad", "bad", 1, self.now)

    def test_derived_audio_config_is_canonicalized(self) -> None:
        artifact = DerivedAudioArtifact(
            "derived-1", "selection-1", "sha256/aa/" + "a" * 64,
            "a" * 64, 1, "ffmpeg", "1", "audio-derivation-v1",
            '{"sample_rate":16000,"channels":1}', self.now,
        )

        self.assertEqual(artifact.config_json, '{"channels":1,"sample_rate":16000}')

    def test_transcription_run_config_is_canonicalized(self) -> None:
        run = TranscriptionRun(
            "run-1", "derived-1", "engine", "1", "model", None, "cpu", "int8",
            '{"word_timestamps":true,"requested_language":null,"vad_enabled":false}',
            None, None, None, self.now, None, TranscriptionStatus.RUNNING,
        )

        self.assertEqual(
            run.config_json,
            '{"requested_language":null,"vad_enabled":false,"word_timestamps":true}',
        )

    def test_transcript_segment_words_are_canonicalized(self) -> None:
        segment = TranscriptSegment(
            "segment-1", "transcript-1", 0, "0", "1", "text",
            words_json='[{"word":"heap","end":0.5,"start":0,"probability":0.98}]',
        )

        self.assertEqual(
            segment.words_json,
            '[{"end":0.5,"probability":0.98,"start":0,"word":"heap"}]',
        )
