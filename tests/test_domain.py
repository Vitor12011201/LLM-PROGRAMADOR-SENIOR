from datetime import UTC, datetime
import unittest

from engineering_brain.domain.errors import ValidationError
from engineering_brain.domain.models import Material, Metadata, SourceArtifact


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
