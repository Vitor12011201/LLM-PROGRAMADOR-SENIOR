from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from engineering_brain.cli import main


class CommandLineFlowTests(unittest.TestCase):
    def test_cli_registers_and_shows_a_material(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            environment = {"ENGINEERING_BRAIN_DATA_DIR": str(root / "data")}
            with patch.dict(os.environ, environment, clear=False):
                author = self._run("author", "add", "--name", "CLI author")
                source = self._run(
                    "source", "add", "--kind", "collection", "--title", "CLI source", "--author-id", author["id"]
                )
                material = self._run(
                    "material", "add", "--source-id", source["id"], "--kind", "article", "--title", "CLI material",
                    "--origin", "https://example.test/cli",
                )
                artifact_path = root / "evidence.txt"
                artifact_path.write_text("CLI evidence", encoding="utf-8")
                artifact = self._run(
                    "material", "attach-artifact", "--material-id", material["id"], "--path", str(artifact_path)
                )
                record = self._run("material", "show", "--id", material["id"])
                verification = self._run("artifact", "verify", "--id", artifact["id"])

            self.assertFalse(artifact["reused_existing_artifact"])
            self.assertEqual(artifact["original_location"], artifact_path.absolute().as_uri())
            self.assertIsNotNone(artifact["managed_artifact"])
            self.assertEqual(record["author"]["id"], author["id"])
            self.assertEqual(record["source"]["id"], source["id"])
            self.assertEqual(record["material"]["id"], material["id"])
            self.assertEqual(len(record["artifacts"]), 1)
            self.assertTrue(verification["is_valid"])

    def _run(self, *arguments: str) -> dict[str, object]:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(arguments)
        self.assertEqual(exit_code, 0)
        return json.loads(output.getvalue())
