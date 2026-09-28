# Engineering Brain

Engineering Brain is a local-first foundation for turning technical sources into a
traceable engineering knowledge pipeline. The current baseline is intentionally
deterministic: it preserves source evidence and records technical media observations
without using an LLM, transcription, or semantic interpretation.

Implemented phases: Source Registry, managed content-addressed artifact storage,
and Media Inspection V1 with `ffprobe`.

## Run locally

The target runtime is **Python >=3.12,<3.13**. The current host validation used
Python 3.14 because Python 3.12 is not installed there yet; Python 3.12 validation
remains pending. The project has no third-party Python runtime dependencies.

Create an isolated environment with Python 3.12:

```bash
python3.12 -m venv .venv
PYTHONPATH=src .venv/bin/python -m engineering_brain author add --name "Example author"
```

By default, local metadata is stored in `data/engineering_brain.sqlite3`, which is
ignored by Git. Managed artifact copies are stored under `data/artifacts/`, also
outside Git. Set `ENGINEERING_BRAIN_DATA_DIR` to use another location; the artifact
directory can be overridden through `ENGINEERING_BRAIN_ARTIFACT_DIR`.

`material attach-artifact` copies the supplied file into content-addressed managed
storage. The source path is retained as provenance, but the stored copy remains
available after the original is moved or removed.

If `ffprobe` is available locally, managed audio and video artifacts can be
inspected deterministically. Inspection metadata is persisted separately from the
source evidence; it does not transcribe or interpret the media.

`ffmpeg` is optional and is only useful for generating local test media. Neither
tool is installed or managed by this project.

## Minimal flow

```bash
PYTHONPATH=src .venv/bin/python -m engineering_brain author add --name "Example author"
PYTHONPATH=src .venv/bin/python -m engineering_brain source add \
  --kind documentation --title "Example docs" --author-id <author-id> \
  --origin "https://example.test/docs"
PYTHONPATH=src .venv/bin/python -m engineering_brain material add \
  --source-id <source-id> --kind article --title "Intro" \
  --origin "https://example.test/docs/intro" --revision "2026-01"
PYTHONPATH=src .venv/bin/python -m engineering_brain material attach-artifact \
  --material-id <material-id> --path ./original-file.pdf
PYTHONPATH=src .venv/bin/python -m engineering_brain material show --id <material-id>
PYTHONPATH=src .venv/bin/python -m engineering_brain artifact verify --id <artifact-id>
PYTHONPATH=src .venv/bin/python -m engineering_brain media inspect --artifact-id <artifact-id>
PYTHONPATH=src .venv/bin/python -m engineering_brain media show --artifact-id <artifact-id>
```

Run the test suite with:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Local state such as `.venv/`, `data/`, managed artifacts, SQLite databases, models,
checkpoints, logs, and environment files is excluded from version control.
