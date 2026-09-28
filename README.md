# Engineering Brain

Engineering Brain is a local-first foundation for turning technical sources into a
traceable engineering knowledge pipeline. The current baseline is intentionally
deterministic: it preserves source evidence, records technical media observations,
and derives canonical audio without semantic interpretation or model inference.

Implemented foundation: Source Registry, managed content-addressed artifact storage,
provenance-aware `SourceArtifact` / `ArtifactObservation`, deterministic Media
Inspection V1 with `ffprobe`, explicit audio-stream selection, and deterministic
canonical WAV derivation with `ffmpeg`. `DerivedAudioArtifact` preserves derivation
provenance; the transcription domain and persistence schema are prepared separately.

Real speech-to-text inference is not implemented yet. Phase 3A does not download or
execute machine-learning models.

## Run locally

The target runtime is **Python >=3.12,<3.13**. The current validation environment is
Python 3.12.14. The project has no third-party Python runtime dependencies.

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

Each managed `SourceArtifact` identifies preserved bytes (SHA-256, byte size and
managed key). Each attachment creates an `ArtifactObservation` for its Material,
retaining the original path and filename. Identical content therefore shares one
managed blob while retaining separate origins; repeating the same Material, bytes
and original location is idempotent.

If `ffprobe` is available locally, managed audio and video artifacts can be
inspected deterministically. `ffmpeg` can derive a selected managed audio stream as
WAV / PCM signed 16-bit little-endian / mono / 16 kHz. Inspection metadata and
derived audio remain separate from source evidence; neither operation transcribes or
interprets media.

`ffprobe` and `ffmpeg` are environment tools: neither is installed or managed by this
project.

The provenance boundary is intentionally explicit:

```text
SourceArtifact
→ MediaInspection
→ AudioSelection
→ DerivedAudioArtifact
→ TranscriptionRun
→ Transcript
→ TranscriptSegment
```

`SourceArtifact != DerivedAudioArtifact != TranscriptionRun != Transcript !=
Interpretation != Knowledge`. `TranscriptionRun` and transcript records are only a
foundation at this stage; no real ASR engine is connected.

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
PYTHONPATH=src .venv/bin/python -m engineering_brain audio select \
  --artifact-id <artifact-id> --inspection-id <inspection-id> [--stream-index <global-stream-index>]
PYTHONPATH=src .venv/bin/python -m engineering_brain audio selection-show --id <selection-id>
PYTHONPATH=src .venv/bin/python -m engineering_brain audio derive --selection-id <selection-id>
PYTHONPATH=src .venv/bin/python -m engineering_brain audio derived-show --id <derived-audio-id>
```

Run the test suite with:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Local state such as `.venv/`, `data/`, managed artifacts, SQLite databases, models,
checkpoints, logs, and environment files is excluded from version control.
