from __future__ import annotations

import json

from engineering_brain.ports.transcription import TranscriptionResult, TranscriptionSegmentResult


class FakeTranscriber:
    def __init__(self, *, failure: bool = False, engine: str = "fake", engine_version: str = "1", model_id: str = "fixture-model", model_revision: str | None = "r1", device: str | None = "cpu", compute_type: str | None = "int8") -> None:
        self.failure, self.engine, self.engine_version, self.model_id, self.model_revision, self.device, self.compute_type = failure, engine, engine_version, model_id, model_revision, device, compute_type
        self.call_count = 0

    def identity(self): return self.engine, self.engine_version, self.model_id, self.model_revision, self.device, self.compute_type
    def transcribe(self, audio_path, config_json):
        self.call_count += 1
        if self.failure: raise RuntimeError("fake transcription failure")
        words = json.dumps([{"word":"heap","start":0,"end":0.5,"probability":0.98}], separators=(",",":"), sort_keys=True)
        return TranscriptionResult("O heap é uma região de memória.", "pt", "pt", "0.98", (
            TranscriptionSegmentResult("0", "1.5", "O heap", words, None, "-0.12", "0.03", "1.17", "0.0"),
            TranscriptionSegmentResult("1.5", "3.0", "é uma região de memória.", None, None, "-0.08", "0.01", "1.05", "0.2"),
        ))
