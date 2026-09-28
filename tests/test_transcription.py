from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from engineering_brain.application.audio_pipeline import AudioPipelineService
from engineering_brain.application.source_registry import SourceRegistryService
from engineering_brain.application.transcription import TranscriptionConfig, TranscriptionService
from engineering_brain.domain.errors import InvalidTranscriptionResultError, RepositoryError
from engineering_brain.domain.models import AudioStream, DerivedAudioArtifact, MediaClassification, MediaInspection
from engineering_brain.infrastructure.local_artifact_store import LocalArtifactStore
from engineering_brain.infrastructure.sqlite_source_registry import SqliteSourceRegistryRepository
from tests.support.fake_transcriber import FakeTranscriber


class Extractor:
    def tool_identity(self): return "fake-extractor", "1"
    def extract(self, source, index, destination): destination.write_bytes(b"wav")


class _PostCompletionReadFailureRepository:
    """Fails only after the real SQLite completion transaction has committed."""

    def __init__(self, repository: SqliteSourceRegistryRepository) -> None:
        self._repository = repository
        self._completion_committed = False

    def __getattr__(self, name: str):
        return getattr(self._repository, name)

    def complete_transcription_run(self, *args, **kwargs) -> None:
        self._repository.complete_transcription_run(*args, **kwargs)
        self._completion_committed = True

    def get_transcription_run(self, run_id: str):
        if self._completion_committed:
            raise RepositoryError("forced post-completion read failure")
        return self._repository.get_transcription_run(run_id)


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(); self.root = Path(self.temp.name)
        self.repo = SqliteSourceRegistryRepository(self.root / "db.sqlite"); self.repo.initialize(); self.store = LocalArtifactStore(self.root / "artifacts")
        self.registry = SourceRegistryService(self.repo, self.store); author = self.registry.register_author("a"); source = self.registry.register_source(kind="v", title="s", author_id=author.id); self.material = self.registry.register_material(source_id=source.id, kind="v", title="m", origin="file://m")
        source_file = self.root / "x"; source_file.write_bytes(b"x"); artifact = self.registry.attach_local_artifact(material_id=self.material.id, path=source_file)[0].artifact
        inspection = MediaInspection("i", artifact.id, "p", "1", "v", __import__("datetime").datetime.now(__import__("datetime").UTC), "completed", MediaClassification.AUDIO, (), None, None, 1, (), (AudioStream(0,"a",None,None,None,None,None,False,None),), (), "{}")
        self.repo.create_media_inspection(inspection); selection = AudioPipelineService(self.repo,self.store,Extractor()).select_audio(artifact.id,inspection.id)[0]; self.derived = AudioPipelineService(self.repo,self.store,Extractor()).derive_audio(selection.id)[0]

    def tearDown(self): self.temp.cleanup()

    def test_success_persists_transcript_segments_and_reuses_equivalent(self):
        fake=FakeTranscriber(); service=TranscriptionService(self.repo,self.store,fake); run,reused=service.transcribe(self.derived.id,TranscriptionConfig())
        transcript=self.repo.get_transcript_for_run(run.id); segments=self.repo.segments_for_transcript(transcript.id); again,reused_again=service.transcribe(self.derived.id,TranscriptionConfig())
        self.assertFalse(reused); self.assertTrue(reused_again); self.assertEqual(run.status.value,"completed"); self.assertEqual(fake.call_count,1); self.assertEqual([x.ordinal for x in segments],[0,1]); self.assertEqual(json.loads(segments[0].words_json)[0]["word"],"heap"); self.assertEqual(run.detected_language,"pt")

    def test_transcription_preserves_segment_engine_metadata(self):
        run, _ = TranscriptionService(self.repo, self.store, FakeTranscriber()).transcribe(
            self.derived.id, TranscriptionConfig()
        )
        transcript = self.repo.get_transcript_for_run(run.id)
        first = self.repo.segments_for_transcript(transcript.id)[0]

        self.assertEqual(first.avg_logprob, "-0.12")
        self.assertEqual(first.no_speech_prob, "0.03")
        self.assertEqual(first.compression_ratio, "1.17")
        self.assertEqual(first.temperature, "0.0")
        self.assertEqual(
            first.words_json,
            '[{"end":0.5,"probability":0.98,"start":0,"word":"heap"}]',
        )

    def test_completed_run_remains_completed_if_post_completion_read_fails(self):
        repository = _PostCompletionReadFailureRepository(self.repo)
        service = TranscriptionService(repository, self.store, FakeTranscriber())

        with self.assertRaisesRegex(RepositoryError, "post-completion read"):
            service.transcribe(self.derived.id, TranscriptionConfig())

        row = self.repo._rows("SELECT * FROM transcription_runs", ())[0]
        self.assertEqual(row["status"], "completed")
        transcript = self.repo.get_transcript_for_run(row["id"])
        self.assertEqual(len(self.repo.segments_for_transcript(transcript.id)), 2)

    def test_failure_marks_run_failed_without_transcript(self):
        service=TranscriptionService(self.repo,self.store,FakeTranscriber(failure=True))
        with self.assertRaises(Exception): service.transcribe(self.derived.id,TranscriptionConfig())
        # The only run is failed; a transcript lookup must fail.
        run=self.repo._rows("SELECT * FROM transcription_runs",())[0]
        self.assertEqual(run["status"],"failed")

    def test_invalid_probability_marks_run_failed(self):
        fake=FakeTranscriber()
        fake.transcribe=lambda path, config: __import__("engineering_brain.ports.transcription",fromlist=["TranscriptionResult"]).TranscriptionResult("bad","pt","pt","1.1",())
        with self.assertRaises(Exception): TranscriptionService(self.repo,self.store,fake).transcribe(self.derived.id,TranscriptionConfig())
        self.assertEqual(self.repo._rows("SELECT * FROM transcription_runs",())[0]["status"],"failed")

    def _invalid(self, result):
        from engineering_brain.ports.transcription import TranscriptionResult
        fake=FakeTranscriber(); fake.transcribe=lambda path, config: result
        with self.assertRaises(InvalidTranscriptionResultError):
            TranscriptionService(self.repo,self.store,fake).transcribe(self.derived.id,TranscriptionConfig())
        run=self.repo._rows("SELECT * FROM transcription_runs",())[0]
        self.assertEqual(run["status"],"failed")
        self.assertEqual(self.repo._rows("SELECT * FROM transcripts",()),[])
        self.assertEqual(self.repo._rows("SELECT * FROM transcript_segments",()),[])

    def test_negative_language_probability_marks_run_failed(self):
        from engineering_brain.ports.transcription import TranscriptionResult
        self._invalid(TranscriptionResult("x","pt","pt","-0.01",()))

    def test_language_probability_above_one_marks_run_failed(self):
        from engineering_brain.ports.transcription import TranscriptionResult
        self._invalid(TranscriptionResult("x","pt","pt","1.01",()))

    def test_negative_segment_start_marks_run_failed(self): self._invalid(_segment_result("-0.1","1"))
    def test_segment_end_before_start_marks_run_failed(self): self._invalid(_segment_result("2","1"))
    def test_negative_segment_ordinal_marks_run_failed(self): self._invalid(_segment_result("0","1",ordinal=-1))
    def test_duplicate_segment_ordinals_mark_run_failed(self):
        from engineering_brain.ports.transcription import TranscriptionResult, TranscriptionSegmentResult
        self._invalid(TranscriptionResult("x","pt","pt","0.98",(TranscriptionSegmentResult("0","1","a",ordinal=0),TranscriptionSegmentResult("1","2","b",ordinal=0))))
    def test_negative_word_start_marks_run_failed(self): self._invalid(_segment_result("0","1",words='[{"word":"heap","start":-0.1,"end":0.4}]'))
    def test_word_end_before_start_marks_run_failed(self): self._invalid(_segment_result("0","1",words='[{"word":"heap","start":0.8,"end":0.4}]'))

    def test_malformed_word_json_marks_run_failed(self):
        self._invalid(_segment_result("0", "1", words="not json"))

    def test_invalid_word_probability_marks_run_failed(self):
        self._invalid(
            _segment_result(
                "0", "1",
                words='[{"word":"heap","start":0,"end":0.4,"probability":1.01}]',
            )
        )

    def test_failed_run_does_not_block_successful_retry(self):
        fake=FakeTranscriber(failure=True); service=TranscriptionService(self.repo,self.store,fake)
        with self.assertRaises(Exception): service.transcribe(self.derived.id,TranscriptionConfig())
        failed=self.repo.get_transcription_run(self.repo._rows("SELECT * FROM transcription_runs",())[0]["id"]); fake.failure=False
        completed,_=service.transcribe(self.derived.id,TranscriptionConfig())
        self.assertNotEqual(failed.id,completed.id); self.assertEqual(completed.status.value,"completed"); self.assertEqual(fake.call_count,2); self.assertIsNotNone(self.repo.get_transcript_for_run(completed.id))

    def test_completed_equivalent_transcription_is_reused(self):
        fake=FakeTranscriber(); service=TranscriptionService(self.repo,self.store,fake); first,_=service.transcribe(self.derived.id,TranscriptionConfig()); count=len(self.repo._rows("SELECT * FROM transcription_runs",()))
        second,reused=service.transcribe(self.derived.id,TranscriptionConfig())
        self.assertTrue(reused); self.assertEqual(first.id,second.id); self.assertEqual(fake.call_count,1); self.assertEqual(len(self.repo._rows("SELECT * FROM transcription_runs",())),count)

    def test_completed_run_is_reused_even_if_failed_equivalent_run_also_exists(self):
        fake=FakeTranscriber(failure=True); service=TranscriptionService(self.repo,self.store,fake)
        with self.assertRaises(Exception): service.transcribe(self.derived.id,TranscriptionConfig())
        fake.failure=False; completed,_=service.transcribe(self.derived.id,TranscriptionConfig()); calls=fake.call_count; again,reused=service.transcribe(self.derived.id,TranscriptionConfig())
        self.assertTrue(reused); self.assertEqual(again.id,completed.id); self.assertEqual(fake.call_count,calls); self.assertEqual(len(self.repo._rows("SELECT * FROM transcription_runs",())),2)

    def test_running_equivalent_run_is_not_reused_as_completed_result(self):
        fake=FakeTranscriber(); service=TranscriptionService(self.repo,self.store,fake)
        engine,version,model,revision,device,compute=fake.identity(); from engineering_brain.domain.models import TranscriptionRun, TranscriptionStatus; from datetime import datetime, UTC
        self.repo.create_transcription_run(TranscriptionRun("running",self.derived.id,engine,version,model,revision,device,compute,TranscriptionConfig().json_value,None,None,None,datetime.now(UTC),None,TranscriptionStatus.RUNNING))
        run,reused=service.transcribe(self.derived.id,TranscriptionConfig())
        self.assertFalse(reused); self.assertEqual(run.status.value,"completed"); self.assertEqual(fake.call_count,1)

    def _identity_change(self, **changes):
        first=FakeTranscriber(); TranscriptionService(self.repo,self.store,first).transcribe(self.derived.id,TranscriptionConfig())
        second=FakeTranscriber(**changes); run,reused=TranscriptionService(self.repo,self.store,second).transcribe(self.derived.id,TranscriptionConfig())
        self.assertFalse(reused); self.assertEqual(second.call_count,1); self.assertNotEqual(run.id,self.repo._rows("SELECT * FROM transcription_runs",())[0]["id"])
    def test_different_model_revision_does_not_reuse_completed_run(self): self._identity_change(model_revision="r2")
    def test_different_engine_version_does_not_reuse_completed_run(self): self._identity_change(engine_version="2")
    def test_different_compute_type_does_not_reuse_completed_run(self): self._identity_change(compute_type="float16")
    def test_different_engine_does_not_reuse_completed_run(self): self._identity_change(engine="other")
    def test_different_model_id_does_not_reuse_completed_run(self): self._identity_change(model_id="other")
    def test_different_device_does_not_reuse_completed_run(self): self._identity_change(device="cuda")
    def test_none_model_revision_is_not_equal_to_literal_none_string(self):
        first = FakeTranscriber(model_revision=None)
        TranscriptionService(self.repo, self.store, first).transcribe(self.derived.id, TranscriptionConfig())
        second = FakeTranscriber(model_revision="None")
        run, reused = TranscriptionService(self.repo, self.store, second).transcribe(
            self.derived.id, TranscriptionConfig()
        )
        self.assertFalse(reused)
        self.assertEqual(second.call_count, 1)
        self.assertNotEqual(run.id, self.repo._rows("SELECT * FROM transcription_runs", ())[0]["id"])
    def test_different_transcription_config_does_not_reuse_completed_run(self):
        fake=FakeTranscriber(); service=TranscriptionService(self.repo,self.store,fake); first,_=service.transcribe(self.derived.id,TranscriptionConfig()); second,reused=service.transcribe(self.derived.id,TranscriptionConfig(vad_enabled=True)); self.assertFalse(reused); self.assertNotEqual(first.id,second.id); self.assertEqual(fake.call_count,2)
    def test_semantically_identical_config_serializes_canonically(self):
        self.assertEqual(TranscriptionConfig().json_value, TranscriptionConfig().json_value)

    def test_corrupted_derived_blob_is_rejected_before_run_creation(self):
        path=self.store.managed_path(self.derived.managed_key); path.chmod(0o644); path.write_bytes(b"corrupt"); fake=FakeTranscriber(); before=len(self.repo._rows("SELECT * FROM transcription_runs",()))
        with self.assertRaises(Exception): TranscriptionService(self.repo,self.store,fake).transcribe(self.derived.id,TranscriptionConfig())
        self.assertEqual(fake.call_count,0); self.assertEqual(len(self.repo._rows("SELECT * FROM transcription_runs",())),before)
        self.assertEqual(self.repo._rows("SELECT * FROM transcripts",()), [])
        self.assertEqual(self.repo._rows("SELECT * FROM transcript_segments",()), [])

    def test_missing_derived_blob_is_rejected_before_run_creation(self):
        self.store.managed_path(self.derived.managed_key).unlink(); fake=FakeTranscriber(); before=len(self.repo._rows("SELECT * FROM transcription_runs",()))
        with self.assertRaises(Exception): TranscriptionService(self.repo,self.store,fake).transcribe(self.derived.id,TranscriptionConfig())
        self.assertEqual(fake.call_count,0); self.assertEqual(len(self.repo._rows("SELECT * FROM transcription_runs",())),before)
        self.assertEqual(self.repo._rows("SELECT * FROM transcripts",()), [])
        self.assertEqual(self.repo._rows("SELECT * FROM transcript_segments",()), [])

    def test_different_derived_audio_artifact_does_not_reuse_completed_run(self):
        source_file = self.root / "other-source"
        source_file.write_bytes(b"different source evidence")
        other_source_artifact = self.registry.attach_local_artifact(
            material_id=self.material.id, path=source_file
        )[0].artifact
        other_inspection = MediaInspection(
            "i-other", other_source_artifact.id, "p", "1", "v", datetime.now(UTC),
            "completed", MediaClassification.AUDIO, (), None, None, 1, (),
            (AudioStream(0, "a", None, None, None, None, None, False, None),), (), "{}",
        )
        self.repo.create_media_inspection(other_inspection)
        other_selection = AudioPipelineService(self.repo, self.store, Extractor()).select_audio(
            other_source_artifact.id, other_inspection.id
        )[0]
        other_file = self.root / "other.wav"
        other_file.write_bytes(b"different canonical wav")
        stored = self.store.ingest(other_file)
        other = DerivedAudioArtifact(
            "derived-other",
            other_selection.id,
            stored.managed_key,
            stored.sha256,
            stored.byte_size,
            "fake-extractor",
            "1",
            "audio-derivation-v1",
            '{"channels":1,"codec":"pcm_s16le","container":"wav","sample_rate":16000}',
            datetime.now(UTC),
        )
        self.repo.create_derived_audio_artifact(other)

        fake = FakeTranscriber()
        service = TranscriptionService(self.repo, self.store, fake)
        first, first_reused = service.transcribe(self.derived.id, TranscriptionConfig())
        second, second_reused = service.transcribe(other.id, TranscriptionConfig())

        self.assertFalse(first_reused)
        self.assertFalse(second_reused)
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(fake.call_count, 2)


def _segment_result(start, end, ordinal=None, words=None):
    from engineering_brain.ports.transcription import TranscriptionResult, TranscriptionSegmentResult
    return TranscriptionResult("x","pt","pt","0.98",(TranscriptionSegmentResult(start,end,"x",words,ordinal),))
