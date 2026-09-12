import asyncio
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from processing_progress import ProcessingProgress, RuntimeHistory


class ProcessingProgressTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'times.json'
        self.history = RuntimeHistory(self.path)
        self.now = 0.0
        self.cancelled = False
        self.progress = ProcessingProgress(self.history, 'model:cpu', 100,
                                           lambda: self.cancelled, lambda: self.now)
        self.progress.begin('aligning', 'Aligning…', 55, 98)

    def test_unknown_work_is_indeterminate_and_uses_real_elapsed_time(self):
        self.now = 7.25
        snapshot = self.progress.snapshot()
        self.assertTrue(snapshot['indeterminate'])
        self.assertEqual(snapshot['pct'], 55)
        self.assertEqual(snapshot['progress']['elapsed_seconds'], 7.25)
        self.assertIsNone(snapshot['progress']['eta_seconds'])

    def test_measured_rate_eta_and_stall(self):
        self.now = 2
        self.progress.update(20, 100)
        self.now = 4
        self.progress.update(40, 100)
        details = self.progress.snapshot()['progress']
        self.assertAlmostEqual(details['speed'], 10)
        self.assertAlmostEqual(details['eta_seconds'], 6)
        self.now = 14
        stalled = self.progress.snapshot()['progress']
        self.assertGreater(stalled['eta_seconds'], details['eta_seconds'])
        self.assertEqual(stalled['done'], 40)

    def test_percent_never_regresses_or_finishes_before_results(self):
        self.now = 3
        self.progress.update(80, 100)
        previous = self.progress.snapshot()['pct']
        self.progress.update(80, 200)
        self.assertGreaterEqual(self.progress.snapshot()['pct'], previous)
        self.progress.update(200, 200)
        self.assertLess(self.progress.snapshot()['progress']['phase_pct'], 100)
        self.progress.finish()
        self.assertEqual(self.progress.snapshot()['pct'], 98)
        self.assertEqual(self.progress.snapshot()['progress']['phase_pct'], 100)

    def test_history_is_scoped_persisted_and_estimates_are_labeled(self):
        self.now = 20
        self.progress.finish()
        history = RuntimeHistory(self.path)
        self.assertEqual(history.estimate('model:cpu:aligning', 50), 10)
        self.assertIsNone(history.estimate('model:cuda:aligning', 50))
        progress = ProcessingProgress(history, 'model:cpu', 50, clock=lambda: self.now)
        progress.begin('aligning', 'Aligning…', 55, 98)
        self.now = 25
        snapshot = progress.snapshot()
        self.assertFalse(snapshot['indeterminate'])
        self.assertTrue(snapshot['progress']['estimated'])
        self.assertEqual(snapshot['progress']['eta_seconds'], 5)
        self.now = 40
        self.assertIsNone(progress.snapshot()['progress']['eta_seconds'])
        self.assertLess(progress.snapshot()['progress']['phase_pct'], 100)

    def test_stage_transition_resets_eta_and_changes_units(self):
        self.now = 2
        self.progress.update(10, 100)
        self.progress.begin('aligning_candidates', 'Scoring…', 80, 92, total=40, unit='words')
        snapshot = self.progress.snapshot()
        self.assertEqual(snapshot['progress']['unit'], 'words')
        self.assertEqual(snapshot['progress']['done'], 0)
        self.assertIsNone(snapshot['progress']['speed'])
        self.assertEqual(snapshot['progress']['elapsed_seconds'], 0)

    def test_cancel_does_not_learn_incomplete_stage(self):
        self.now = 10
        self.cancelled = True
        with self.assertRaises(InterruptedError):
            self.progress.finish()
        self.assertIsNone(self.history.estimate('model:cpu:aligning', 100))
        self.assertIsNone(self.progress.snapshot()['progress']['eta_seconds'])

    def test_invalid_samples_and_corrupt_history_are_ignored(self):
        self.progress.update(float('nan'), 10)
        self.progress.update(1, float('inf'))
        self.assertTrue(self.progress.snapshot()['indeterminate'])
        self.path.write_text('broken', encoding='utf-8')
        self.assertIsNone(RuntimeHistory(self.path).estimate('x', 10))


class WorkerProgressTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import app
        self.app = app
        self.task = app.QueueTask('test', 'sync', 1, 'Test', 'Example line')
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.history = RuntimeHistory(Path(self.directory.name) / 'times.json')
        self.updates = []
        async def broadcast():
            self.updates.append(self.task.progress.copy())
        for name, value in [('_RUNTIME_HISTORY', self.history), ('_inference_lock', asyncio.Lock())]:
            patcher = patch.object(app, name, value)
            patcher.start(); self.addCleanup(patcher.stop)
        for name, value in [('_q_broadcast', broadcast), ('_processing_audio_duration', lambda _: 100), ('_get_device', lambda _: 'cpu')]:
            patcher = patch.object(app, name, value)
            patcher.start(); self.addCleanup(patcher.stop)

    async def test_alignment_uses_callback_and_correct_stage_window(self):
        received = {}
        class Model:
            def align(_, path, lyrics, **kwargs):
                received.update(kwargs)
                kwargs['progress_callback'](20, 100)
                time.sleep(.3)
                kwargs['progress_callback'](100, 100)
                return SimpleNamespace(segments=[])
        with patch.object(self.app, 'get_align_model', AsyncMock(return_value=Model())):
            await self.app._whisper_sync_worker(self.task, 'fake.wav', 'Example line')
        self.assertTrue(callable(received['progress_callback']))
        self.assertTrue(all(x['pct'] >= 55 for x in self.updates))
        self.assertTrue(any(x['progress']['done'] == 20 for x in self.updates))

    async def test_streaming_reports_work_even_for_empty_segment(self):
        self.task.type = 'auto'
        def transcribe(path, **kwargs):
            def segments():
                yield SimpleNamespace(text='', start=0, end=20)
                time.sleep(.3)
                yield SimpleNamespace(text='Example line', start=20, end=100, words=[])
            return segments(), SimpleNamespace(duration=100)
        with patch.object(self.app, 'get_verify_model', AsyncMock(return_value=SimpleNamespace(transcribe_original=transcribe))):
            lines = await self.app._faster_stream_transcribe_worker(self.task, 'fake.wav')
        self.assertEqual(lines[0]['line'], 'Example line')
        self.assertEqual(len(self.task.live_lines), 1)
        self.assertTrue(any(x['progress']['done'] == 20 for x in self.updates))
        self.assertTrue(all(x['stage'] == 'transcribing' for x in self.updates))

    async def test_cancel_keeps_inference_lock_until_worker_stops(self):
        started, stopped = threading.Event(), threading.Event()
        def runner(report):
            started.set()
            try:
                while True:
                    time.sleep(.01)
                    report.check_cancel()
            finally:
                stopped.set()
        job = asyncio.create_task(self.app._run_model_job(self.task, 'fake.wav', 'small', runner, phase='aligning', label='Aligning', start=55))
        self.assertTrue(await asyncio.to_thread(started.wait, 2))
        self.assertTrue(self.app._inference_lock.locked())
        job.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(job, 2)
        self.assertTrue(stopped.is_set())
        self.assertFalse(self.app._inference_lock.locked())

    async def test_model_returning_after_cancel_is_not_successful(self):
        def runner(report):
            self.task.cancel_requested = True
            return 'late result'
        with self.assertRaises(asyncio.CancelledError):
            await self.app._run_model_job(self.task, 'fake.wav', 'small', runner, phase='aligning', label='Aligning', start=55)
        self.assertEqual(self.history.samples, {})


if __name__ == '__main__':
    unittest.main()
