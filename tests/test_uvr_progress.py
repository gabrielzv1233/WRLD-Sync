import unittest
import asyncio
import concurrent.futures
import subprocess
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import app


class UVRProgressSpyTests(unittest.TestCase):
    def test_parses_tqdm_iteration_progress(self):
        spy = app._UVRProgressSpy()

        spy.write("\r 40%|████      | 4/10 [00:08<00:12,  2.00s/it]")

        self.assertEqual(spy.latest(), {
            "pct": 40,
            "done": 4.0,
            "total": 10.0,
            "elapsed": "00:08",
            "eta": "00:12",
            "rate": "2.00s/it",
            "unit": "chunks",
        })

    def test_handles_completed_and_unknown_rate_output(self):
        spy = app._UVRProgressSpy()
        spy.write("\r  0%|          | 0/3 [00:00<?, ?it/s]")
        self.assertEqual(spy.latest()["pct"], 0)
        self.assertEqual(spy.latest()["eta"], "?")

        spy.write("\r100%|██████████| 3/3 [00:09<00:00,  3.00s/it]")
        self.assertEqual(spy.latest()["pct"], 100)
        self.assertEqual(spy.latest()["done"], 3.0)


class UVRProgressWindowTests(unittest.TestCase):
    def test_windows_finish_before_each_tasks_model_loading_stage(self):
        expected = {"sync": 50, "verify": 36, "transcribe": 28, "auto": 26}
        for task_type, end in expected.items():
            with self.subTest(task_type=task_type):
                task = type("Task", (), {"type": task_type})()
                start, actual_end = app._uvr_progress_window(task)
                self.assertLess(start, actual_end)
                self.assertEqual(actual_end, end)


class UVRPreparationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "source.flac"
        self.source.write_bytes(b"audio")
        self.task = SimpleNamespace(
            type="auto",
            preprocess_vocals=True,
            separator_model="uvr-bs-roformer",
            progress={"pct": 100, "msg": "Download complete"},
            cancel_requested=False,
            analysis_source={},
            song_id=1,
            local_hash="",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("app._q_broadcast", new_callable=AsyncMock)
    @patch("app._get_vocal_reference", return_value=None)
    @patch("app._audio_content_hash", return_value="a" * 64)
    async def test_cached_stem_resets_full_download_bar_and_reports_cache_hit(
        self, _audio_hash, _reference, broadcast,
    ):
        config = {"separator_model": "uvr-bs-roformer", "separator_target": "all_vocals"}
        config_hash = app.hashlib.sha256(app.json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        stem = self.root / "stems" / ("a" * 64) / config_hash / "vocals.flac"
        stem.parent.mkdir(parents=True)
        stem.write_bytes(b"vocals")

        with patch.object(app, "_STEM_CACHE_DIR", self.root / "stems"):
            result = await app._prepare_analysis_audio(self.task, str(self.source))

        self.assertEqual(result, str(stem))
        self.assertEqual(self.task.progress["pct"], 26)
        self.assertEqual(self.task.progress["msg"], "Using cached all-vocals stem ✓")
        self.assertGreaterEqual(broadcast.await_count, 2)

    @patch("app._q_broadcast", new_callable=AsyncMock)
    @patch("app._get_vocal_reference", return_value=None)
    @patch("app._audio_content_hash", return_value="b" * 64)
    async def test_fresh_separation_exposes_uvr_progress(
        self, _audio_hash, _reference, broadcast,
    ):
        def fake_separate(_source, _model, destination, spy, cancelled):
            spy.write("\r100%|██████████| 5/5 [00:10<00:00,  2.00s/it]")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"vocals")
            return destination

        with (
            patch.object(app, "_STEM_CACHE_DIR", self.root / "stems"),
            patch("app._separate_vocals", side_effect=fake_separate),
        ):
            result = await app._prepare_analysis_audio(self.task, str(self.source))

        self.assertEqual(Path(result).read_bytes(), b"vocals")
        self.assertEqual(self.task.progress["pct"], 26)
        self.assertEqual(self.task.progress["msg"], "Caching all-vocals stem…")
        self.assertEqual(self.task.progress["uvr"]["pct"], 100)
        self.assertGreaterEqual(broadcast.await_count, 3)

    @patch('app._q_broadcast', new_callable=AsyncMock)
    @patch('app._get_vocal_reference', return_value=None)
    @patch('app._audio_content_hash', return_value='c' * 64)
    async def test_cancel_stops_worker_before_returning(self, *_):
        started, stopped = threading.Event(), threading.Event()
        def separate(source, model, destination, spy, cancelled):
            started.set()
            while not cancelled():
                time.sleep(.01)
            stopped.set()
            raise InterruptedError('Cancelled')
        with patch.object(app, '_STEM_CACHE_DIR', self.root / 'stems'), patch('app._separate_vocals', side_effect=separate):
            future = asyncio.create_task(app._prepare_analysis_audio(self.task, str(self.source)))
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            self.task.cancel_requested = True
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(future, 2)
            self.assertTrue(stopped.is_set())
            self.assertFalse(list(self.root.rglob('vocals.flac')))

    @patch('app._q_broadcast', new_callable=AsyncMock)
    @patch('app._get_vocal_reference', return_value=None)
    @patch('app._audio_content_hash', return_value='d' * 64)
    async def test_coroutine_cancellation_waits_for_worker_cleanup(self, *_):
        started, stopped = threading.Event(), threading.Event()
        def separate(source, model, destination, spy, cancelled):
            started.set()
            while not cancelled():
                time.sleep(.01)
            stopped.set()
            raise InterruptedError('Cancelled')
        with patch.object(app, '_STEM_CACHE_DIR', self.root / 'stems'), patch('app._separate_vocals', side_effect=separate):
            future = asyncio.create_task(app._prepare_analysis_audio(self.task, str(self.source)))
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            future.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(future, 2)
            self.assertTrue(stopped.is_set())


class UVRProcessTests(unittest.TestCase):
    def run_worker(self, script, cancel_during=False):
        original_popen = subprocess.Popen
        children = []
        def popen(command, **kwargs):
            if len(command) > 2 and str(command[2]).endswith('uvr_worker.py'):
                command = [command[0], '-u', '-c', script, *command[3:]]
                process = original_popen(command, **kwargs)
                children.append(process)
                return process
            return original_popen(command, **kwargs)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dest = root / 'cache/vocals.flac'
            spy, cancel = app._UVRProgressSpy(), threading.Event()
            with patch('app.subprocess.Popen', side_effect=popen), concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(app._separate_vocals, root / 'audio.flac', 'uvr-bs-roformer', dest, spy, cancel.is_set)
                if cancel_during:
                    deadline = time.monotonic() + 5
                    while spy.latest() is None and time.monotonic() < deadline:
                        time.sleep(.02)
                    self.assertIsNotNone(spy.latest(), 'Worker did not start')
                    started = time.monotonic()
                    cancel.set()
                    with self.assertRaises(InterruptedError):
                        future.result(timeout=5)
                    self.assertLess(time.monotonic() - started, 5)
                    self.assertFalse(dest.exists())
                else:
                    self.assertEqual(future.result(timeout=5), dest)
                    self.assertEqual(dest.read_bytes(), b'complete-stem')
                self.assertIsNotNone(children[0].poll())
                self.assertFalse(list(dest.parent.glob('.uvr-*')))

    def test_cancel_interrupts_native_work_and_cleans_partial_stem(self):
        self.run_worker('''
import sys, time
from pathlib import Path
out = Path(sys.argv[sys.argv.index('--output-dir')+1])
(out/'partial.flac').write_bytes(b'incomplete')
print('\\r 10%|x         | 1/10 [00:01<00:09, 1.00s/it]', flush=True)
time.sleep(60)
''', cancel_during=True)

    def test_success_publishes_only_finished_stem(self):
        self.run_worker('''
import sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index('--output-dir')+1])
stem = out/'test_vocals.flac'
stem.write_bytes(b'complete-stem')
(out/'result.txt').write_text(str(stem.resolve()), encoding='utf-8')
''')


if __name__ == "__main__":
    unittest.main()
