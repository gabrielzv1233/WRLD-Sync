import unittest
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
        def fake_separate(_source, _model, destination, spy):
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


if __name__ == "__main__":
    unittest.main()
