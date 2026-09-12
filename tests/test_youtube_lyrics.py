import json
import asyncio
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from youtube_lyrics import extract_youtube_lyrics, parse_captions


class CaptionTests(unittest.TestCase):
    def test_json_segments_markup_and_music(self):
        payload = {"events": [
            {"segs": [{"utf8": "[Music]"}]},
            {"segs": [{"utf8": "<i>Hello</i> "}, {"utf8": "&amp; 世界"}]},
            {"segs": [{"utf8": "♪"}]},
            {"segs": [{"utf8": "I &lt;3 you"}]},
        ]}
        self.assertEqual(parse_captions(json.dumps(payload), "json3"), "Hello & 世界\nI <3 you")

    def test_vtt_rolling_overlap_keeps_later_repeated_chorus(self):
        payload = """WEBVTT

00:00.000 --> 00:03.000
First line

00:02.000 --> 00:05.000 align:start
First line
<00:02.500><c>Second line</c>

00:10.000 --> 00:12.000
First line
"""
        self.assertEqual(parse_captions(payload, "vtt"), "First line\nSecond line\nFirst line")

    def test_adjacent_repetitions_are_kept(self):
        events = [{"tStartMs": start, "dDurationMs": 1000, "segs": [{"utf8": "Go"}]}
                  for start in (0, 1000)]
        self.assertEqual(parse_captions(json.dumps({"events": events}), "json3"), "Go\nGo")


class ExtractionTests(unittest.TestCase):
    def extract(self, info, responses=None, url="https://www.youtube.com/watch?v=test"):
        client = MagicMock()
        client.get.side_effect = responses or [SimpleNamespace(
            text='{"events":[{"segs":[{"utf8":"Lyrics"}]}]}', raise_for_status=lambda: None)]
        with patch("youtube_lyrics.subprocess.run", return_value=SimpleNamespace(stdout=json.dumps(info))) as run, \
                patch("youtube_lyrics.httpx.Client") as factory:
            factory.return_value.__enter__.return_value = client
            result = extract_youtube_lyrics(url)
        return result, client, run

    def track(self, name, ext="json3"):
        return [{"ext": ext, "url": "https://example.com/" + name}]

    def test_manual_original_language_preferred(self):
        result, client, _ = self.extract({"language": "es", "subtitles": {
            "en": self.track("english"), "es": self.track("spanish")},
            "automatic_captions": {"es": self.track("auto")}})
        self.assertEqual(result["lyrics_source"], "youtube")
        self.assertEqual(result["lyrics_language"], "es")
        self.assertEqual(client.get.call_args.args[0], "https://example.com/spanish")

    def test_music_auto_original_and_no_live_chat(self):
        url = "https://music.youtube.com/watch?v=test"
        result, client, run = self.extract({"subtitles": {"live_chat": self.track("chat")},
            "automatic_captions": {"en": self.track("en"), "de-orig": self.track("original")}}, url=url)
        self.assertEqual(result["lyrics_source"], "youtube_auto")
        self.assertEqual(result["lyrics_language"], "de-orig")
        self.assertEqual(run.call_args.args[0][-1], url)
        self.assertEqual(client.get.call_count, 1)

    def test_unavailable_or_empty_captions(self):
        result, client, _ = self.extract({})
        self.assertEqual(result, {})
        client.get.assert_not_called()

    def test_failed_json_falls_back_to_vtt(self):
        result, _, _ = self.extract({"subtitles": {"en": self.track("json") + self.track("vtt", "vtt")}}, [
            httpx.ConnectError("offline"),
            SimpleNamespace(text="WEBVTT\n\n00:00.000 --> 00:01.000\nFallback", raise_for_status=lambda: None)])
        self.assertEqual(result["lyrics"], "Fallback")

    def test_requests_are_bounded(self):
        result, client, _ = self.extract({"automatic_captions": {
            str(index): self.track(str(index)) for index in range(20)}},
            [httpx.ConnectError("offline")] * 4)
        self.assertEqual(result, {})
        self.assertEqual(client.get.call_count, 4)

    def test_timeout_propagates_to_best_effort_caller(self):
        with patch("youtube_lyrics.subprocess.run", side_effect=subprocess.TimeoutExpired("yt-dlp", 45)):
            with self.assertRaises(subprocess.TimeoutExpired):
                extract_youtube_lyrics("https://youtube.com/watch?v=test")


class LocalUrlTests(unittest.TestCase):
    def load(self, url, extraction=None, error=None):
        import app
        meta = {"track_hash": "test", "path": "audio.flac", "processed": {"lyrics": "Saved", "lines": []}}
        with patch.object(app, "_fetch_remote_audio", new=AsyncMock(return_value=("audio.flac", "Audio", url))), \
                patch.object(app, "_register_local_track", return_value=meta), \
                patch.object(app, "extract_youtube_lyrics", return_value=extraction or {}, side_effect=error) as extract:
            result = asyncio.run(app.load_local_url(app.LocalUrlRequest(url=url)))
        return result, extract

    def test_both_youtube_services_receive_defaults_preserving_cache(self):
        for host in ("www.youtube.com", "music.youtube.com"):
            with self.subTest(host=host):
                result, extract = self.load(f"https://{host}/watch?v=test", {"lyrics": "Captions", "lyrics_source": "youtube"})
                self.assertEqual(result["lyrics"], "Captions")
                self.assertEqual(result["processed"]["lyrics"], "Saved")
                extract.assert_called_once()

    def test_caption_failure_and_absence_keep_audio(self):
        for error in (None, RuntimeError("unavailable")):
            result, _ = self.load("https://youtu.be/test", error=error)
            self.assertEqual(result["path"], "audio.flac")
            self.assertIn("lyrics_notice", result)

    def test_other_sources_do_not_extract(self):
        result, extract = self.load("https://soundcloud.com/artist/song")
        extract.assert_not_called()
        self.assertNotIn("lyrics_notice", result)


if __name__ == "__main__":
    unittest.main()
