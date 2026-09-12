"""Best-effort plain lyrics from YouTube's uploaded or automatic subtitles."""

import html
import json
import re
import shutil
import subprocess
import sys

import httpx


def parse_captions(payload: str, extension: str) -> str:
    cues = []
    if extension == "json3":
        for event in json.loads(payload).get("events", []):
            text = "".join(segment.get("utf8", "") for segment in event.get("segs", []))
            cues.append((float(event.get("tStartMs", 0)) / 1000,
                         float(event.get("dDurationMs", 0)) / 1000, text))
    else:
        def seconds(value):
            parts = value.replace(",", ".").split(":")
            return sum(float(part) * 60 ** index for index, part in enumerate(reversed(parts)))

        for block in re.split(r"\n\s*\n", payload.replace("\r\n", "\n")):
            lines = block.splitlines()
            for index, line in enumerate(lines):
                match = re.match(r"([\d:.]+)\s+-->\s+([\d:.]+)", line)
                if match:
                    start, end = map(seconds, match.groups())
                    cues.append((start, end - start, "\n".join(lines[index + 1:])))
                    break

    result = []
    previous = []
    previous_end = -1
    for start, duration, raw in cues:
        # Strip caption markup before unescaping literal text such as &lt;3.
        text = html.unescape(re.sub(r"<[^>]+>", "", raw))
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        lines = [line for line in lines if line and not re.fullmatch(
            r"(?:\[(?:music|applause|laughter)\]|[♪♫\s]+)", line, re.I)]
        if not lines:
            continue
        # Rolling captions repeat the previous line while the next appears.
        # Only remove overlap while cues overlap in time; keep repeated choruses.
        overlap = 0
        if start < previous_end:
            for count in range(1, min(len(previous), len(lines)) + 1):
                if previous[-count:] == lines[:count]:
                    overlap = count
        result.extend(lines[overlap:])
        previous = lines
        previous_end = start + duration
    return "\n".join(result)


def extract_youtube_lyrics(url: str) -> dict:
    """Failures are handled by the caller so audio remains usable."""
    exe = shutil.which("yt-dlp")
    command = [exe] if exe else [sys.executable, "-m", "yt_dlp"]
    proc = subprocess.run(command + [
        "--ignore-config", "--no-playlist", "--skip-download", "--dump-single-json",
        "--no-warnings", "--socket-timeout", "10", "--retries", "0",
        "--extractor-args", "youtube:skip=translated_subs", url,
    ], capture_output=True, text=True, encoding="utf-8", timeout=45, check=True)
    info = json.loads(proc.stdout)
    preferred = str(info.get("language") or "").split("-")[0]

    def language_rank(language):
        base = language.split("-")[0]
        return (0 if preferred and base == preferred else
                1 if language.endswith("-orig") else 2 if base == "en" else 3, language)

    attempts = 0
    with httpx.Client(timeout=10, follow_redirects=True) as client:
        for source in ("subtitles", "automatic_captions"):
            tracks = info.get(source) or {}
            for language in sorted(tracks, key=language_rank):
                if language == "live_chat":
                    continue
                formats = [track for track in tracks[language]
                           if track.get("ext") in ("json3", "vtt") and track.get("url")]
                formats.sort(key=lambda track: track["ext"] != "json3")
                for track in formats:
                    if attempts >= 4:
                        return {}
                    attempts += 1
                    try:
                        response = client.get(track["url"], headers=info.get("http_headers") or {})
                        response.raise_for_status()
                        lyrics = parse_captions(response.text, track["ext"])
                    except (httpx.HTTPError, ValueError, TypeError, KeyError):
                        continue
                    if lyrics:
                        return {"lyrics": lyrics, "lyrics_source": "youtube_auto" if source == "automatic_captions" else "youtube",
                                "lyrics_language": language}
    return {}
