from dataclasses import dataclass, field as dc_field
from contextlib import asynccontextmanager
import xml.etree.ElementTree as ET
import queue as thread_queue
import urllib.parse
import subprocess
import mimetypes
import threading
import tempfile
import asyncio
import hashlib
import pathlib
import sqlite3
import shutil
import signal
import json
import time
import uuid
import sys
import gc
import io
import os
import re

# ---------------------------------------------------------------------------
# Windows: shut down cleanly when the console window is closed
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    import ctypes
    import ctypes.wintypes

    _CTRL_CLOSE_EVENT = 2

    @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.DWORD)
    def _win_ctrl_handler(ctrl_type):
        if ctrl_type == _CTRL_CLOSE_EVENT:
            # Send SIGINT to ourselves — uvicorn catches it and shuts down gracefully
            os.kill(os.getpid(), signal.SIGINT)
            # Block briefly so uvicorn has time to start shutdown before Windows
            # forcibly terminates the process (~5 s window)
            threading.Event().wait(4)
        return False  # pass event to next handler

    ctypes.windll.kernel32.SetConsoleCtrlHandler(_win_ctrl_handler, True)

import httpx
import stable_whisper
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from model_manager import DownloadPaused, MODEL_SPECS, catalog_status, ensure_model as ensure_managed_model, get_spec as get_managed_spec, is_installed as managed_model_installed, remove_model as remove_managed_model
from model_catalog import get_catalog_model, load_catalog, model_map
from hubert_runtime import HubertDownloadPaused, HubertFAEngine, ensure_hubertfa, get_hubert_install, hubert_installed, remove_hubertfa

BASE = "https://juicewrldapi.com/juicewrld"
CONSOLE = Console(highlight=False)

# Faster-Whisper/CTranslate2 needs CUDA 12 cuBLAS + cuDNN on Windows. A CUDA
# PyTorch wheel already carries those DLLs under torch/lib, so expose that
# directory to the process before CTranslate2 tries to load a CUDA model. This
# avoids making users separately install a full CUDA toolkit when the matching
# runtime DLLs are already present in the project's venv.
_CUDA_DLL_DIR_HANDLE = None
if sys.platform == "win32":
    try:
        import torch as _bootstrap_torch
        _torch_lib = pathlib.Path(_bootstrap_torch.__file__).resolve().parent / "lib"
        _has_cuda_runtime_dlls = (
            any(_torch_lib.glob("cublas64_*.dll"))
            and any(_torch_lib.glob("cudnn64_*.dll"))
        )
        if _has_cuda_runtime_dlls:
            os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                _CUDA_DLL_DIR_HANDLE = os.add_dll_directory(str(_torch_lib))
    except Exception:
        pass


def _display_name(value: str) -> str:
    return urllib.parse.unquote(str(value or "")).replace("\\", "/").rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# tqdm progress spy — captures stable_whisper alignment/transcription progress
# ---------------------------------------------------------------------------
_TQDM_RE = re.compile(
    r'([A-Za-z][\w ]*):\s*(\d+)%'          # label + percent
    r'.*?([\d.]+)/([\d.]+)'                  # done / total (seconds)
    r'.*?\[(\d+:\d+)<(\d+:\d+),\s*([\d.]+)s/sec\]'  # [elapsed<eta, speed]
)


class _ProgressSpy:
    """Thread-safe stderr capturer that parses tqdm progress lines."""
    def __init__(self):
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self.silent = False   # set True to stop forwarding tqdm to terminal

    def write(self, s: str) -> int:
        for chunk in re.split(r'[\r\n]', s):
            chunk = chunk.strip()
            if not chunk:
                continue
            m = _TQDM_RE.search(chunk)
            if m:
                with self._lock:
                    self._latest = {
                        'label': m.group(1).strip(),
                        'pct':   int(m.group(2)),
                        'done':  float(m.group(3)),
                        'total': float(m.group(4)),
                        'elapsed': m.group(5),
                        'eta':   m.group(6),
                        'speed': float(m.group(7)),
                    }
        return len(s)

    def flush(self): pass
    def isatty(self) -> bool: return False
    def fileno(self): raise io.UnsupportedOperation("fileno")

    def latest(self) -> dict | None:
        with self._lock:
            return self._latest


class _TeeStderr:
    """Tees stderr to a _ProgressSpy and the original stderr."""
    def __init__(self, spy: _ProgressSpy, orig):
        self._spy = spy
        self._orig = orig

    def write(self, s: str) -> int:
        self._spy.write(s)
        if not self._spy.silent:
            try:
                self._orig.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass

    def isatty(self) -> bool: return False
    def fileno(self): raise io.UnsupportedOperation("fileno")


# ---------------------------------------------------------------------------
# Upload directory — local files submitted for whisper processing
# ---------------------------------------------------------------------------
UPLOAD_DIR = pathlib.Path(tempfile.gettempdir()) / "jw_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Audio cache — last downloaded file, reused by /api/sync and /api/verify
# ---------------------------------------------------------------------------
_audio_cache: dict | None = None   # {"path": str, "file": str}
_audio_cache_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Local processed-lyrics database
# ---------------------------------------------------------------------------
_DATA_DIR = pathlib.Path(__file__).parent / "data"
DB_PATH = _DATA_DIR / "wrld_sync.sqlite3"
_CACHE_DIR = pathlib.Path(__file__).parent / "cache"
_STEM_CACHE_DIR = _CACHE_DIR / "stems"
_VOCAL_REFERENCE_DIR = _CACHE_DIR / "vocal-references"
_UVR_MODEL_DIR = pathlib.Path(__file__).parent / "models" / "uvr"
for _dir in (_CACHE_DIR, _STEM_CACHE_DIR, _VOCAL_REFERENCE_DIR, _UVR_MODEL_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

_ADVANCED_DEFAULTS = {
    "preprocess_vocals": False,
    "separator_model": "uvr-bs-roformer",
    "separator_target": "all_vocals",
    "yt_dlp_enabled": True,
    "yt_dlp_quality": "high",
}
_STATE_DIR = pathlib.Path(os.getenv("WRLD_SYNC_STATE_DIR", pathlib.Path(__file__).parent))
_STATE_DIR.mkdir(parents=True, exist_ok=True)
_ADVANCED_PATH = _STATE_DIR / ".advanced_settings.json"

def _load_advanced_settings() -> dict:
    values = dict(_ADVANCED_DEFAULTS)
    try:
        raw = json.loads(_ADVANCED_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            values.update({k: raw[k] for k in _ADVANCED_DEFAULTS if k in raw})
    except (OSError, json.JSONDecodeError):
        pass
    values["preprocess_vocals"] = bool(values["preprocess_vocals"])
    values["yt_dlp_enabled"] = bool(values["yt_dlp_enabled"])
    if values["separator_model"] not in {m["id"] for m in load_catalog()["models"] if "separation" in m.get("tasks", [])}:
        values["separator_model"] = _ADVANCED_DEFAULTS["separator_model"]
    if values["separator_target"] != "all_vocals":
        values["separator_target"] = "all_vocals"
    if values["yt_dlp_quality"] not in ("high", "medium", "small"):
        values["yt_dlp_quality"] = "high"
    return values

def _save_advanced_settings(values: dict) -> None:
    _ADVANCED_PATH.write_text(json.dumps(values, indent=2), encoding="utf-8")

ADVANCED_SETTINGS = _load_advanced_settings()


def _db_connect() -> sqlite3.Connection:
    _DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _db_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_lyrics (
                song_id INTEGER PRIMARY KEY,
                song_name TEXT NOT NULL DEFAULT '',
                source_type TEXT NOT NULL DEFAULT '',
                lyrics_text TEXT NOT NULL DEFAULT '',
                synced_lines_json TEXT NOT NULL,
                ttml TEXT NOT NULL DEFAULT '',
                settings_json TEXT NOT NULL DEFAULT '{}',
                processed_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_processed_lyrics_updated ON processed_lyrics(updated_at DESC)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS local_tracks (
                track_hash TEXT PRIMARY KEY,
                original_name TEXT NOT NULL DEFAULT '',
                source_url TEXT NOT NULL DEFAULT '',
                file_path TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                artist TEXT NOT NULL DEFAULT '',
                duration REAL NOT NULL DEFAULT 0,
                cover_mime TEXT NOT NULL DEFAULT '',
                cover_blob BLOB,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_local_tracks_updated ON local_tracks(updated_at DESC)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS local_processed_lyrics (
                track_hash TEXT PRIMARY KEY,
                track_name TEXT NOT NULL DEFAULT '',
                source_type TEXT NOT NULL DEFAULT '',
                lyrics_text TEXT NOT NULL DEFAULT '',
                synced_lines_json TEXT NOT NULL,
                ttml TEXT NOT NULL DEFAULT '',
                settings_json TEXT NOT NULL DEFAULT '{}',
                processed_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(track_hash) REFERENCES local_tracks(track_hash) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_local_processed_updated ON local_processed_lyrics(updated_at DESC)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vocal_references (
                owner_key TEXT PRIMARY KEY,
                source_name TEXT NOT NULL DEFAULT '',
                source_url TEXT NOT NULL DEFAULT '',
                cached_path TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )


def _get_processed_lyrics(song_id: int) -> dict | None:
    if song_id <= 0:
        return None
    try:
        with _db_connect() as conn:
            row = conn.execute(
                "SELECT * FROM processed_lyrics WHERE song_id = ?", (int(song_id),)
            ).fetchone()
    except sqlite3.Error as exc:
        CONSOLE.print(f"[yellow]Could not read local lyrics cache: {exc}[/yellow]")
        return None
    if row is None:
        return None
    try:
        lines = json.loads(row["synced_lines_json"])
    except Exception:
        lines = []
    try:
        settings = json.loads(row["settings_json"])
    except Exception:
        settings = {}
    return {
        "song_id": row["song_id"],
        "song_name": row["song_name"],
        "source_type": row["source_type"],
        "lyrics": row["lyrics_text"],
        "lines": lines,
        "ttml": row["ttml"],
        "settings": settings,
        "processed_at": row["processed_at"],
        "updated_at": row["updated_at"],
    }


def _get_local_processed(track_hash: str) -> dict | None:
    track_hash = str(track_hash or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", track_hash):
        return None
    try:
        with _db_connect() as conn:
            row = conn.execute(
                "SELECT * FROM local_processed_lyrics WHERE track_hash = ?", (track_hash,)
            ).fetchone()
    except sqlite3.Error as exc:
        CONSOLE.print(f"[yellow]Could not read local-file lyrics cache: {exc}[/yellow]")
        return None
    if row is None:
        return None
    try:
        lines = json.loads(row["synced_lines_json"])
    except Exception:
        lines = []
    try:
        settings = json.loads(row["settings_json"])
    except Exception:
        settings = {}
    return {
        "track_hash": row["track_hash"],
        "track_name": row["track_name"],
        "source_type": row["source_type"],
        "lyrics": row["lyrics_text"],
        "lines": lines,
        "ttml": row["ttml"],
        "settings": settings,
        "processed_at": row["processed_at"],
        "updated_at": row["updated_at"],
    }


def _get_local_track(track_hash: str, include_cover: bool = False) -> dict | None:
    track_hash = str(track_hash or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", track_hash):
        return None
    fields = "*" if include_cover else "track_hash,original_name,source_url,file_path,title,artist,duration,cover_mime,created_at,updated_at"
    try:
        with _db_connect() as conn:
            row = conn.execute(f"SELECT {fields} FROM local_tracks WHERE track_hash = ?", (track_hash,)).fetchone()
    except sqlite3.Error as exc:
        CONSOLE.print(f"[yellow]Could not read local track: {exc}[/yellow]")
        return None
    return dict(row) if row is not None else None



def _owner_key(song_id: int = 0, track_hash: str = "") -> str:
    if int(song_id or 0) > 0:
        return f"catalog:{int(song_id)}"
    track_hash = str(track_hash or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", track_hash):
        return f"local:{track_hash}"
    return ""


def _task_owner_key(task) -> str:
    return _owner_key(getattr(task, "song_id", 0), getattr(task, "local_hash", ""))


def _get_vocal_reference(owner_key: str) -> dict | None:
    if not owner_key:
        return None
    with _db_connect() as conn:
        row = conn.execute("SELECT * FROM vocal_references WHERE owner_key = ?", (owner_key,)).fetchone()
    if row is None:
        return None
    data = dict(row)
    cached = pathlib.Path(data.get("cached_path") or "")
    if not cached.is_file():
        with _db_connect() as conn:
            conn.execute("DELETE FROM vocal_references WHERE owner_key = ?", (owner_key,))
        return None
    return data


def _set_vocal_reference(owner_key: str, cached_path: pathlib.Path, source_name: str, source_url: str = "") -> dict:
    if not owner_key:
        raise ValueError("A catalog song or local track is required for a vocal reference.")
    now = time.time()
    with _db_connect() as conn:
        conn.execute(
            """INSERT INTO vocal_references(owner_key, source_name, source_url, cached_path, created_at, updated_at)
               VALUES(?, ?, ?, ?, ?, ?)
               ON CONFLICT(owner_key) DO UPDATE SET
                 source_name=excluded.source_name, source_url=excluded.source_url,
                 cached_path=excluded.cached_path, updated_at=excluded.updated_at""",
            (owner_key, source_name, source_url, str(cached_path), now, now),
        )
    return _get_vocal_reference(owner_key) or {}


def _delete_vocal_reference(owner_key: str) -> bool:
    row = _get_vocal_reference(owner_key)
    with _db_connect() as conn:
        cur = conn.execute("DELETE FROM vocal_references WHERE owner_key = ?", (owner_key,))
    if row:
        try:
            pathlib.Path(row.get("cached_path") or "").unlink(missing_ok=True)
        except OSError:
            pass
    return cur.rowcount > 0


def _normalize_audio_only(source: pathlib.Path, dest: pathlib.Path) -> pathlib.Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp.flac")
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(source), "-map", "0:a:0",
        "-vn", "-sn", "-dn", "-map_metadata", "-1", "-c:a", "flac", str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Could not decode vocal reference: {proc.stderr.strip() or 'ffmpeg failed'}")
    tmp.replace(dest)
    return dest


def _separator_catalog_item(model_id: str) -> dict:
    item = get_catalog_model(model_id)
    if item.get("source") != "audio-separator":
        raise ValueError(f"{model_id} is not an audio-separator model")
    return item


def _separator_model_installed(model_id: str) -> bool:
    try:
        item = _separator_catalog_item(model_id)
    except ValueError:
        return False
    return (_UVR_MODEL_DIR / str(item.get("asset") or item.get("runtime_id") or "")).is_file()


def _ensure_separator_model(model_id: str) -> pathlib.Path:
    item = _separator_catalog_item(model_id)
    asset = str(item.get("asset") or item.get("runtime_id") or "")
    if not asset:
        raise RuntimeError(f"No separator asset configured for {model_id}")
    target = _UVR_MODEL_DIR / asset
    if target.is_file():
        return target
    exe = shutil.which("audio-separator")
    if exe:
        proc = subprocess.run(
            [exe, "--model_filename", asset, "--model_file_dir", str(_UVR_MODEL_DIR), "--download_model_only"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "audio-separator model download failed")
    else:
        from audio_separator.separator import Separator
        sep = Separator(model_file_dir=str(_UVR_MODEL_DIR), output_dir=str(_CACHE_DIR), output_format="FLAC")
        sep.load_model(model_filename=asset)
        del sep
    if not target.is_file():
        raise RuntimeError(f"audio-separator did not create expected model asset {asset}")
    return target


def _separate_vocals(source: pathlib.Path, model_id: str, cache_path: pathlib.Path) -> pathlib.Path:
    item = _separator_catalog_item(model_id)
    asset = str(item.get("asset") or item.get("runtime_id") or "")
    _ensure_separator_model(model_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wrld-uvr-") as temp_dir:
        from audio_separator.separator import Separator
        sep = Separator(
            model_file_dir=str(_UVR_MODEL_DIR), output_dir=temp_dir,
            output_format="FLAC", output_single_stem="Vocals",
        )
        sep.load_model(model_filename=asset)
        outputs = sep.separate(str(source))
        del sep
        candidates = [pathlib.Path(x) for x in (outputs or [])]
        candidates = [x if x.is_absolute() else pathlib.Path(temp_dir) / x for x in candidates]
        vocal = next((x for x in candidates if x.is_file() and "vocal" in x.name.lower()), None)
        if vocal is None:
            vocal = next((x for x in pathlib.Path(temp_dir).glob("*.flac") if x.is_file()), None)
        if vocal is None:
            raise RuntimeError("UVR separation completed without a vocals stem")
        shutil.copy2(vocal, cache_path)
    return cache_path


async def _prepare_analysis_audio(task, source_path: str) -> str:
    owner = _task_owner_key(task)
    reference = _get_vocal_reference(owner)
    if reference:
        task.analysis_source = {
            "type": "manual_vocal_reference",
            "label": "Manual vocal reference",
            "name": reference.get("source_name") or pathlib.Path(reference["cached_path"]).name,
        }
        return str(reference["cached_path"])

    if not bool(getattr(task, "preprocess_vocals", False)):
        task.analysis_source = {"type": "original_mix", "label": "Original mix"}
        return source_path

    separator_model = getattr(task, "separator_model", "") or ADVANCED_SETTINGS["separator_model"]
    task.progress = {**task.progress, "stage": "separating", "step": "preprocessing", "msg": "Preparing cached all-vocals stem…"}
    await _q_broadcast()
    audio_hash = await asyncio.to_thread(_audio_content_hash, pathlib.Path(source_path))
    config = {"separator_model": separator_model, "separator_target": "all_vocals"}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    stem_dir = _STEM_CACHE_DIR / audio_hash / config_hash
    stem_path = stem_dir / "vocals.flac"
    metadata_path = stem_dir / "metadata.json"
    if not stem_path.is_file():
        await asyncio.to_thread(_separate_vocals, pathlib.Path(source_path), separator_model, stem_path)
        metadata_path.write_text(json.dumps({"audio_hash": audio_hash, **config}, indent=2), encoding="utf-8")
    task.analysis_source = {
        "type": "uvr",
        "label": f"UVR · {get_catalog_model(separator_model).get('label', separator_model)}",
        "separator_model": separator_model,
        "audio_hash": audio_hash,
        "cache_path": str(stem_path),
    }
    return str(stem_path)


def _store_processed_lyrics(task) -> None:
    """Persist a successful result to the catalog-song or local-track table."""
    song_id = int(getattr(task, "song_id", 0) or 0)
    local_hash = str(getattr(task, "local_hash", "") or "").strip().lower()
    if song_id <= 0 and not re.fullmatch(r"[0-9a-f]{64}", local_hash):
        return
    result = getattr(task, "result", None) or {}
    lines = result.get("lines") or []
    if not lines:
        return

    ttml = ""
    used_word_timing = False
    try:
        ttml, used_word_timing = _build_ttml(
            lines,
            word_timing=bool(getattr(task, "word_timing", True)),
            detect_interludes=bool(getattr(task, "detect_interludes", True)),
            inline_parenthetical_background=bool(getattr(task, "inline_parenthetical_background", True)),
            interlude_threshold=float(getattr(task, "interlude_threshold", 2.0)),
        )
    except Exception as exc:
        CONSOLE.print(f"[yellow]Could not build cached TTML for song {task.song_id}: {exc}[/yellow]")

    fast_mode = False
    settings = {
        "task_type": getattr(task, "type", ""),
        "engine": ENGINE_PREF,
        "device_pref": DEVICE_PREF,
        "resolved_device": _get_device("torch") if ((getattr(task, "sync_model", "") or ALIGN_MODEL_SIZE) in MODEL_SPECS or (getattr(task, "transcribe_model", "") or VERIFY_MODEL_SIZE) in MODEL_SPECS) else _get_device(ENGINE_PREF),
        "align_model": getattr(task, "sync_model", "") or ALIGN_MODEL_SIZE,
        "verify_model": getattr(task, "transcribe_model", "") or VERIFY_MODEL_SIZE,
        "sync_model": getattr(task, "sync_model", "") or ALIGN_MODEL_SIZE,
        "transcribe_model": getattr(task, "transcribe_model", "") or VERIFY_MODEL_SIZE,
        "auto_mode": result.get("auto_mode", ""),
        "fast_auto": False,  # legacy field retained for older cache readers
        "word_timing_requested": bool(getattr(task, "word_timing", True)),
        "word_timing_used": used_word_timing,
        "detect_interludes": bool(getattr(task, "detect_interludes", True)),
        "inline_parenthetical_background": bool(getattr(task, "inline_parenthetical_background", True)),
        "allow_overlapping_lyrics": bool(getattr(task, "allow_overlapping_lyrics", False)),
        "interlude_threshold": float(getattr(task, "interlude_threshold", 2.0)),
        "preprocess_vocals": bool(getattr(task, "preprocess_vocals", False)),
        "separator_model": str(getattr(task, "separator_model", "") or ""),
        "separator_target": str(getattr(task, "separator_target", "") or "all_vocals"),
        "analysis_source": dict(getattr(task, "analysis_source", {}) or {}),
        "background_vocals": "parenthetical-inline-toggle",
        "alignment": {
            "original_split": True,
            "fast_mode": fast_mode,
            "token_step": 200 if fast_mode else 100,
            "nonspeech_skip": 2.0 if fast_mode else 5.0,
            "suppress_silence": True,
            "suppress_word_ts": True,
        },
    }
    lyrics_text = str(result.get("text") or getattr(task, "lyrics", "") or "")
    now = time.time()
    with _db_connect() as conn:
        if song_id > 0:
            conn.execute(
                """
                INSERT INTO processed_lyrics (
                    song_id, song_name, source_type, lyrics_text, synced_lines_json,
                    ttml, settings_json, processed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(song_id) DO UPDATE SET
                    song_name = excluded.song_name,
                    source_type = excluded.source_type,
                    lyrics_text = excluded.lyrics_text,
                    synced_lines_json = excluded.synced_lines_json,
                    ttml = excluded.ttml,
                    settings_json = excluded.settings_json,
                    processed_at = excluded.processed_at,
                    updated_at = excluded.updated_at
                """,
                (
                    song_id, str(task.song_name or ""), str(task.type or ""),
                    lyrics_text, json.dumps(lines, ensure_ascii=False), ttml,
                    json.dumps(settings, ensure_ascii=False), now, now,
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO local_processed_lyrics (
                    track_hash, track_name, source_type, lyrics_text, synced_lines_json,
                    ttml, settings_json, processed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(track_hash) DO UPDATE SET
                    track_name = excluded.track_name,
                    source_type = excluded.source_type,
                    lyrics_text = excluded.lyrics_text,
                    synced_lines_json = excluded.synced_lines_json,
                    ttml = excluded.ttml,
                    settings_json = excluded.settings_json,
                    processed_at = excluded.processed_at,
                    updated_at = excluded.updated_at
                """,
                (
                    local_hash, str(task.song_name or ""), str(task.type or ""),
                    lyrics_text, json.dumps(lines, ensure_ascii=False), ttml,
                    json.dumps(settings, ensure_ascii=False), now, now,
                ),
            )


async def ensure_audio(song_path: str):
    """Async generator: yields SSE dicts during download, then {"_path": str} as last item."""
    global _audio_cache

    async with _audio_cache_lock:
        if _audio_cache and _audio_cache["path"] == song_path:
            yield {"stage": "downloading", "pct": 100, "msg": "Audio already cached ✓"}
            yield {"_path": _audio_cache["file"]}
            return

        # New song — evict old cached file
        if _audio_cache:
            try:
                os.unlink(_audio_cache["file"])
            except OSError:
                pass
            _audio_cache = None

        yield {"stage": "downloading", "pct": 0, "msg": "Downloading audio…"}
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        downloaded, last_pct = 0, -1
        async with httpx.AsyncClient(timeout=180) as client:
            async with client.stream(
                "GET", BASE + "/files/download/", params={"path": song_path}
            ) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(tmp_path, "wb") as f:
                    async for chunk in r.aiter_bytes(65_536):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = min(99, int(downloaded / total * 100))
                            if pct >= last_pct + 5:
                                last_pct = pct
                                yield {"stage": "downloading", "pct": pct,
                                       "msg": f"Downloading… {pct}%"}

        _audio_cache = {"path": song_path, "file": tmp_path}
        yield {"stage": "downloading", "pct": 100, "msg": "Download complete"}
        yield {"_path": tmp_path}


# ---------------------------------------------------------------------------
# Model loading — separate align (sync) and verify models, lazy-loaded
# ---------------------------------------------------------------------------
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
WHISPER_ENGINES = ["faster", "torch"]
SYNC_MODELS = WHISPER_MODELS + ["qwen3-forced-aligner-0.6b", "hubert-fa-combined"]
TRANSCRIBE_MODELS = WHISPER_MODELS + ["qwen3-asr-0.6b", "qwen3-asr-1.7b", "parakeet-tdt-0.6b-v3"]
_PREF_DIR = _STATE_DIR

def _read_pref(filename: str, choices: list, default: str) -> str:
    try:
        val = (_PREF_DIR / filename).read_text().strip()
        if val in choices:
            return val
    except OSError:
        pass
    return default

def _write_pref(filename: str, value: str) -> None:
    try:
        (_PREF_DIR / filename).write_text(value)
    except OSError:
        pass

DEVICE_PREF: str = _read_pref(".device_pref", ["auto", "cpu", "cuda"], "auto")
ENGINE_PREF: str = _read_pref(".engine_pref", WHISPER_ENGINES, "faster")

def _torch_cuda_usable() -> bool:
    try:
        import torch
        if not torch.cuda.is_available(): return False
        major, minor = torch.cuda.get_device_capability()
        cap = major + minor / 10
        arch_caps = []
        for arch in torch.cuda.get_arch_list():
            digits = "".join(ch for ch in arch if ch.isdigit())
            if len(digits) >= 2: arch_caps.append(int(digits[:-1]) + int(digits[-1]) / 10)
        return not arch_caps or cap >= min(arch_caps)
    except Exception:
        return False

def _faster_cuda_usable() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False

def _cuda_usable(engine: str | None = None) -> bool:
    engine = engine or ENGINE_PREF
    return _faster_cuda_usable() if engine == "faster" else _torch_cuda_usable()

_device_fallback_warned: set[str] = set()
def _get_device(engine: str | None = None) -> str:
    engine = engine or ENGINE_PREF
    if DEVICE_PREF == "cpu": return "cpu"
    if _cuda_usable(engine): return "cuda"
    if DEVICE_PREF == "cuda" and engine not in _device_fallback_warned:
        _device_fallback_warned.add(engine)
        CONSOLE.print(f"[yellow]CUDA was requested for {engine}, but that backend cannot use it. Falling back to CPU.[/yellow]")
    return "cpu"

def _release_models() -> None:
    global _align_model, _verify_model, _align_model_id, _verify_model_id, _qwen_aligner_runtime
    _align_model = None; _verify_model = None
    _qwen_aligner_runtime = None
    _align_model_id = None; _verify_model_id = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception: pass

def _load_whisper_model(size: str):
    if ENGINE_PREF == "faster":
        faster_device = _get_device("faster")
        try:
            compute_type = "float16" if faster_device == "cuda" else "int8"
            return stable_whisper.load_faster_whisper(size, device=faster_device, compute_type=compute_type)
        except Exception as exc:
            CONSOLE.print(f"[yellow]Faster-Whisper could not initialize on {faster_device.upper()}: {exc}[/yellow]")
            CONSOLE.print("[yellow]Falling back to PyTorch Whisper for this model load.[/yellow]")
    return stable_whisper.load_model(size, device=_get_device("torch"))

def _managed_torch_args() -> tuple[str, object]:
    import torch
    device = "cuda:0" if _get_device("torch") == "cuda" else "cpu"
    # RTX 40/50 series handles BF16 well, and it avoids the numerical edge cases
    # some newer speech models can hit in FP16. CPU stays FP32.
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return device, dtype

class _QwenAlignerRuntime:
    def __init__(self, model, processor):
        self.model, self.processor = model, processor

class _QwenASRRuntime:
    def __init__(self, model, processor, aligner):
        self.model, self.processor, self.aligner = model, processor, aligner

_qwen_aligner_runtime = None
_qwen_aligner_load_lock = threading.Lock()

def _load_qwen_aligner(model_id: str):
    global _qwen_aligner_runtime
    if not managed_model_installed(model_id):
        raise RuntimeError(f"{get_managed_spec(model_id).label} is not installed yet. Wait for its model-download queue task.")
    with _qwen_aligner_load_lock:
        if _qwen_aligner_runtime is not None:
            return _qwen_aligner_runtime
        import torch
        from transformers import AutoModelForTokenClassification, AutoProcessor
        device, dtype = _managed_torch_args()
        path = str(get_managed_spec(model_id).path)
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(
            path, local_files_only=True, dtype=dtype,
        ).to(device).eval()
        _qwen_aligner_runtime = _QwenAlignerRuntime(model, processor)
        return _qwen_aligner_runtime

def _load_qwen_asr(model_id: str):
    if not managed_model_installed(model_id):
        raise RuntimeError(f"{get_managed_spec(model_id).label} is not installed yet. Wait for its model-download queue task.")
    align_id = "qwen3-forced-aligner-0.6b"
    if not managed_model_installed(align_id):
        raise RuntimeError("Qwen3 Forced Aligner is required for timestamped Qwen transcription and is not installed yet.")
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    device, dtype = _managed_torch_args()
    path = str(get_managed_spec(model_id).path)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        path, local_files_only=True, dtype=dtype,
    ).to(device).eval()
    # Share one aligner instance between Sync and Qwen ASR so a 1.8 GB model
    # is not needlessly loaded into VRAM twice.
    aligner = _load_qwen_aligner(align_id)
    return _QwenASRRuntime(model, processor, aligner)

class _ParakeetRuntime:
    def __init__(self, model, processor): self.model, self.processor = model, processor

def _load_parakeet(model_id: str):
    if not managed_model_installed(model_id):
        raise RuntimeError(f"{get_managed_spec(model_id).label} is not installed yet. Wait for its model-download queue task.")
    import torch
    from transformers import AutoProcessor, ParakeetForTDT
    path = str(get_managed_spec(model_id).path)
    device = "cuda" if _get_device("torch") == "cuda" else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = ParakeetForTDT.from_pretrained(path, local_files_only=True, dtype=dtype).to(device).eval()
    return _ParakeetRuntime(model, processor)


def _load_hubertfa():
    if not hubert_installed():
        raise RuntimeError("HuBERT FA combined is not installed yet. Wait for its model-download queue task.")
    return HubertFAEngine()


def _runtime_model_label(model_id: str) -> str:
    try:
        return str(get_catalog_model(model_id).get("label") or model_id)
    except ValueError:
        return f"Whisper {model_id}"

# Existing variable names are retained for cache/backward compatibility.
ALIGN_MODEL_SIZE: str = _read_pref(".model_pref_align", SYNC_MODELS, _read_pref(".model_pref", SYNC_MODELS, os.getenv("WHISPER_MODEL", "small")))
VERIFY_MODEL_SIZE: str = _read_pref(".model_pref_verify", TRANSCRIBE_MODELS, "base")
_align_model = None
_verify_model = None
_align_model_id: str | None = None
_verify_model_id: str | None = None
_model_lock = asyncio.Lock()
_inference_lock = asyncio.Lock()

async def get_align_model(model_id: str | None = None):
    global _align_model, _align_model_id
    model_id = model_id or ALIGN_MODEL_SIZE
    if _align_model is None or _align_model_id != model_id:
        async with _model_lock:
            if _align_model is None or _align_model_id != model_id:
                label = _runtime_model_label(model_id)
                CONSOLE.print(f"[cyan]→[/cyan] Loading sync model [bold]{label}[/bold]")
                loop = asyncio.get_running_loop()
                if model_id == "qwen3-forced-aligner-0.6b":
                    loaded = await loop.run_in_executor(None, lambda: _load_qwen_aligner(model_id))
                elif model_id == "hubert-fa-combined":
                    loaded = await loop.run_in_executor(None, _load_hubertfa)
                else:
                    loaded = await loop.run_in_executor(None, lambda: _load_whisper_model(model_id))
                _align_model, _align_model_id = loaded, model_id
                CONSOLE.print("[green]✓[/green] Sync model ready")
    return _align_model

async def get_verify_model(model_id: str | None = None):
    global _verify_model, _verify_model_id
    model_id = model_id or VERIFY_MODEL_SIZE
    if _verify_model is None or _verify_model_id != model_id:
        async with _model_lock:
            if _verify_model is None or _verify_model_id != model_id:
                label = _runtime_model_label(model_id)
                CONSOLE.print(f"[cyan]→[/cyan] Loading transcription model [bold]{label}[/bold]")
                loop = asyncio.get_running_loop()
                if model_id.startswith("qwen3-asr-"):
                    loaded = await loop.run_in_executor(None, lambda: _load_qwen_asr(model_id))
                elif model_id.startswith("parakeet-"):
                    loaded = await loop.run_in_executor(None, lambda: _load_parakeet(model_id))
                else:
                    loaded = await loop.run_in_executor(None, lambda: _load_whisper_model(model_id))
                _verify_model, _verify_model_id = loaded, model_id
                CONSOLE.print("[green]✓[/green] Transcription model ready")
    return _verify_model

# Backward-compat alias used by legacy SSE endpoints
async def get_model():
    return await get_align_model()


# ---------------------------------------------------------------------------
# Task Queue
# ---------------------------------------------------------------------------

@dataclass
class QueueTask:
    id: str
    type: str           # "sync" | "verify" | "auto"
    song_id: int
    song_name: str
    lyrics: str         # may be empty
    local_path: str = ""  # non-empty → use this file instead of fetching from API
    local_hash: str = ""  # SHA-256 of decoded audio for local files
    fast_auto: bool = False  # legacy compatibility only; Auto is always a full transcription
    word_timing: bool = True
    detect_interludes: bool = True
    inline_parenthetical_background: bool = True
    allow_overlapping_lyrics: bool = False
    interlude_threshold: float = 2.0
    preprocess_vocals: bool = False
    separator_model: str = "uvr-bs-roformer"
    separator_target: str = "all_vocals"
    analysis_source: dict = dc_field(default_factory=dict)
    status: str = "pending"   # pending|running|done|error|cancelled
    progress: dict = dc_field(default_factory=dict)
    error: str = ""
    created_at: float = dc_field(default_factory=time.time)
    cancel_requested: bool = False
    pause_requested: bool = False
    result: dict | None = None
    live_lines: list[dict] = dc_field(default_factory=list)
    model_id: str = ""
    sync_model: str = ""
    transcribe_model: str = ""

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "type": self.type,
            "song_id": self.song_id,
            "song_name": self.song_name,
            "local_hash": self.local_hash,
            "status": self.status,
            "progress": self.progress,
            "error": self.error,
            "created_at": self.created_at,
            "pause_requested": self.pause_requested,
            "model_id": self.model_id,
            "settings": {
                "fast_auto": self.fast_auto,
                "word_timing": self.word_timing,
                "detect_interludes": self.detect_interludes,
                "inline_parenthetical_background": self.inline_parenthetical_background,
                "allow_overlapping_lyrics": self.allow_overlapping_lyrics,
                "interlude_threshold": self.interlude_threshold,
                "preprocess_vocals": self.preprocess_vocals,
                "separator_model": self.separator_model,
                "separator_target": self.separator_target,
                "analysis_source": self.analysis_source,
                "sync_model": self.sync_model or ALIGN_MODEL_SIZE,
                "transcribe_model": self.transcribe_model or VERIFY_MODEL_SIZE,
            },
        }
        if self.result and self.status in ("done", "error"):
            d["result"] = self.result
        if self.status in ("running", "cancelling") and self.live_lines:
            d["live_lines"] = self.live_lines
        return d


_task_queue: asyncio.Queue = asyncio.Queue()
_tasks: dict[str, QueueTask] = {}          # id → task (all states)
_queued_task_ids: set[str] = set()         # task IDs with a live asyncio.Queue ticket
_active_task: QueueTask | None = None

# SSE broadcast via asyncio.Condition — all stream generators wait on this
_q_cond: asyncio.Condition | None = None   # initialised in lifespan
_q_state_json: str = '{"active":null,"pending":[],"history":[]}'



async def _queue_put_once(task: QueueTask) -> bool:
    """Put a task into the processor queue only when it has no live queue ticket."""
    if task.id in _queued_task_ids:
        return False
    _queued_task_ids.add(task.id)
    await _task_queue.put(task)
    return True


def _all_catalog_status() -> list[dict]:
    managed = {item["id"]: item for item in catalog_status()}
    out: list[dict] = []
    for raw in load_catalog()["models"]:
        item = dict(raw)
        model_id = str(item["id"])
        source = item.get("source")
        if model_id in managed:
            item.update(managed[model_id])
        elif source == "audio-separator":
            item.update({"installed": _separator_model_installed(model_id), "managed": True, "path": str(_UVR_MODEL_DIR / str(item.get("asset") or ""))})
        elif source == "whisper":
            # Whisper/stable-ts manages its own download cache on first load.
            item.update({"installed": None, "managed": False, "install_state": "runtime-managed"})
        elif source == "hubertfa_release":
            item.update({
                "installed": hubert_installed(),
                "managed": True,
                "path": str(pathlib.Path(__file__).parent / "models" / model_id),
                "install_state": "installed" if hubert_installed() else "missing",
            })
        out.append(item)
    return out


def _catalog_model_installed(model_id: str) -> bool | None:
    if model_id in MODEL_SPECS:
        return managed_model_installed(model_id)
    item = get_catalog_model(model_id)
    if item.get("source") == "audio-separator":
        return _separator_model_installed(model_id)
    if item.get("source") == "whisper":
        return None
    if item.get("source") == "hubertfa_release":
        return hubert_installed()
    return False


def _task_model_requirements(task_type: str, sync_model: str | None = None, transcribe_model: str | None = None) -> list[str]:
    sync_model = sync_model or ALIGN_MODEL_SIZE
    transcribe_model = transcribe_model or VERIFY_MODEL_SIZE
    ids: list[str] = []
    if task_type == "sync" and (sync_model in MODEL_SPECS or sync_model == "hubert-fa-combined"):
        ids.append(sync_model)
    if task_type in ("auto", "transcribe", "verify") and transcribe_model in MODEL_SPECS:
        spec = get_managed_spec(transcribe_model)
        ids.extend(spec.dependencies)
        ids.append(transcribe_model)
    return list(dict.fromkeys(ids))


async def _enqueue_model_download(model_id: str) -> str | None:
    try:
        item = get_catalog_model(model_id)
    except ValueError:
        return None
    if item.get("source") not in ("huggingface", "audio-separator", "hubertfa_release"):
        return None
    if _catalog_model_installed(model_id):
        return None
    for task in _tasks.values():
        if task.type == "model_download" and task.model_id == model_id and task.status in ("pending", "running", "cancelling", "paused"):
            return task.id
    for dep in item.get("dependencies") or []:
        await _enqueue_model_download(str(dep))
    task = QueueTask(
        id=str(uuid.uuid4())[:8], type="model_download", song_id=0,
        song_name=str(item.get("label") or model_id), lyrics="", model_id=model_id,
        sync_model=ALIGN_MODEL_SIZE, transcribe_model=VERIFY_MODEL_SIZE,
    )
    _tasks[task.id] = task
    await _queue_put_once(task)
    await _q_broadcast()
    return task.id


async def _run_model_download_task(task: QueueTask) -> None:
    item = get_catalog_model(task.model_id)
    label = str(item.get("label") or task.model_id)
    loop = asyncio.get_running_loop()
    task.progress = {"stage": "checking", "msg": f"Checking {label}…", "pct": 0, "step": "model"}
    await _q_broadcast()

    if item.get("source") == "hubertfa_release":
        def progress_update(ev: dict) -> None:
            task.progress = {**ev, "step": "model"}
            asyncio.run_coroutine_threadsafe(_q_broadcast(), loop)

        def cancelled() -> bool:
            return bool(task.cancel_requested)

        def paused() -> bool:
            return bool(task.pause_requested)

        try:
            path = await loop.run_in_executor(None, lambda: ensure_hubertfa(progress_update, cancelled, paused))
        except HubertDownloadPaused:
            task.status = "paused"
            task.pause_requested = False
            return
        except InterruptedError:
            raise asyncio.CancelledError()
        task.result = {"model_id": task.model_id, "path": str(path), "installed": True}
        task.progress = {"stage": "done", "msg": f"{label} ready", "pct": 100, "step": "done"}
        return

    if item.get("source") == "audio-separator":
        if task.pause_requested:
            task.status = "paused"
            task.pause_requested = False
            return
        if task.cancel_requested:
            raise asyncio.CancelledError()
        task.progress = {"stage": "downloading", "msg": f"Downloading {label} through audio-separator…", "pct": 5, "step": "model"}
        await _q_broadcast()
        path = await loop.run_in_executor(None, lambda: _ensure_separator_model(task.model_id))
        if task.cancel_requested:
            try:
                pathlib.Path(path).unlink(missing_ok=True)
            except OSError:
                pass
            raise asyncio.CancelledError()
        task.result = {"model_id": task.model_id, "path": str(path), "installed": True}
        task.progress = {"stage": "done", "msg": f"{label} ready", "pct": 100, "step": "done"}
        return

    spec = get_managed_spec(task.model_id)

    def progress_update(ev: dict) -> None:
        task.progress = {**ev, "step": "model"}
        asyncio.run_coroutine_threadsafe(_q_broadcast(), loop)

    def cancelled() -> bool:
        return bool(task.cancel_requested)

    def paused() -> bool:
        return bool(task.pause_requested)

    try:
        path = await loop.run_in_executor(None, lambda: ensure_managed_model(task.model_id, progress_update, cancelled, paused))
    except DownloadPaused:
        task.status = "paused"
        task.pause_requested = False
        return
    except InterruptedError:
        raise asyncio.CancelledError()
    task.result = {"model_id": task.model_id, "path": str(path), "installed": True}
    task.progress = {"stage": "done", "msg": f"{spec.label} ready", "pct": 100, "step": "done"}



async def _q_broadcast() -> None:
    global _q_state_json
    active   = _active_task.to_dict() if _active_task else None
    pending  = [t.to_dict() for t in _tasks.values() if t.status in ("pending", "paused")]
    done_list = [t for t in _tasks.values() if t.status in ("done", "error", "cancelled")]
    history  = [t.to_dict() for t in sorted(done_list, key=lambda t: t.created_at)][-20:]
    _q_state_json = json.dumps({"active": active, "pending": pending, "history": history})
    if _q_cond is not None:
        async with _q_cond:
            _q_cond.notify_all()


# ── Shared whisper helpers ────────────────────────────────────────────────

def _align(model_obj, tmp_path: str, lyrics: str, fast_mode: bool = False):
    # original_split=True keeps one output segment per input lyric line.
    # Optional fast alignment remains available internally for compatibility.
    # The UI Auto action now performs a full transcription instead; Sync uses
    # the normal alignment path for the raw Lyrics text.
    return model_obj.align(
        tmp_path, lyrics, language="en",
        original_split=True,
        # Let stable-ts keep real silence between words/lines. The optional
        # internal fast path uses a 2-second skip threshold; normal Sync uses 5s.
        nonspeech_skip=2.0 if fast_mode else 5.0,
        fast_mode=fast_mode,
        token_step=200 if fast_mode else 100,
        suppress_silence=True,
        suppress_word_ts=True,
        verbose=False,
    )


def _word_dict(word) -> dict | None:
    raw = str(getattr(word, "word", ""))
    clean = raw.strip()
    if not clean:
        return None
    item = {
        "word": clean,
        "text": raw,
        "start": round(float(word.start), 3),
        "end": round(float(word.end), 3),
    }
    probability = getattr(word, "probability", None)
    if probability is not None:
        try:
            item["probability"] = round(float(probability), 4)
        except (TypeError, ValueError):
            pass
    return item


def _lines_from_alignment(result, lyrics: str) -> list[dict]:
    """Preserve line start/end plus stable-ts's real per-word timing."""
    segments = list(result.segments or [])
    all_words: list[dict] = []
    segment_words: list[list[dict]] = []

    for seg in segments:
        words = []
        for w in (seg.words or []):
            item = _word_dict(w)
            if item:
                words.append(item)
                all_words.append(item)
        segment_words.append(words)

    if not all_words:
        for seg in segments:
            text = seg.text.strip()
            if text:
                all_words.append({
                    "word": text,
                    "text": text,
                    "start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                })

    lines: list[dict] = []
    if lyrics and segments:
        lyric_lines = [line.strip() for line in lyrics.split("\n") if line.strip()]
        if len(segments) == len(lyric_lines):
            for i, (line_text, seg) in enumerate(zip(lyric_lines, segments)):
                lines.append({
                    "line": line_text,
                    "start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "words": segment_words[i],
                })
        else:
            # Compatibility fallback for backends that don't preserve the input
            # split exactly. Slice the aligned word stream by lyric word count.
            ptr = 0
            for line_text in lyric_lines:
                n = max(1, len(line_text.split()))
                chunk = all_words[ptr:ptr + n]
                if chunk:
                    lines.append({
                        "line": line_text,
                        "start": chunk[0]["start"],
                        "end": chunk[-1]["end"],
                        "words": chunk,
                    })
                ptr += n
                if ptr >= len(all_words):
                    break
    else:
        for i, seg in enumerate(segments):
            text = seg.text.strip()
            if text:
                lines.append({
                    "line": text,
                    "start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "words": segment_words[i],
                })
    return lines


def _line_from_faster_segment(segment) -> dict | None:
    """Convert one faster-whisper segment into the app's timed-line shape."""
    text = str(getattr(segment, "text", "") or "").strip()
    if not text:
        return None
    words = []
    for raw_word in (getattr(segment, "words", None) or []):
        item = _word_dict(raw_word)
        if item:
            words.append(item)
    return {
        "line": text,
        "start": round(float(getattr(segment, "start", 0.0)), 3),
        "end": round(float(getattr(segment, "end", 0.0)), 3),
        "words": words,
    }


async def _faster_stream_transcribe_worker(task: QueueTask, tmp_path: str) -> list[dict] | None:
    """Stream faster-whisper segments into QueueTask.live_lines as they decode.

    Returns None when the loaded model is not a faster-whisper model, allowing
    the caller to fall back to stable-ts/PyTorch transcription.
    """
    model_obj = await get_verify_model()
    transcribe_original = getattr(model_obj, "transcribe_original", None)
    if not callable(transcribe_original):
        return None

    loop = asyncio.get_running_loop()
    updates: thread_queue.Queue = thread_queue.Queue()

    def _run():
        try:
            segments, info = transcribe_original(
                tmp_path,
                language="en",
                word_timestamps=True,
            )
            updates.put(("meta", float(getattr(info, "duration", 0.0) or 0.0)))
            for segment in segments:
                if task.cancel_requested:
                    break
                line = _line_from_faster_segment(segment)
                if line:
                    updates.put(("line", line))
            updates.put(("done", None))
        except BaseException as exc:
            updates.put(("error", exc))

    lines: list[dict] = []
    total_duration = 0.0
    finished = False

    async with _inference_lock:
        future = loop.run_in_executor(None, _run)
        while not finished:
            changed = False
            while True:
                try:
                    kind, payload = updates.get_nowait()
                except thread_queue.Empty:
                    break

                if kind == "meta":
                    total_duration = float(payload or 0.0)
                elif kind == "line":
                    lines.append(payload)
                    task.live_lines = list(lines)
                    changed = True
                elif kind == "error":
                    await future
                    raise payload
                elif kind == "done":
                    finished = True
                    break

            if task.cancel_requested:
                task.progress = {
                    "stage": "transcribing", "msg": "Cancelling…", "pct": 0,
                    "step": "transcribing",
                }
                await _q_broadcast()

            if changed:
                end = lines[-1]["end"] if lines else 0.0
                pct = 30 + ((min(1.0, end / total_duration) * 68) if total_duration else 0)
                task.progress = {
                    "stage": "transcribing",
                    "msg": f"Transcribing live… {len(lines)} lines",
                    "pct": pct,
                    "step": "transcribing",
                    "live": True,
                }
                await _q_broadcast()

            if not finished:
                if future.done() and updates.empty():
                    await future
                    finished = True
                    break
                await asyncio.sleep(0.06)

        await future

    if task.cancel_requested:
        raise asyncio.CancelledError()

    # Raw faster-whisper segments already include real word timestamps, including
    # natural silence between words. Preserve those timings for TTML/rendering.
    return _sanitize_timed_lines(
        lines, inline_parenthetical_background=task.inline_parenthetical_background
    )


def _new_terminal_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=24),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
        transient=True,
        disable=not CONSOLE.is_terminal,
    )


def _join_word_text(parts: list[str]) -> str:
    text = ""
    for part in parts:
        part = str(part or "")
        if not part:
            continue
        if not text:
            text = part.strip()
        elif re.fullmatch(r"[,.!?;:%)\]}]+", part):
            text += part
        elif part.startswith(("'", "’")) and len(part) <= 3:
            text += part
        else:
            text += " " + part.strip()
    return text.strip()


def _group_words_into_lines(words: list[dict], *, max_words: int = 10, max_chars: int = 64, pause: float = 0.75) -> list[dict]:
    """Turn timestamped words into readable lyric-sized lines without changing timestamps."""
    lines: list[dict] = []
    cur: list[dict] = []
    for word in words:
        if not cur:
            cur = [word]
            continue
        prev = cur[-1]
        candidate = _join_word_text([x.get("word", "") for x in cur] + [word.get("word", "")])
        gap = float(word.get("start", 0)) - float(prev.get("end", 0))
        prev_text = str(prev.get("word", ""))
        should_break = (
            gap >= pause
            or len(cur) >= max_words
            or len(candidate) > max_chars
            or bool(re.search(r"[.!?][\"')\]]*$", prev_text))
        )
        if should_break:
            lines.append({
                "line": _join_word_text([x.get("word", "") for x in cur]),
                "start": cur[0]["start"], "end": cur[-1]["end"], "words": cur,
            })
            cur = [word]
        else:
            cur.append(word)
    if cur:
        lines.append({
            "line": _join_word_text([x.get("word", "") for x in cur]),
            "start": cur[0]["start"], "end": cur[-1]["end"], "words": cur,
        })
    return lines


def _timed_item_value(item, key: str, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _align_items_to_lyric_lines(items, lyrics: str) -> list[dict]:
    """Map sequential forced-aligner word spans back onto the user's original lyric lines."""
    raw_items = list(getattr(items, "items", items) or [])
    timed = []
    for it in raw_items:
        text = str(_timed_item_value(it, "text", "") or "").strip()
        if not text:
            continue
        start = round(float(_timed_item_value(it, "start_time", 0.0) or 0.0), 3)
        end = round(float(_timed_item_value(it, "end_time", start) or start), 3)
        if end <= start:
            end = round(start + 0.001, 3)
        timed.append({"word": text, "start": start, "end": end})

    out: list[dict] = []
    ptr = 0
    lyric_lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    for line in lyric_lines:
        original_tokens = re.findall(r"\S+", line)
        # Qwen's English forced aligner emits one timed item per cleaned word.
        count = max(1, len(original_tokens))
        chunk = timed[ptr:ptr + count]
        if not chunk:
            break
        words = []
        for i, item in enumerate(chunk):
            display = original_tokens[i] if i < len(original_tokens) else item["word"]
            words.append({**item, "word": display})
        out.append({"line": line, "start": words[0]["start"], "end": words[-1]["end"], "words": words})
        ptr += count
    if ptr < len(timed):
        remainder = timed[ptr:]
        out.append({
            "line": _join_word_text([w["word"] for w in remainder]),
            "start": remainder[0]["start"], "end": remainder[-1]["end"], "words": remainder,
        })
    return out


def _qwen_timestamps_to_lines(text: str, items) -> list[dict]:
    text = str(text or "").strip()
    raw_items = list(getattr(items, "items", items) or [])
    if not raw_items:
        return [{"line": text, "start": 0.0, "end": 0.5, "words": []}] if text else []
    text_tokens = re.findall(r"\S+", text)
    words: list[dict] = []
    for i, item in enumerate(raw_items):
        raw = str(_timed_item_value(item, "text", "") or "").strip()
        display = text_tokens[i] if i < len(text_tokens) else raw
        start = round(float(_timed_item_value(item, "start_time", 0.0) or 0.0), 3)
        end = round(float(_timed_item_value(item, "end_time", start) or start), 3)
        if end <= start:
            end = round(start + 0.001, 3)
        words.append({"word": display or raw, "start": start, "end": end})
    return _group_words_into_lines(words)


def _parakeet_tokens_to_words(tokens: list[dict]) -> list[dict]:
    words: list[dict] = []
    cur_text = ""; cur_start = None; cur_end = None
    def flush():
        nonlocal cur_text, cur_start, cur_end
        text = cur_text.strip()
        if text and cur_start is not None:
            words.append({"word": text, "start": round(float(cur_start), 3), "end": round(max(float(cur_end or cur_start), float(cur_start) + .001), 3)})
        cur_text = ""; cur_start = None; cur_end = None
    for item in tokens or []:
        tok = str(item.get("token", "") or "")
        start = float(item.get("start", 0.0) or 0.0)
        end = float(item.get("end", start) or start)
        if not tok:
            continue
        starts_new = bool(tok[:1].isspace())
        stripped = tok.strip()
        if starts_new and cur_text:
            flush()
        if re.fullmatch(r"[,.!?;:%)\]}]+", stripped) and cur_text:
            cur_text += stripped
            cur_end = max(cur_end or end, end)
            continue
        if cur_start is None: cur_start = start
        cur_end = max(cur_end or end, end)
        cur_text += tok if cur_text else stripped
    flush()
    return words


async def _qwen_sync_worker(task: QueueTask, tmp_path: str, lyrics: str) -> list[dict]:
    runtime = await get_align_model(task.sync_model or ALIGN_MODEL_SIZE)
    loop = asyncio.get_running_loop()
    task.progress = {"stage": "aligning", "msg": "Qwen forced alignment…", "pct": 55, "step": "aligning"}
    await _q_broadcast()

    def run_alignment():
        import torch
        from whisper.audio import load_audio
        processor, net = runtime.processor, runtime.model
        audio = load_audio(tmp_path)  # ffmpeg -> mono float32 16 kHz
        inputs, word_lists = processor.prepare_forced_aligner_inputs(
            audio=audio, transcript=lyrics, language="English",
        )
        inputs = inputs.to(net.device, net.dtype)
        with torch.inference_mode():
            outputs = net(**inputs)
        return processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=net.config.timestamp_token_id,
        )[0]

    async with _inference_lock:
        fut = loop.run_in_executor(None, run_alignment)
        elapsed = 0.0
        while not fut.done():
            if task.cancel_requested:
                await asyncio.shield(fut)
                raise asyncio.CancelledError()
            pct = min(96, 55 + (elapsed / 90.0) ** 0.5 * 35)
            task.progress = {"stage": "aligning", "msg": f"Qwen forced alignment… {elapsed:.0f}s", "pct": pct, "step": "aligning"}
            await _q_broadcast(); await asyncio.sleep(.5); elapsed += .5
        timestamps = await fut
    return _sanitize_timed_lines(
        _align_items_to_lyric_lines(timestamps, lyrics),
        inline_parenthetical_background=task.inline_parenthetical_background,
    )


async def _hubertfa_sync_worker(task: QueueTask, tmp_path: str, lyrics: str) -> list[dict]:
    runtime = await get_align_model("hubert-fa-combined")
    loop = asyncio.get_running_loop()
    task.progress = {"stage": "aligning", "msg": "HuBERT FA phoneme alignment…", "pct": 55, "step": "aligning"}
    await _q_broadcast()

    async with _inference_lock:
        fut = loop.run_in_executor(None, lambda: runtime.align(tmp_path, lyrics))
        elapsed = 0.0
        while not fut.done():
            if task.cancel_requested:
                await asyncio.shield(fut)
                raise asyncio.CancelledError()
            pct = min(96, 55 + (elapsed / 90.0) ** 0.5 * 35)
            task.progress = {"stage": "aligning", "msg": f"HuBERT FA phoneme alignment… {elapsed:.0f}s", "pct": pct, "step": "aligning"}
            await _q_broadcast()
            await asyncio.sleep(.5)
            elapsed += .5
        timestamps = await fut

    return _sanitize_timed_lines(
        _align_items_to_lyric_lines(timestamps, lyrics),
        inline_parenthetical_background=task.inline_parenthetical_background,
    )


async def _managed_transcribe_worker(task: QueueTask, tmp_path: str) -> list[dict] | None:
    model_id = task.transcribe_model or VERIFY_MODEL_SIZE
    if model_id not in MODEL_SPECS:
        return None
    runtime = await get_verify_model(model_id)
    loop = asyncio.get_running_loop()
    label = get_managed_spec(model_id).label
    task.progress = {"stage": "transcribing", "msg": f"{label} transcription…", "pct": 30, "step": "transcribing", "live": False}
    await _q_broadcast()

    def run_qwen():
        import torch
        from whisper.audio import load_audio
        asr_processor, asr_net = runtime.processor, runtime.model
        audio = load_audio(tmp_path)  # decode once; Qwen and its aligner both consume 16 kHz PCM
        inputs = asr_processor.apply_transcription_request(
            audio=audio, language="English",
        ).to(asr_net.device, asr_net.dtype)
        with torch.inference_mode():
            output_ids = asr_net.generate(**inputs, max_new_tokens=4096, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        parsed = asr_processor.decode(generated_ids, return_format="parsed")[0]
        transcript = str(parsed.get("transcription", "") or "").strip()
        language = parsed.get("language") or "English"

        aligner = runtime.aligner
        align_processor, align_net = aligner.processor, aligner.model
        align_inputs, word_lists = align_processor.prepare_forced_aligner_inputs(
            audio=audio, transcript=transcript, language=language,
        )
        align_inputs = align_inputs.to(align_net.device, align_net.dtype)
        with torch.inference_mode():
            outputs = align_net(**align_inputs)
        timestamps = align_processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=align_inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=align_net.config.timestamp_token_id,
        )[0]
        return transcript, timestamps

    def run_parakeet():
        from whisper.audio import load_audio
        processor = runtime.processor; net = runtime.model
        sr = int(processor.feature_extractor.sampling_rate)
        if sr != 16000:
            raise RuntimeError(f"Parakeet processor expected unsupported sample rate {sr} Hz")
        audio = load_audio(tmp_path)  # ffmpeg -> mono float32 16 kHz
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = inputs.to(net.device, dtype=net.dtype)
        output = net.generate(**inputs, return_dict_in_generate=True)
        decoded, timestamps = processor.decode(output.sequences, durations=output.durations, skip_special_tokens=True)
        text = decoded[0] if isinstance(decoded, list) else str(decoded)
        token_items = timestamps[0] if timestamps and isinstance(timestamps[0], list) else timestamps
        words = _parakeet_tokens_to_words(token_items or [])
        return text, words

    runner = run_qwen if model_id.startswith("qwen3-asr-") else run_parakeet
    async with _inference_lock:
        fut = loop.run_in_executor(None, runner)
        elapsed = 0.0
        while not fut.done():
            if task.cancel_requested:
                await asyncio.shield(fut)
                raise asyncio.CancelledError()
            pct = min(96, 30 + (elapsed / 150.0) ** 0.5 * 60)
            task.progress = {"stage": "transcribing", "msg": f"{label} transcription… {elapsed:.0f}s", "pct": pct, "step": "transcribing", "live": False}
            await _q_broadcast(); await asyncio.sleep(.5); elapsed += .5
        raw = await fut

    if model_id.startswith("qwen3-asr-"):
        text, timestamps = raw
        lines = _qwen_timestamps_to_lines(text, timestamps)
    else:
        text, words = raw
        lines = _group_words_into_lines(words) if words else ([{"line": text, "start": 0.0, "end": 0.5, "words": []}] if text else [])
    return _sanitize_timed_lines(lines, inline_parenthetical_background=task.inline_parenthetical_background)


async def _selected_transcribe_worker(task: QueueTask, tmp_path: str) -> list[dict]:
    managed = await _managed_transcribe_worker(task, tmp_path)
    if managed is not None:
        return managed
    lines = await _faster_stream_transcribe_worker(task, tmp_path)
    if lines is not None:
        return lines
    return await _whisper_sync_worker(task, tmp_path, "")


async def _selected_verify_worker(task: QueueTask, tmp_path: str, lyrics: str) -> list[dict]:
    model_id = task.transcribe_model or VERIFY_MODEL_SIZE
    if model_id in MODEL_SPECS:
        lines = await _managed_transcribe_worker(task, tmp_path) or []
        transcription = " ".join(str(x.get("line", "")) for x in lines)
        return _verify_lines([l for l in lyrics.splitlines() if l.strip()], transcription)
    return await _whisper_verify_worker(task, tmp_path, lyrics)


async def _whisper_sync_worker(task: QueueTask, tmp_path: str, lyrics: str, fast_mode: bool = False) -> list[dict]:
    """Run align/transcribe in executor. Returns lines."""
    label = "Aligning" if lyrics else "Transcribing"
    spy = _ProgressSpy()
    spy.silent = True  # capture stable-ts tqdm instead of dumping raw progress into uvicorn logs
    loop = asyncio.get_running_loop()
    model_obj = await (get_align_model(task.sync_model or ALIGN_MODEL_SIZE) if lyrics else get_verify_model(task.transcribe_model or VERIFY_MODEL_SIZE))
    started = time.perf_counter()
    display = _display_name(task.song_name) or f"song {task.song_id}"

    def _run():
        orig = sys.stderr
        sys.stderr = _TeeStderr(spy, orig)
        try:
            if lyrics:
                return _align(model_obj, tmp_path, lyrics, fast_mode=fast_mode)
            return model_obj.transcribe(tmp_path, word_timestamps=True, verbose=False)
        finally:
            sys.stderr = orig

    terminal = _new_terminal_progress()
    terminal_id = terminal.add_task(f"[cyan]{label}[/cyan] {display}", total=100)
    terminal.start()
    try:
        async with _inference_lock:
            fut = loop.run_in_executor(None, _run)
            cancelled = False
            elapsed = 0
            while not fut.done():
                if task.cancel_requested:
                    cancelled = True
                    task.progress = {"stage": "aligning", "msg": "Cancelling…", "pct": 0, "step": "aligning"}
                    await _q_broadcast()
                    await asyncio.shield(fut)
                    break
                prog = spy.latest()
                if prog:
                    terminal.update(terminal_id, completed=prog["pct"])
                    pct = 55 + prog["pct"] * 0.44
                    msg = (f"{label}: {prog['pct']}%  "
                           f"{prog['done']:.1f}/{prog['total']:.1f}s  "
                           f"[{prog['elapsed']}<{prog['eta']}, {prog['speed']:.2f}s/sec]")
                else:
                    terminal.update(terminal_id, completed=min(95, (elapsed / 180) ** 0.5 * 100))
                    pct = min(54, int((elapsed / 180) ** 0.5 * 54))
                    msg = f"{label}… {elapsed}s"
                task.progress = {
                    "stage": "aligning", "pct": pct, "msg": msg, "step": "aligning",
                    **({"progress": prog} if prog else {}),
                }
                await _q_broadcast()
                await asyncio.sleep(0.5)
                elapsed += 1
            if not cancelled:
                result = await fut
    finally:
        terminal.stop()

    if cancelled:
        raise asyncio.CancelledError()

    lines = _sanitize_timed_lines(
        _lines_from_alignment(result, lyrics),
        inline_parenthetical_background=task.inline_parenthetical_background,
    )
    CONSOLE.print(
        f"[green]✓[/green] {'Aligned' if lyrics else 'Transcribed'} [bold]{display}[/bold] "
        f"[dim]({len(lines)} lines, {time.perf_counter() - started:.1f}s)[/dim]"
    )
    return lines


async def _whisper_verify_worker(task: QueueTask, tmp_path: str, lyrics: str) -> list[dict]:
    """Free-transcribe + compare. Returns verify_results list."""
    spy = _ProgressSpy()
    spy.silent = True
    loop = asyncio.get_running_loop()
    model_v = await get_verify_model()
    started = time.perf_counter()
    display = _display_name(task.song_name) or f"song {task.song_id}"

    def _run():
        orig = sys.stderr
        sys.stderr = _TeeStderr(spy, orig)
        try:
            return model_v.transcribe(
                tmp_path, verbose=False, word_timestamps=False,
                suppress_silence=False, regroup=False,
            )
        finally:
            sys.stderr = orig

    terminal = _new_terminal_progress()
    terminal_id = terminal.add_task(f"[cyan]Verifying[/cyan] {display}", total=100)
    terminal.start()
    try:
        async with _inference_lock:
            fut = loop.run_in_executor(None, _run)
            cancelled = False
            elapsed = 0
            while not fut.done():
                if task.cancel_requested:
                    cancelled = True
                    task.progress = {"stage": "transcribing", "msg": "Cancelling…", "pct": 0, "step": "verifying"}
                    await _q_broadcast()
                    await asyncio.shield(fut)
                    break
                prog = spy.latest()
                if prog:
                    terminal.update(terminal_id, completed=prog["pct"])
                    pct = 42 + prog["pct"] * 0.53
                    msg = (f"Transcribing: {prog['pct']}%  "
                           f"{prog['done']:.1f}/{prog['total']:.1f}s  "
                           f"[{prog['elapsed']}<{prog['eta']}, {prog['speed']:.2f}s/sec]")
                else:
                    terminal.update(terminal_id, completed=min(95, (elapsed / 180) ** 0.5 * 100))
                    pct = min(41, int((elapsed / 180) ** 0.5 * 41))
                    msg = f"Transcribing… {elapsed}s"
                task.progress = {
                    "stage": "transcribing", "pct": pct, "msg": msg, "step": "verifying",
                    **({"progress": prog} if prog else {}),
                }
                await _q_broadcast()
                await asyncio.sleep(0.5)
                elapsed += 1
            if not cancelled:
                result = await fut
    finally:
        terminal.stop()

    if cancelled:
        raise asyncio.CancelledError()

    transcription = result.text or ""
    lyric_lines = [l for l in lyrics.split("\n") if l.strip()]
    verified = _verify_lines(lyric_lines, transcription)
    CONSOLE.print(
        f"[green]✓[/green] Verified [bold]{display}[/bold] "
        f"[dim]({len(verified)} lines, {time.perf_counter() - started:.1f}s)[/dim]"
    )
    return verified


async def _download_audio(task: QueueTask, song: dict) -> str:
    """Stream audio via ensure_audio, broadcasting progress. Returns tmp_path."""
    tmp_path = None
    async for ev in ensure_audio(song["path"]):
        if task.cancel_requested:
            raise asyncio.CancelledError()
        if "_path" in ev:
            tmp_path = ev["_path"]
        else:
            task.progress = {**ev, "step": "downloading"}
            await _q_broadcast()
    if tmp_path is None:
        raise ValueError("Audio download failed.")
    return tmp_path


# ── Task runners ──────────────────────────────────────────────────────────

async def _run_sync_task(task: QueueTask) -> None:
    if task.local_path:
        tmp_path = task.local_path
        lyrics   = task.lyrics
    else:
        task.progress = {"stage": "fetching", "msg": "Fetching song info…", "step": "fetching", "pct": 2}
        await _q_broadcast()
        song = await jw_get(f"/songs/{task.song_id}/")
        if not song.get("path"):
            raise ValueError("No audio file for this song.")
        lyrics   = task.lyrics or song.get("lyrics", "") or ""
        tmp_path = await _download_audio(task, song)
    task.lyrics = lyrics
    if task.cancel_requested:
        raise asyncio.CancelledError()
    tmp_path = await _prepare_analysis_audio(task, tmp_path)
    sync_model = task.sync_model or ALIGN_MODEL_SIZE
    label = get_managed_spec(sync_model).label if sync_model in MODEL_SPECS else f"Whisper {sync_model}"
    task.progress = {"stage": "loading", "msg": f"Loading {label}…", "step": "loading", "pct": 52}
    await _q_broadcast()
    if sync_model == "qwen3-forced-aligner-0.6b":
        lines = await _qwen_sync_worker(task, tmp_path, lyrics)
    elif sync_model == "hubert-fa-combined":
        lines = await _hubertfa_sync_worker(task, tmp_path, lyrics)
    else:
        lines = await _whisper_sync_worker(task, tmp_path, lyrics)
    task.result   = {"lines": lines}
    task.progress = {"stage": "done", "msg": f"Done — {len(lines)} lines synced", "step": "done", "pct": 100}


async def _run_verify_task(task: QueueTask) -> None:
    if task.local_path:
        tmp_path = task.local_path
        lyrics   = task.lyrics
    else:
        task.progress = {"stage": "fetching", "msg": "Fetching song info…", "step": "fetching", "pct": 2}
        await _q_broadcast()
        song = await jw_get(f"/songs/{task.song_id}/")
        if not song.get("path"):
            raise ValueError("No audio file for this song.")
        lyrics = task.lyrics or song.get("lyrics", "") or ""
        tmp_path = await _download_audio(task, song)
    if not lyrics:
        raise ValueError("No lyrics to verify against.")
    if task.cancel_requested:
        raise asyncio.CancelledError()
    tmp_path = await _prepare_analysis_audio(task, tmp_path)
    transcribe_model = task.transcribe_model or VERIFY_MODEL_SIZE
    label = get_managed_spec(transcribe_model).label if transcribe_model in MODEL_SPECS else f"Whisper {transcribe_model}"
    task.progress = {"stage": "loading", "msg": f"Loading {label}…", "step": "loading", "pct": 38}
    await _q_broadcast()
    verify_results = await _selected_verify_worker(task, tmp_path, lyrics)
    counts = {"present": 0, "uncertain": 0, "absent": 0}
    for r in verify_results:
        counts[r["status"]] += 1
    task.result   = {"verify": verify_results, "counts": counts}
    task.progress = {"stage": "done",
                     "msg": f"Done — ✓{counts['present']} ?{counts['uncertain']} ✗{counts['absent']}",
                     "step": "done", "pct": 100}


async def _run_transcribe_task(task: QueueTask) -> None:
    """Transcribe audio to plain text, ignoring any existing lyrics."""
    if task.local_path:
        tmp_path = task.local_path
    else:
        task.progress = {"stage": "fetching", "msg": "Fetching song info…", "step": "fetching", "pct": 2}
        await _q_broadcast()
        song = await jw_get(f"/songs/{task.song_id}/")
        if not song.get("path"):
            raise ValueError("No audio file for this song.")
        tmp_path = await _download_audio(task, song)
    if task.cancel_requested:
        raise asyncio.CancelledError()
    tmp_path = await _prepare_analysis_audio(task, tmp_path)
    transcribe_model = task.transcribe_model or VERIFY_MODEL_SIZE
    label = get_managed_spec(transcribe_model).label if transcribe_model in MODEL_SPECS else f"Whisper {transcribe_model}"
    task.progress = {"stage": "loading", "msg": f"Loading {label}…", "step": "loading", "pct": 30}
    await _q_broadcast()

    lines = await _selected_transcribe_worker(task, tmp_path)

    plain_text = "\n".join(l["line"] for l in lines)
    task.live_lines = []
    task.result   = {"lines": lines, "text": plain_text}
    task.progress = {"stage": "done", "msg": f"Transcribed — {len(lines)} lines", "step": "done", "pct": 100}


async def _run_auto_task(task: QueueTask) -> None:
    """Run a full Whisper transcription and keep both raw + timed lyrics."""
    if task.local_path:
        tmp_path = task.local_path
    else:
        task.progress = {"stage": "fetching", "msg": "Fetching song info…", "step": "fetching", "pct": 2}
        await _q_broadcast()
        song = await jw_get(f"/songs/{task.song_id}/")
        if not song.get("path"):
            raise ValueError("No audio file for this song.")
        tmp_path = await _download_audio(task, song)

    if task.cancel_requested:
        raise asyncio.CancelledError()

    tmp_path = await _prepare_analysis_audio(task, tmp_path)
    transcribe_model = task.transcribe_model or VERIFY_MODEL_SIZE
    label = get_managed_spec(transcribe_model).label if transcribe_model in MODEL_SPECS else f"Whisper {transcribe_model}"
    task.progress = {"stage": "loading", "msg": f"Loading {label}…", "step": "loading", "pct": 28}
    await _q_broadcast()

    lines = await _selected_transcribe_worker(task, tmp_path)

    plain_text = "\n".join(line.get("line", "") for line in lines).strip()
    task.lyrics = plain_text
    task.live_lines = []
    task.result = {
        "lines": lines,
        "text": plain_text,
        "auto_mode": "full-transcription",
        "word_timing": any(line.get("words") for line in lines),
    }
    task.progress = {
        "stage": "done",
        "msg": f"Auto complete — transcribed {len(lines)} lines",
        "step": "done",
        "pct": 100,
    }


# ── Background processor ──────────────────────────────────────────────────

async def _queue_processor() -> None:
    global _active_task
    while True:
        task = await _task_queue.get()
        _queued_task_ids.discard(task.id)

        if task.status in ("cancelled", "cancelling", "paused"):
            _task_queue.task_done()
            await _q_broadcast()
            continue

        _active_task = task
        task.status  = "running"
        await _q_broadcast()

        try:
            if task.type == "sync":
                await _run_sync_task(task)
            elif task.type == "verify":
                await _run_verify_task(task)
            elif task.type == "auto":
                await _run_auto_task(task)
            elif task.type == "transcribe":
                await _run_transcribe_task(task)
            elif task.type == "model_download":
                await _run_model_download_task(task)
            if task.status in ("running", "cancelling"):   # runner didn't set error/cancelled
                task.status = "done"
            if task.status == "done" and task.result and task.result.get("lines"):
                try:
                    _store_processed_lyrics(task)
                except sqlite3.Error as exc:
                    CONSOLE.print(f"[yellow]Could not save processed lyrics for {task.song_id}: {exc}[/yellow]")
        except asyncio.CancelledError:
            task.status = "cancelled"
        except Exception as exc:
            task.status = "error"
            task.error  = str(exc)
            CONSOLE.print(f"[red]Task {task.id} failed:[/red] {exc}")
        finally:
            _active_task = None
            _task_queue.task_done()
            await _q_broadcast()
            # Prune old history (keep last 30)
            done_list = [t for t in _tasks.values() if t.status in ("done", "error", "cancelled")]
            if len(done_list) > 30:
                for old in sorted(done_list, key=lambda t: t.created_at)[:-30]:
                    _tasks.pop(old.id, None)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def _log_startup_diagnostics() -> None:
    managed_selected = ALIGN_MODEL_SIZE in MODEL_SPECS or VERIFY_MODEL_SIZE in MODEL_SPECS
    resolved = _get_device("torch") if managed_selected else _get_device(ENGINE_PREF)
    table = Table(show_header=False, box=None, padding=(0, 2), expand=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Python", sys.executable)
    table.add_row("Models", f"sync {ALIGN_MODEL_SIZE} · transcribe {VERIFY_MODEL_SIZE} · Whisper engine {ENGINE_PREF}")
    table.add_row("Device", f"{DEVICE_PREF} → {resolved}")

    try:
        import torch
        torch_cuda = torch.version.cuda or "CPU-only"
        torch_ok = _torch_cuda_usable()
        table.add_row("PyTorch", f"{torch.__version__} · CUDA build {torch_cuda} · usable: {'yes' if torch_ok else 'no'}")
        if torch.cuda.is_available():
            table.add_row("Torch GPU", torch.cuda.get_device_name(0))
    except Exception as exc:
        table.add_row("PyTorch", f"unavailable ({exc})")

    try:
        import ctranslate2
        ct2_count = ctranslate2.get_cuda_device_count()
        table.add_row("CTranslate2", f"{ctranslate2.__version__} · CUDA GPUs: {ct2_count}")
    except Exception as exc:
        table.add_row("CTranslate2", f"unavailable ({exc})")

    table.add_row("Lyrics DB", str(DB_PATH))
    CONSOLE.print("[bold magenta]WRLD Sync backend[/bold magenta]")
    CONSOLE.print(table)




@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db()
    _log_startup_diagnostics()
    global _q_cond
    _q_cond = asyncio.Condition()
    proc = asyncio.create_task(_queue_processor())
    yield
    proc.cancel()
    try:
        await proc
    except asyncio.CancelledError:
        pass


app = FastAPI(title="WRLD Sync", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def jw_get(path: str, params: dict | None = None) -> dict:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(BASE + path, params=params)
            r.raise_for_status()
            return r.json()
    except httpx.TimeoutException:
        raise HTTPException(504, "juicewrldapi.com took too long to respond. Try again in a moment.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, f"juicewrldapi.com returned an error ({e.response.status_code}).")
    except httpx.HTTPError:
        raise HTTPException(502, "Couldn't reach juicewrldapi.com. Check your connection and try again.")


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@app.get("/api/search")
async def search(q: str, page_size: int = 20):
    return await jw_get("/songs/", {"search": q, "page_size": page_size})


@app.get("/api/song/{song_id}")
async def get_song(song_id: int):
    song = await jw_get(f"/songs/{song_id}/")
    cached = _get_processed_lyrics(song_id)
    if cached:
        song["_local_processed"] = cached
    return song


@app.get("/api/processed/{song_id}")
async def get_processed(song_id: int):
    cached = _get_processed_lyrics(song_id)
    if not cached:
        raise HTTPException(404, "No locally processed lyrics stored for this song.")
    return cached


@app.delete("/api/processed/{song_id}")
async def delete_processed(song_id: int):
    with _db_connect() as conn:
        cur = conn.execute("DELETE FROM processed_lyrics WHERE song_id = ?", (int(song_id),))
    return {"deleted": cur.rowcount > 0}


class ProcessedSaveRequest(BaseModel):
    song_name: str = ""
    source_type: str = "manual"
    lyrics: str = ""
    lines: list[dict]
    ttml: str = ""
    settings: dict = Field(default_factory=dict)


@app.post("/api/processed/{song_id}")
async def save_processed(song_id: int, req: ProcessedSaveRequest):
    if song_id <= 0:
        raise HTTPException(400, "A catalog song ID is required for persistent processed lyrics.")
    lines = _sanitize_timed_lines(req.lines)
    if not lines:
        raise HTTPException(400, "No timed lyric lines to save.")
    ttml = req.ttml.strip()
    if not ttml:
        ttml, _ = _build_ttml(
            lines,
            word_timing=bool(req.settings.get("word_timing", True)),
            detect_interludes=bool(req.settings.get("detect_interludes", True)),
            inline_parenthetical_background=bool(req.settings.get("inline_parenthetical_background", True)),
        )
    settings = {
        "task_type": req.source_type or "manual",
        "engine": ENGINE_PREF,
        "device_pref": DEVICE_PREF,
        "resolved_device": _get_device("torch") if (ALIGN_MODEL_SIZE in MODEL_SPECS or VERIFY_MODEL_SIZE in MODEL_SPECS) else _get_device(ENGINE_PREF),
        "align_model": ALIGN_MODEL_SIZE,
        "verify_model": VERIFY_MODEL_SIZE,
        "sync_model": ALIGN_MODEL_SIZE,
        "transcribe_model": VERIFY_MODEL_SIZE,
        **dict(req.settings or {}),
    }
    now = time.time()
    with _db_connect() as conn:
        conn.execute(
            """
            INSERT INTO processed_lyrics (
                song_id, song_name, source_type, lyrics_text, synced_lines_json,
                ttml, settings_json, processed_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(song_id) DO UPDATE SET
                song_name=excluded.song_name, source_type=excluded.source_type,
                lyrics_text=excluded.lyrics_text, synced_lines_json=excluded.synced_lines_json,
                ttml=excluded.ttml, settings_json=excluded.settings_json, updated_at=excluded.updated_at
            """,
            (int(song_id), req.song_name, req.source_type, req.lyrics,
             json.dumps(lines, ensure_ascii=False), ttml, json.dumps(settings, ensure_ascii=False), now, now),
        )
    return _get_processed_lyrics(song_id)


@app.get("/api/local/processed/{track_hash}")
async def get_local_processed(track_hash: str):
    cached = _get_local_processed(track_hash)
    if not cached:
        raise HTTPException(404, "No processed lyrics stored for this local track.")
    return cached


@app.delete("/api/local/processed/{track_hash}")
async def delete_local_processed(track_hash: str):
    track_hash = track_hash.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", track_hash):
        raise HTTPException(400, "Invalid local track hash.")
    with _db_connect() as conn:
        cur = conn.execute("DELETE FROM local_processed_lyrics WHERE track_hash = ?", (track_hash,))
    return {"deleted": cur.rowcount > 0}


@app.post("/api/local/processed/{track_hash}")
async def save_local_processed(track_hash: str, req: ProcessedSaveRequest):
    track_hash = track_hash.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", track_hash):
        raise HTTPException(400, "Invalid local track hash.")
    if not _get_local_track(track_hash):
        raise HTTPException(404, "Local track is not registered.")
    lines = _sanitize_timed_lines(req.lines)
    if not lines:
        raise HTTPException(400, "No timed lyric lines to save.")
    ttml = req.ttml.strip()
    if not ttml:
        ttml, _ = _build_ttml(
            lines,
            word_timing=bool(req.settings.get("word_timing", True)),
            detect_interludes=bool(req.settings.get("detect_interludes", True)),
            inline_parenthetical_background=bool(req.settings.get("inline_parenthetical_background", True)),
        )
    settings = {
        "task_type": req.source_type or "manual",
        "engine": ENGINE_PREF,
        "device_pref": DEVICE_PREF,
        "resolved_device": _get_device("torch") if (ALIGN_MODEL_SIZE in MODEL_SPECS or VERIFY_MODEL_SIZE in MODEL_SPECS) else _get_device(ENGINE_PREF),
        "align_model": ALIGN_MODEL_SIZE,
        "verify_model": VERIFY_MODEL_SIZE,
        "sync_model": ALIGN_MODEL_SIZE,
        "transcribe_model": VERIFY_MODEL_SIZE,
        **dict(req.settings or {}),
    }
    now = time.time()
    with _db_connect() as conn:
        conn.execute(
            """
            INSERT INTO local_processed_lyrics (
                track_hash, track_name, source_type, lyrics_text, synced_lines_json,
                ttml, settings_json, processed_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(track_hash) DO UPDATE SET
                track_name=excluded.track_name, source_type=excluded.source_type,
                lyrics_text=excluded.lyrics_text, synced_lines_json=excluded.synced_lines_json,
                ttml=excluded.ttml, settings_json=excluded.settings_json, updated_at=excluded.updated_at
            """,
            (track_hash, req.song_name, req.source_type, req.lyrics,
             json.dumps(lines, ensure_ascii=False), ttml, json.dumps(settings, ensure_ascii=False), now, now),
        )
    return _get_local_processed(track_hash)


@app.get("/api/songs/all")
async def get_all_songs():
    """Every song in the catalog, via the upstream `?all=true` bulk variant
    (bypasses pagination entirely instead of walking pages)."""
    data = await jw_get("/songs/", {"all": "true"})
    songs = data if isinstance(data, list) else data.get("results", [])
    return {"songs": songs}


@app.get("/api/versions/all")
async def get_all_versions():
    """Every saved version/grouping row across the whole catalog, via the
    upstream `?all=true` bulk variant (bypasses pagination entirely)."""
    rows = await jw_get("/versions/", {"all": "true"})
    if not isinstance(rows, list):
        rows = rows.get("results", [])
    for row in rows:
        if "title" in row:
            row["version_title"] = row.pop("title")
    return {"versions": rows}


@app.get("/api/versions/{song_id}")
async def get_versions(song_id: int):
    """Version/grouping rows for a song (and its group-mates), if any.

    Upstream calls the group's display name "title"; our own contract with
    the frontend uses "version_title" (matching the old Supabase column
    name), so translate it here rather than leaking the upstream naming.
    """
    data = await jw_get(f"/versions/{song_id}/")
    for row in data.get("results", []):
        if "title" in row:
            row["version_title"] = row.pop("title")
    return data


class VersionSaveRequest(BaseModel):
    token: str
    group_id: int
    version: str | None = None
    version_title: str | None = None


async def _write_version(song_id: int, req: VersionSaveRequest, method: str, pk: int | None = None):
    if not req.token:
        raise HTTPException(401, "No auth token provided.")
    # Create (POST) targets the list route; update (PATCH) targets the row's
    # own detail route — Django's `versions/<int:song_id>/<int:pk>/`.
    path = f"/versions/{song_id}/{pk}/" if pk is not None else f"/versions/{song_id}/"
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        r = await client.request(
            method,
            BASE + path,
            headers={
                "Authorization": f"Token {req.token}",
                "Content-Type": "application/json",
            },
            json={
                # Upstream requires both fields present and non-blank —
                # "title" (not "version_title") and "version" as a string.
                # Fall back to "Null" when a song has nothing more specific
                # (e.g. it's alone with no other versions, or a "reset to
                # automatic" that has no real label to give).
                "group_id": req.group_id,
                "version": req.version or "Null",
                "title": req.version_title or "unknown",
            },
        )
        body = r.text
        if r.status_code in (401, 403):
            raise HTTPException(403, "Token rejected — editor role required.")
        if not r.is_success:
            raise HTTPException(r.status_code, f"API error {r.status_code}: {body}")
        return r.json()


@app.post("/api/versions/{song_id}")
async def create_version(song_id: int, req: VersionSaveRequest):
    """First-time save for a song that has no version row yet."""
    return await _write_version(song_id, req, "POST")


@app.patch("/api/versions/{song_id}/{pk}")
async def update_version(song_id: int, pk: int, req: VersionSaveRequest):
    """Update of a song's existing version row (pk = that row's own id)."""
    return await _write_version(song_id, req, "PATCH", pk=pk)


@app.get("/api/model")
async def get_model_info():
    # Managed Qwen/Parakeet models run through PyTorch; Whisper's displayed
    # device follows the selected Whisper engine for backward compatibility.
    managed_selected = ALIGN_MODEL_SIZE in MODEL_SPECS or VERIFY_MODEL_SIZE in MODEL_SPECS
    avail_device = "cuda" if (_torch_cuda_usable() if managed_selected else _cuda_usable()) else "cpu"
    return {
        "model": ALIGN_MODEL_SIZE,  # legacy
        "loaded": _align_model is not None,
        "align_model": ALIGN_MODEL_SIZE,
        "verify_model": VERIFY_MODEL_SIZE,
        "device_pref": DEVICE_PREF,
        "engine": ENGINE_PREF,
        "engines": WHISPER_ENGINES,
        "device": avail_device,
        "align_loaded": _align_model is not None and _align_model_id == ALIGN_MODEL_SIZE,
        "verify_loaded": _verify_model is not None and _verify_model_id == VERIFY_MODEL_SIZE,
        "models": WHISPER_MODELS,  # legacy
        "sync_models": SYNC_MODELS,
        "transcribe_models": TRANSCRIBE_MODELS,
        "managed_models": _all_catalog_status(),
    }


@app.get("/api/models")
async def get_managed_models():
    return {
        "models": _all_catalog_status(),
        "models_dir": str(pathlib.Path(__file__).parent / "models"),
        "notes": load_catalog().get("notes", ""),
    }


@app.post("/api/models/{model_id}/download")
async def download_managed_model(model_id: str):
    try:
        item = get_catalog_model(model_id)
    except ValueError:
        raise HTTPException(404, f"Unknown model '{model_id}'")
    if item.get("source") not in ("huggingface", "audio-separator", "hubertfa_release"):
        reason = item.get("blocked_reason") or "This model is not downloaded through WRLD Sync's project-local model manager."
        raise HTTPException(409, reason)
    task_id = await _enqueue_model_download(model_id)
    return {"model_id": model_id, "installed": _catalog_model_installed(model_id), "task_id": task_id}


@app.delete("/api/models/{model_id}")
async def delete_managed_model(model_id: str):
    global _align_model, _verify_model, _align_model_id, _verify_model_id
    try:
        item = get_catalog_model(model_id)
    except ValueError:
        raise HTTPException(404, f"Unknown model '{model_id}'")
    if _align_model_id == model_id:
        _align_model = None
        _align_model_id = None
    if _verify_model_id == model_id:
        _verify_model = None
        _verify_model_id = None
    if model_id in MODEL_SPECS:
        return await asyncio.to_thread(remove_managed_model, model_id)
    if item.get("source") == "audio-separator":
        asset = _UVR_MODEL_DIR / str(item.get("asset") or item.get("runtime_id") or "")
        existed = asset.is_file()
        asset.unlink(missing_ok=True)
        return {"model_id": model_id, "removed": existed}
    if item.get("source") == "hubertfa_release":
        return await asyncio.to_thread(remove_hubertfa)
    raise HTTPException(409, "This model is not managed by WRLD Sync's removable project-local model manager.")


@app.get("/api/settings/advanced")
async def get_advanced_settings():
    return {
        **ADVANCED_SETTINGS,
        "cookies_supported": False,
        "supported_url_services": ["YouTube", "YouTube Music", "SoundCloud"],
        "separator_models": [m for m in _all_catalog_status() if "separation" in m.get("tasks", [])],
    }


@app.post("/api/settings/advanced")
async def set_advanced_settings(body: dict):
    global ADVANCED_SETTINGS
    values = dict(ADVANCED_SETTINGS)
    if "preprocess_vocals" in body:
        values["preprocess_vocals"] = bool(body["preprocess_vocals"])
    if "separator_model" in body:
        model_id = str(body["separator_model"] or "").strip()
        try:
            item = get_catalog_model(model_id)
        except ValueError:
            raise HTTPException(400, f"Unknown separator model '{model_id}'")
        if "separation" not in item.get("tasks", []):
            raise HTTPException(400, f"'{model_id}' is not a separation model")
        values["separator_model"] = model_id
    if "separator_target" in body:
        target = str(body["separator_target"] or "").strip()
        if target != "all_vocals":
            raise HTTPException(400, "Only all_vocals is currently supported; it preserves backing vocals, echoes, and ad-libs.")
        values["separator_target"] = target
    if "yt_dlp_enabled" in body:
        values["yt_dlp_enabled"] = bool(body["yt_dlp_enabled"])
    if "yt_dlp_quality" in body:
        quality = str(body["yt_dlp_quality"] or "").strip().lower()
        if quality not in ("high", "medium", "small"):
            raise HTTPException(400, "yt_dlp_quality must be high | medium | small")
        values["yt_dlp_quality"] = quality
    ADVANCED_SETTINGS = values
    _save_advanced_settings(values)
    return await get_advanced_settings()


@app.post("/api/model")
async def set_model(body: dict):
    global ALIGN_MODEL_SIZE, VERIFY_MODEL_SIZE, DEVICE_PREF, ENGINE_PREF
    global _align_model, _verify_model, _align_model_id, _verify_model_id

    changed_device = False
    changed_engine = False
    queued_models: list[dict] = []

    if "engine" in body:
        engine = str(body["engine"]).strip()
        if engine not in WHISPER_ENGINES:
            raise HTTPException(400, f"engine must be one of: {', '.join(WHISPER_ENGINES)}")
        if ENGINE_PREF != engine:
            ENGINE_PREF = engine
            _write_pref(".engine_pref", engine)
            changed_engine = True

    if "device_pref" in body:
        pref = str(body["device_pref"]).strip()
        if pref not in ("auto", "cpu", "cuda"):
            raise HTTPException(400, "device_pref must be auto | cpu | cuda")
        if DEVICE_PREF != pref:
            DEVICE_PREF = pref
            _write_pref(".device_pref", pref)
            changed_device = True

    if "align_model" in body or "model" in body:
        name = str(body.get("align_model") or body.get("model", "")).strip()
        if name not in SYNC_MODELS:
            raise HTTPException(400, f"Unknown sync model '{name}'")
        if ALIGN_MODEL_SIZE != name:
            ALIGN_MODEL_SIZE = name
            _align_model = None
            _align_model_id = None
            _write_pref(".model_pref_align", name)
            _write_pref(".model_pref", name)
        if (name in MODEL_SPECS and not managed_model_installed(name)) or (name == "hubert-fa-combined" and not hubert_installed()):
            tid = await _enqueue_model_download(name)
            if tid:
                queued_models.append({"model_id": name, "task_id": tid})

    if "verify_model" in body or "transcribe_model" in body:
        name = str(body.get("verify_model") or body.get("transcribe_model", "")).strip()
        if name not in TRANSCRIBE_MODELS:
            raise HTTPException(400, f"Unknown transcription model '{name}'")
        if VERIFY_MODEL_SIZE != name:
            VERIFY_MODEL_SIZE = name
            _verify_model = None
            _verify_model_id = None
            _write_pref(".model_pref_verify", name)
        if name in MODEL_SPECS and not managed_model_installed(name):
            # Dependencies are enqueued first by _enqueue_model_download.
            tid = await _enqueue_model_download(name)
            if tid:
                queued_models.append({"model_id": name, "task_id": tid})

    if changed_device or changed_engine:
        _release_models()

    managed_selected = ALIGN_MODEL_SIZE in MODEL_SPECS or VERIFY_MODEL_SIZE in MODEL_SPECS
    return {
        "align_model": ALIGN_MODEL_SIZE,
        "verify_model": VERIFY_MODEL_SIZE,
        "device_pref": DEVICE_PREF,
        "engine": ENGINE_PREF,
        "device": "cuda" if (_torch_cuda_usable() if managed_selected else _cuda_usable()) else "cpu",
        "align_loaded": _align_model is not None and _align_model_id == ALIGN_MODEL_SIZE,
        "verify_loaded": _verify_model is not None and _verify_model_id == VERIFY_MODEL_SIZE,
        "sync_models": SYNC_MODELS,
        "transcribe_models": TRANSCRIBE_MODELS,
        "managed_models": _all_catalog_status(),
        "queued_models": queued_models,
    }


@app.get("/api/radio/random")
async def radio_random(
    no_lyrics: bool = False,
    no_synced: bool = False,
    missing_either: bool = False,
    category: str = "",
):
    """Returns a random song, retrying until filter conditions are met (up to 50 attempts).
    no_lyrics:      only songs with no lyrics
    no_synced:      only songs with no synced_lyrics
    missing_either: only songs missing at least one of lyrics / synced_lyrics
    category:       filter by category string (released|unreleased|unsurfaced|recording_session)
    Combining no_lyrics+no_synced = missing both (AND logic).
    """
    for _ in range(50):
        data = await jw_get("/radio/random/")
        song = data.get("song") or {}
        if no_lyrics and song.get("lyrics"):
            continue
        if no_synced and song.get("synced_lyrics"):
            continue
        if missing_either and song.get("lyrics") and song.get("synced_lyrics"):
            continue  # skip songs that have BOTH; keep songs missing at least one
        if category and song.get("category") != category:
            continue
        return data
    raise HTTPException(404, "No matching song found after 50 attempts — try less restrictive filters.")


@app.get("/api/stream")
async def stream_audio(path: str, request: Request):
    # Use httpx params so the path value is properly URL-encoded
    # (paths can contain '&', spaces, etc. that would break the query string)
    req_headers = {}
    if "range" in request.headers:
        req_headers["Range"] = request.headers["range"]

    client = httpx.AsyncClient(timeout=None)
    try:
        upstream_req = client.build_request(
            "GET", BASE + "/files/download/",
            params={"path": path},
            headers=req_headers,
        )
        upstream = await client.send(upstream_req, stream=True)

        resp_headers = {"Accept-Ranges": "bytes"}
        for key in ("content-type", "content-length", "content-range"):
            if key in upstream.headers:
                resp_headers[key] = upstream.headers[key]
        if "content-type" not in resp_headers:
            resp_headers["content-type"] = "audio/mpeg"

        async def gen():
            try:
                async for chunk in upstream.aiter_bytes(32_768):
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(gen(), status_code=upstream.status_code, headers=resp_headers)
    except Exception as e:
        await client.aclose()
        raise HTTPException(502, f"Upstream error: {e}")


class SyncRequest(BaseModel):
    song_id: int
    lyrics: str = ""   # optional override; if set, used instead of song.lyrics


class VerifyRequest(BaseModel):
    song_id: int
    lyrics: str = ""


def _normalize_words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9'\s]", "", text.lower()).split()


def _verify_lines(lyric_lines: list[str], transcription: str) -> list[dict]:
    """Score each lyric line against the free transcription by word overlap."""
    trans_words = set(_normalize_words(transcription))
    results = []
    for line in lyric_lines:
        stripped = line.strip()
        if not stripped:
            continue
        words = _normalize_words(stripped)
        if not words:
            continue
        score = sum(1 for w in words if w in trans_words) / len(words)
        status = "present" if score >= 0.55 else ("uncertain" if score >= 0.25 else "absent")
        results.append({"line": stripped, "status": status, "score": round(score, 2)})
    return results


TTML_NS = "http://www.w3.org/ns/ttml"
TTM_NS = "http://www.w3.org/ns/ttml#metadata"
ITUNES_NS = "http://music.apple.com/lyric-ttml-internal"
XML_NS = "http://www.w3.org/XML/1998/namespace"


def _ttml_time(value: float) -> str:
    value = max(0.0, float(value))
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = value % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sanitize_timed_lines(
    lines: list[dict],
    min_duration: float = 0.001,
    *,
    inline_parenthetical_background: bool = True,
) -> list[dict]:
    """Validate timestamps without squeezing genuine silence out of them.

    Stable-ts/CTranslate2 can intentionally leave short or long gaps between
    words. Those gaps are meaningful, so TTML/Whisper timing is preserved as
    measured instead of forcing every word to touch the next one. The 1 ms
    non-overlap workaround is now confined to the legacy LRC parser in the UI.
    """
    cleaned: list[dict] = []
    min_duration = max(0.0001, float(min_duration))

    for raw in lines:
        text = str(raw.get("line", ""))
        start = max(0.0, _safe_float(raw.get("start")))
        candidate_end = _safe_float(raw.get("end"), start + 0.5)

        valid_words: list[dict] = []
        in_parenthetical_bg = False
        stripped_line = text.strip()
        whole_line_parenthetical = len(stripped_line) >= 2 and stripped_line.startswith("(") and stripped_line.endswith(")")
        for raw_word in list(raw.get("words") or []):
            raw_text = str(raw_word.get("text") or raw_word.get("word") or "")
            word_text = str(raw_word.get("word") or raw_text).strip()
            if not word_text:
                continue
            word_start = max(0.0, _safe_float(raw_word.get("start"), start))
            word_end = _safe_float(raw_word.get("end"), word_start + min_duration)
            if word_end <= word_start:
                word_end = word_start + min_duration

            # Apple TTML supports inset background vocals with ttm:role="x-bg".
            # Keep an explicit imported flag when present, otherwise use the
            # common lyric convention where parenthesized ad-libs/backgrounds
            # are background vocals. This is only presentation metadata and
            # does not alter the model's timestamps.
            explicit_bg = raw_word.get("background")
            existing_source = str(raw_word.get("background_source") or "")
            opens_bg = "(" in word_text
            closes_bg = ")" in word_text

            if existing_source == "explicit":
                background = bool(explicit_bg)
                background_source = "explicit" if background else ""
            elif existing_source == "parenthetical-line":
                background = bool(explicit_bg)
                background_source = "parenthetical-line" if background else ""
            elif existing_source == "parenthetical-inline":
                background = bool(explicit_bg)
                background_source = "parenthetical-inline" if background else ""
            elif explicit_bg is not None:
                # Older cached/imported words may have a boolean but no source.
                # Inline parentheses are treated as auto-detected; standalone
                # parenthetical lines remain background regardless of the toggle.
                if whole_line_parenthetical:
                    background = bool(explicit_bg)
                    background_source = "parenthetical-line" if background else ""
                elif "(" in text and ")" in text:
                    background = bool(explicit_bg)
                    background_source = "parenthetical-inline" if background else ""
                else:
                    background = bool(explicit_bg)
                    background_source = "explicit" if background else ""
            elif whole_line_parenthetical:
                background = True
                background_source = "parenthetical-line"
            else:
                background = in_parenthetical_bg or opens_bg
                background_source = "parenthetical-inline" if background else ""

            item = {
                "word": word_text,
                "text": raw_text or word_text,
                "start": word_start,
                "end": word_end,
                "background": background,
                "background_source": background_source,
            }
            if "probability" in raw_word:
                item["probability"] = raw_word["probability"]
            valid_words.append(item)

            if explicit_bg is None and not whole_line_parenthetical:
                if opens_bg and not closes_bg:
                    in_parenthetical_bg = True
                if closes_bg:
                    in_parenthetical_bg = False

        # A line must contain its timed spans. Extend the paragraph boundary if
        # model rounding put the first/last word a few ms outside the segment.
        if valid_words:
            start = min(start, min(w["start"] for w in valid_words))
            candidate_end = max(candidate_end, max(w["end"] for w in valid_words))

        end = candidate_end if candidate_end > start else start + min_duration
        cleaned.append({
            "line": text,
            "start": start,
            "end": end,
            "words": valid_words,
        })

    return cleaned


def _build_ttml(
    lines: list[dict],
    *,
    word_timing: bool = False,
    detect_interludes: bool = True,
    inline_parenthetical_background: bool = True,
    interlude_threshold: float = 2.0,
    language: str = "en",
) -> tuple[str, bool]:
    lines = _sanitize_timed_lines(
        lines, inline_parenthetical_background=inline_parenthetical_background
    )
    if not lines:
        raise ValueError("No timed lyric lines to encode.")

    # Apple expects every lyric line to use timed spans when Word mode is
    # declared. Imported LRC has no real word data, so remain line-timed rather
    # than inventing fake word timing.
    use_word_timing = bool(word_timing) and all(line["words"] for line in lines if line["line"].strip())

    ET.register_namespace("", TTML_NS)
    ET.register_namespace("itunes", ITUNES_NS)
    ET.register_namespace("ttm", TTM_NS)

    tt = ET.Element(
        f"{{{TTML_NS}}}tt",
        {
            f"{{{ITUNES_NS}}}timing": "Word" if use_word_timing else "Line",
            f"{{{XML_NS}}}lang": language or "en",
        },
    )
    body = ET.SubElement(tt, f"{{{TTML_NS}}}body")

    def new_lyric_div(line: dict):
        return ET.SubElement(body, f"{{{TTML_NS}}}div", {
            "begin": _ttml_time(line["start"]),
            "end": _ttml_time(line["end"]),
        })

    def add_word_span(parent, word: dict):
        span = ET.SubElement(parent, f"{{{TTML_NS}}}span", {
            "begin": _ttml_time(word["start"]),
            "end": _ttml_time(word["end"]),
        })
        span.text = word["word"]
        return span

    def effective_background(line: dict, word: dict) -> bool:
        if not word.get("background"):
            return False
        source = str(word.get("background_source") or "")
        if source == "explicit":
            return True
        if source == "parenthetical-line":
            return True
        if source == "parenthetical-inline":
            return bool(inline_parenthetical_background)
        line_text = str(line.get("line") or "").strip()
        whole_line = len(line_text) >= 2 and line_text.startswith("(") and line_text.endswith(")")
        if not whole_line and "(" in line_text and ")" in line_text:
            return bool(inline_parenthetical_background)
        return True

    def add_line(div, line: dict, block_end: float | None = None) -> None:
        # Overlapping lines can end out of order. A parent lyric div must remain
        # active through the latest child end, never shrink to a shorter newer line.
        div.set("end", _ttml_time(max(line["end"], block_end if block_end is not None else line["end"])))
        p = ET.SubElement(div, f"{{{TTML_NS}}}p", {
            "begin": _ttml_time(line["start"]),
            "end": _ttml_time(line["end"]),
        })
        if use_word_timing and line["line"].strip():
            i = 0
            while i < len(line["words"]):
                word = line["words"][i]
                if effective_background(line, word):
                    # Apple background vocals are represented by an untimed
                    # x-bg wrapper containing the real timed word/beat spans.
                    bg = ET.SubElement(p, f"{{{TTML_NS}}}span", {
                        f"{{{TTM_NS}}}role": "x-bg",
                    })
                    while i < len(line["words"]) and effective_background(line, line["words"][i]):
                        child = add_word_span(bg, line["words"][i])
                        i += 1
                        if i < len(line["words"]) and line["words"][i].get("background"):
                            child.tail = " "
                    if i < len(line["words"]):
                        bg.tail = " "
                    continue

                span = add_word_span(p, word)
                i += 1
                if i < len(line["words"]):
                    span.tail = " "
        else:
            p.text = line["line"]

    threshold = max(0.0, float(interlude_threshold))
    lyric_div = new_lyric_div(lines[0])
    active_block_end = lines[0]["end"]
    for i, line in enumerate(lines):
        active_block_end = max(active_block_end, line["end"])
        add_line(lyric_div, line, active_block_end)
        if not detect_interludes or i + 1 >= len(lines):
            continue

        nxt = lines[i + 1]
        gap_start = active_block_end
        gap_end = nxt["start"]
        if gap_end - gap_start >= threshold:
            # Apple explicitly defines Instrumental as a song-part value. Start
            # only after every overlapping foreground/background vocal has ended.
            ET.SubElement(body, f"{{{TTML_NS}}}div", {
                "begin": _ttml_time(gap_start),
                "end": _ttml_time(gap_end),
                f"{{{ITUNES_NS}}}song-part": "Instrumental",
            })
            lyric_div = new_lyric_div(nxt)
            active_block_end = nxt["end"]

    # Pretty indentation inside word-timed <p> elements would introduce
    # untimed whitespace nodes between child spans, so keep Word TTML compact.
    if not use_word_timing:
        ET.indent(tt, space="  ")
    return ET.tostring(tt, encoding="unicode"), use_word_timing


class ProposeRequest(BaseModel):
    song_id: int
    lines: list[dict]
    token: str
    plain_lyrics: str = ""
    word_timing: bool = False
    detect_interludes: bool = True
    inline_parenthetical_background: bool = True
    allow_overlapping_lyrics: bool = False
    interlude_threshold: float = 2.0


@app.post("/api/propose")
async def propose_lyrics(req: ProposeRequest):
    if not req.token:
        raise HTTPException(401, "No auth token provided.")

    # TTML is the canonical synced-lyrics format. It retains real line end
    # times and, when available/enabled, stable-ts per-word timings.
    ttml, used_word_timing = _build_ttml(
        req.lines,
        word_timing=req.word_timing,
        detect_interludes=req.detect_interludes,
        inline_parenthetical_background=req.inline_parenthetical_background,
        interlude_threshold=req.interlude_threshold,
    )

    # Fetch song name for the proposal title
    song = await jw_get(f"/songs/{req.song_id}/")

    # follow_redirects=True so any 301/302 on the URL is followed, but also
    # preserve the method (httpx sends a new POST on 307/308 redirects).
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        r = await client.post(
            BASE + "/accounts/editor/proposals/",
            headers={
                "Authorization": f"Token {req.token}",
                "Content-Type": "application/json",
            },
            json={
                "change_type": "update",
                "song": req.song_id,
                "title": song.get("name", str(req.song_id)),
                "editor_notes": f"Synced lyrics generated with WRLD Sync (Apple TTML, {'word' if used_word_timing else 'line'} timing)",
                "proposed_data": {
                    "synced_lyrics": ttml,
                    **({"lyrics": req.plain_lyrics} if req.plain_lyrics.strip() else {}),
                },
            },
        )
        body = r.text
        if r.status_code == 403:
            raise HTTPException(403, "Token rejected — editor role required.")
        if r.status_code == 405:
            raise HTTPException(405, f"API returned 405 Method Not Allowed. Your account may not have editor permissions, or the endpoint changed. Response: {body}")
        if not r.is_success:
            raise HTTPException(r.status_code, f"API error {r.status_code}: {body}")
        return r.json()


GENIUS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0 Safari/537.36"
    )
}

def _parse_genius_page(html: str) -> str:
    """Extract and clean lyrics from a Genius page HTML string."""
    soup = BeautifulSoup(html, "html.parser")
    containers = soup.find_all("div", attrs={"data-lyrics-container": "true"})
    if not containers:
        raise ValueError("Couldn't find a lyrics container — Genius may have changed their page structure.")

    lines = []
    for container in containers:
        for br in container.find_all("br"):
            br.replace_with("\n")
        lines.append(container.get_text())

    raw = "\n".join(lines)

    # Remove any line that contains a bracketed annotation e.g. [Chorus], [Verse 1]
    bracket_re = re.compile(r"\[.*?\]")
    cleaned = "\n".join(line for line in raw.splitlines() if not bracket_re.search(line))
    return cleaned.strip()


@app.get("/api/genius")
async def genius_lyrics(title: str = "", artist: str = "Juice WRLD", url: str = ""):
    """Fetch lyrics from Genius. Pass `url` to use a specific page; otherwise searches by title+artist."""
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            if url:
                # Direct URL fetch — skip search
                if "genius.com" not in url:
                    raise HTTPException(400, "URL must be a genius.com link.")
                song_url = url
                song_title = url.split("/")[-1].replace("-lyrics", "").replace("-", " ").title()
            else:
                # Search by title + artist
                query = f"{artist} {title}".strip()
                r = await client.get(
                    "https://genius.com/api/search/song",
                    params={"q": query},
                    headers=GENIUS_HEADERS,
                )
                r.raise_for_status()
                sections = r.json().get("response", {}).get("sections", [])
                hits = sections[0].get("hits", []) if sections else []
                if not hits:
                    raise HTTPException(404, f"No Genius results for '{query}'")
                song_url   = hits[0]["result"]["url"]
                song_title = hits[0]["result"]["full_title"]

            r2 = await client.get(song_url, headers=GENIUS_HEADERS)
            r2.raise_for_status()

            lyrics = _parse_genius_page(r2.text)
            return {"lyrics": lyrics, "title": song_title, "url": song_url}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(502, f"Genius fetch failed: {e}")


@app.post("/api/verify")
async def verify_lyrics_audio(req: VerifyRequest):
    """SSE: free-transcribe audio, compare each lyric line by word overlap."""

    async def event_stream():
        tmp_path = None

        def sse(data: dict) -> str:
            return f"data: {json.dumps(data)}\n\n"

        try:
            yield sse({"stage": "fetching", "msg": "Fetching song info…"})
            song = await jw_get(f"/songs/{req.song_id}/")
            if not song.get("path"):
                yield sse({"stage": "error", "msg": "No audio file for this song."})
                return

            lyrics = (req.lyrics or song.get("lyrics") or "").strip()
            if not lyrics:
                yield sse({"stage": "error", "msg": "No lyrics to verify against. Add lyrics first."})
                return

            tmp_path = None
            async for ev in ensure_audio(song["path"]):
                if "_path" in ev:
                    tmp_path = ev["_path"]
                else:
                    yield sse(ev)

            yield sse({"stage": "loading", "msg": "Loading Whisper model…"})
            model = await get_model()

            yield sse({"stage": "transcribing", "pct": 0, "msg": "Transcribing audio (free pass)…"})
            loop = asyncio.get_event_loop()
            spy = _ProgressSpy()
            spy.silent = True

            def _run_verify():
                orig = sys.stderr
                sys.stderr = _TeeStderr(spy, orig)
                try:
                    return model.transcribe(tmp_path, verbose=False, word_timestamps=False, suppress_silence=False, regroup=False)
                finally:
                    sys.stderr = orig

            fut = loop.run_in_executor(None, _run_verify)
            elapsed = 0
            while not fut.done():
                prog = spy.latest()
                if prog:
                    pct = 42 + prog['pct'] * 0.53
                    msg = (f"{prog['label']}: {prog['pct']}%  "
                           f"{prog['done']:.1f}/{prog['total']:.1f}s  "
                           f"[{prog['elapsed']}<{prog['eta']}, {prog['speed']:.2f}s/sec]")
                else:
                    pct = min(41, int((elapsed / 180) ** 0.5 * 41))
                    msg = f"Transcribing… {elapsed}s"
                yield sse({"stage": "transcribing", "pct": pct, "msg": msg,
                           **({"progress": prog} if prog else {})})
                await asyncio.sleep(1)
                elapsed += 1
            result = await fut

            transcription = result.text or ""
            lyric_lines = [l for l in lyrics.split("\n") if l.strip()]
            line_results = _verify_lines(lyric_lines, transcription)

            counts = {"present": 0, "uncertain": 0, "absent": 0}
            for r2 in line_results:
                counts[r2["status"]] += 1

            yield sse({"stage": "done", "results": line_results,
                       "counts": counts, "transcription": transcription})

        except Exception as e:
            yield sse({"stage": "error", "msg": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/sync")
async def sync_lyrics(req: SyncRequest):
    """SSE endpoint — streams progress events then a final 'done' event with lines."""

    async def event_stream():
        tmp_path = None

        def sse(data: dict) -> str:
            return f"data: {json.dumps(data)}\n\n"

        try:
            # Stage 1: fetch metadata
            yield sse({"stage": "fetching", "msg": "Fetching song info…"})
            song = await jw_get(f"/songs/{req.song_id}/")

            if not song.get("path"):
                yield sse({"stage": "error", "msg": "No audio file for this song."})
                return

            # Stage 2: download (cached — skipped if same song was already downloaded)
            tmp_path = None
            async for ev in ensure_audio(song["path"]):
                if "_path" in ev:
                    tmp_path = ev["_path"]
                else:
                    yield sse(ev)

            # Stage 3: load model (no-op if already cached)
            yield sse({"stage": "loading", "msg": "Loading Whisper model…"})
            model = await get_model()

            # Stage 4: align / transcribe
            # Run in thread pool and stream tick events every second while waiting,
            # since the executor blocks the generator and tqdm only prints to terminal.
            # Custom lyrics (from Genius / manual paste) override song.lyrics
            lyrics = (req.lyrics or song.get("lyrics") or "").strip()
            label = "Aligning lyrics to audio" if lyrics else "Transcribing audio"
            loop = asyncio.get_event_loop()
            spy = _ProgressSpy()
            spy.silent = True

            def _run_sync():
                orig = sys.stderr
                sys.stderr = _TeeStderr(spy, orig)
                try:
                    if lyrics:
                        return _align(model, tmp_path, lyrics)
                    else:
                        return model.transcribe(tmp_path, word_timestamps=True, verbose=False)
                finally:
                    sys.stderr = orig

            fut = loop.run_in_executor(None, _run_sync)

            elapsed = 0
            while not fut.done():
                prog = spy.latest()
                if prog:
                    pct = 55 + prog['pct'] * 0.44
                    msg = (f"{prog['label']}: {prog['pct']}%  "
                           f"{prog['done']:.1f}/{prog['total']:.1f}s  "
                           f"[{prog['elapsed']}<{prog['eta']}, {prog['speed']:.2f}s/sec]")
                else:
                    pct = min(54, int((elapsed / 180) ** 0.5 * 54))
                    msg = f"{label}… {elapsed}s"
                yield sse({"stage": "aligning", "pct": pct, "msg": msg,
                           **({"progress": prog} if prog else {})})
                await asyncio.sleep(1)
                elapsed += 1

            result = await fut
            lines = _lines_from_alignment(result, lyrics)

            yield sse({"stage": "done", "lines": lines})

        except Exception as e:
            yield sse({"stage": "error", "msg": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Local audio files / URLs
# ---------------------------------------------------------------------------

def _duration_label(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds or 0))))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _audio_content_hash(path: pathlib.Path) -> str:
    """SHA-256 decoded audio only, excluding tags, artwork, and container metadata."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
        "-vn", "-sn", "-dn", "-map_metadata", "-1",
        "-ac", "2", "-ar", "48000", "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    digest = hashlib.sha256()
    assert proc.stdout is not None
    while True:
        chunk = proc.stdout.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
    code = proc.wait()
    if code != 0:
        raise ValueError(f"Could not decode audio for hashing: {stderr.strip() or 'ffmpeg failed'}")
    return digest.hexdigest()


def _read_local_metadata(path: pathlib.Path, fallback_name: str) -> dict:
    title = path.stem
    artist = "Unknown artist"
    duration = 0.0
    cover_blob = None
    cover_mime = ""
    try:
        from mutagen import File as MutagenFile

        easy = MutagenFile(path, easy=True)
        if easy is not None:
            if getattr(easy, "info", None) is not None:
                duration = float(getattr(easy.info, "length", 0.0) or 0.0)
            tags = getattr(easy, "tags", None) or {}
            raw_title = tags.get("title") or []
            raw_artist = tags.get("artist") or []
            if raw_title:
                title = str(raw_title[0]).strip() or title
            if raw_artist:
                artist = str(raw_artist[0]).strip() or artist

        raw = MutagenFile(path, easy=False)
        if raw is not None:
            if not duration and getattr(raw, "info", None) is not None:
                duration = float(getattr(raw.info, "length", 0.0) or 0.0)
            tags = getattr(raw, "tags", None)
            if tags is not None:
                # ID3 / MP3
                getall = getattr(tags, "getall", None)
                if callable(getall):
                    pics = getall("APIC")
                    if pics:
                        cover_blob = bytes(pics[0].data)
                        cover_mime = str(getattr(pics[0], "mime", "") or "image/jpeg")
                # MP4/M4A
                if cover_blob is None and hasattr(tags, "get"):
                    covers = tags.get("covr") or []
                    if covers:
                        cover_blob = bytes(covers[0])
                        imageformat = getattr(covers[0], "imageformat", None)
                        cover_mime = "image/png" if imageformat == 14 else "image/jpeg"
            # FLAC
            pictures = getattr(raw, "pictures", None) or []
            if cover_blob is None and pictures:
                cover_blob = bytes(pictures[0].data)
                cover_mime = str(getattr(pictures[0], "mime", "") or "image/jpeg")
    except Exception as exc:
        CONSOLE.print(f"[yellow]Could not read tags for {_display_name(fallback_name)}: {exc}[/yellow]")

    if not duration:
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
                capture_output=True, text=True, timeout=15,
            )
            if probe.returncode == 0:
                duration = float(probe.stdout.strip() or 0.0)
        except Exception:
            pass

    return {
        "title": title or pathlib.Path(fallback_name).stem or "Local audio",
        "artist": artist or "Unknown artist",
        "duration": max(0.0, duration),
        "cover_blob": cover_blob,
        "cover_mime": cover_mime,
    }


def _register_local_track(path: pathlib.Path, original_name: str, source_url: str = "") -> dict:
    track_hash = _audio_content_hash(path)
    meta = _read_local_metadata(path, original_name)
    now = time.time()
    with _db_connect() as conn:
        conn.execute(
            """
            INSERT INTO local_tracks (
                track_hash, original_name, source_url, file_path, title, artist,
                duration, cover_mime, cover_blob, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(track_hash) DO UPDATE SET
                original_name=excluded.original_name,
                source_url=CASE WHEN excluded.source_url != '' THEN excluded.source_url ELSE local_tracks.source_url END,
                file_path=excluded.file_path, title=excluded.title, artist=excluded.artist,
                duration=excluded.duration, cover_mime=excluded.cover_mime,
                cover_blob=COALESCE(excluded.cover_blob, local_tracks.cover_blob), updated_at=excluded.updated_at
            """,
            (
                track_hash, original_name, source_url, str(path), meta["title"], meta["artist"],
                meta["duration"], meta["cover_mime"], meta["cover_blob"], now, now,
            ),
        )
    return {
        "track_hash": track_hash,
        "path": str(path),
        "name": original_name,
        "title": meta["title"],
        "artist": meta["artist"],
        "duration": meta["duration"],
        "duration_label": _duration_label(meta["duration"]),
        "category": "local_file",
        "cover_url": f"/api/local/{track_hash}/cover" if meta["cover_blob"] else "",
        "audio_url": f"/api/local/{track_hash}/audio",
        "processed": _get_local_processed(track_hash),
    }


@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Accept local audio, register it by decoded-audio hash, and return tags/artwork."""
    suffix = pathlib.Path(file.filename or "audio").suffix or ".mp3"
    uid = str(uuid.uuid4())[:8]
    dest = UPLOAD_DIR / f"{uid}{suffix}"
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                out.write(chunk)
        return await asyncio.to_thread(_register_local_track, dest, file.filename or "local file", "")
    except ValueError as exc:
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(400, str(exc))
    except Exception:
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _valid_remote_url(url: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Enter a valid http:// or https:// audio URL.")
    return parsed


def _yt_dlp_service(url: str) -> str:
    try:
        host = (_valid_remote_url(url).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""
    if host in {"youtu.be", "youtube.com", "m.youtube.com", "music.youtube.com"} or host.endswith(".youtube.com"):
        return "YouTube Music" if "music.youtube.com" in host else "YouTube"
    if host == "soundcloud.com" or host.endswith(".soundcloud.com"):
        return "SoundCloud"
    return ""


def _yt_dlp_format(quality: str) -> str:
    return {
        "high": "bestaudio/best",
        "medium": "bestaudio[abr<=192]/bestaudio/best",
        "small": "bestaudio[abr<=96]/bestaudio/best",
    }.get(str(quality or "").lower(), "bestaudio/best")


def _download_with_yt_dlp(url: str, *, quality: str = "high") -> tuple[pathlib.Path, str]:
    _valid_remote_url(url)
    uid = uuid.uuid4().hex[:12]
    output_template = str(UPLOAD_DIR / f"{uid}.%(ext)s")
    exe = shutil.which("yt-dlp")
    cmd = [exe] if exe else [sys.executable, "-m", "yt_dlp"]
    cmd += [
        "--no-playlist", "--no-progress", "--no-warnings",
        "--max-filesize", "2G",
        "-f", _yt_dlp_format(quality),
        "-x", "--audio-format", "flac",
        "-o", output_template,
        "--print", "after_move:filepath",
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or "yt-dlp extraction failed").strip()
        raise ValueError(message[-1200:])
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    path = pathlib.Path(lines[-1]) if lines else UPLOAD_DIR / f"{uid}.flac"
    if not path.is_file():
        matches = sorted(UPLOAD_DIR.glob(f"{uid}.*"), key=lambda x: x.stat().st_mtime, reverse=True)
        path = next((x for x in matches if x.is_file()), path)
    if not path.is_file():
        raise ValueError("yt-dlp completed but did not leave a usable audio file.")
    service = _yt_dlp_service(url) or "Remote media"
    return path, f"{service} audio"


async def _download_direct_url(url: str) -> tuple[pathlib.Path, str, str]:
    parsed = _valid_remote_url(url)
    name = urllib.parse.unquote(pathlib.PurePosixPath(parsed.path).name) or "remote-audio"
    suffix = pathlib.Path(name).suffix
    uid = uuid.uuid4().hex[:12]
    dest = UPLOAD_DIR / f"{uid}{suffix or '.audio'}"
    max_bytes = 2 * 1024 * 1024 * 1024
    downloaded = 0
    timeout = httpx.Timeout(30.0, read=180.0, write=30.0, pool=30.0)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                disposition = response.headers.get("content-disposition", "")
                match = re.search(r"filename\*?=(?:UTF-8''|\")?([^\";]+)", disposition, re.I)
                if match:
                    name = urllib.parse.unquote(match.group(1).strip())
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if content_type in {"text/html", "application/xhtml+xml"}:
                    raise ValueError("URL returned a webpage instead of direct audio.")
                if not suffix:
                    suffix = mimetypes.guess_extension(content_type) or ".audio"
                    dest = dest.with_suffix(suffix)
                with dest.open("wb") as out:
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            raise HTTPException(413, "Remote audio is larger than the 2 GB local-file limit.")
                        out.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return dest, name, url


async def _fetch_remote_audio(url: str, *, quality: str | None = None, allow_ytdlp: bool | None = None) -> tuple[pathlib.Path, str, str]:
    url = str(url or "").strip()
    _valid_remote_url(url)
    use_ytdlp = ADVANCED_SETTINGS["yt_dlp_enabled"] if allow_ytdlp is None else bool(allow_ytdlp)
    quality = quality or ADVANCED_SETTINGS["yt_dlp_quality"]
    known_service = _yt_dlp_service(url)
    if known_service:
        if not use_ytdlp:
            raise ValueError(f"{known_service} links require yt-dlp extraction, which is disabled in Advanced settings.")
        path, name = await asyncio.to_thread(_download_with_yt_dlp, url, quality=quality)
        return path, name, url
    try:
        return await _download_direct_url(url)
    except ValueError:
        if not use_ytdlp:
            raise
        path, name = await asyncio.to_thread(_download_with_yt_dlp, url, quality=quality)
        return path, name, url


class LocalUrlRequest(BaseModel):
    url: str


@app.post("/api/local-url")
async def load_local_url(req: LocalUrlRequest):
    url = req.url.strip()
    try:
        path, name, source_url = await _fetch_remote_audio(url)
        return await asyncio.to_thread(_register_local_track, path, name, source_url)
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, f"Audio URL returned HTTP {exc.response.status_code}.")
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Could not download that audio URL: {exc}")
    except ValueError as exc:
        raise HTTPException(400, str(exc))


def _vocal_reference_payload(song_id: int = 0, track_hash: str = "") -> dict:
    owner = _owner_key(song_id, track_hash)
    if not owner:
        raise HTTPException(400, "A song_id or local track_hash is required.")
    ref = _get_vocal_reference(owner)
    if not ref:
        return {"attached": False, "owner_key": owner}
    return {
        "attached": True,
        "owner_key": owner,
        "source_name": ref.get("source_name") or "Vocal reference",
        "source_url": ref.get("source_url") or "",
        "updated_at": ref.get("updated_at"),
    }


@app.get("/api/vocal-reference")
async def get_vocal_reference(song_id: int = 0, track_hash: str = ""):
    return _vocal_reference_payload(song_id, track_hash)


@app.delete("/api/vocal-reference")
async def delete_vocal_reference(song_id: int = 0, track_hash: str = ""):
    owner = _owner_key(song_id, track_hash)
    if not owner:
        raise HTTPException(400, "A song_id or local track_hash is required.")
    removed = _delete_vocal_reference(owner)
    return {"attached": False, "owner_key": owner, "removed": removed}


@app.post("/api/vocal-reference/upload")
async def upload_vocal_reference(
    file: UploadFile = File(...),
    song_id: int = Form(0),
    track_hash: str = Form(""),
):
    owner = _owner_key(song_id, track_hash)
    if not owner:
        raise HTTPException(400, "A song_id or local track_hash is required.")
    suffix = pathlib.Path(file.filename or "vocals").suffix or ".audio"
    raw = UPLOAD_DIR / f"vocal-{uuid.uuid4().hex[:12]}{suffix}"
    stable = _VOCAL_REFERENCE_DIR / f"{hashlib.sha256(owner.encode('utf-8')).hexdigest()}.flac"
    try:
        with raw.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                out.write(chunk)
        await asyncio.to_thread(_normalize_audio_only, raw, stable)
        _set_vocal_reference(owner, stable, file.filename or "Uploaded vocals", "")
        return _vocal_reference_payload(song_id, track_hash)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    finally:
        raw.unlink(missing_ok=True)


class VocalReferenceUrlRequest(BaseModel):
    url: str
    song_id: int = 0
    track_hash: str = ""


@app.post("/api/vocal-reference/url")
async def vocal_reference_from_url(req: VocalReferenceUrlRequest):
    owner = _owner_key(req.song_id, req.track_hash)
    if not owner:
        raise HTTPException(400, "A song_id or local track_hash is required.")
    downloaded: pathlib.Path | None = None
    stable = _VOCAL_REFERENCE_DIR / f"{hashlib.sha256(owner.encode('utf-8')).hexdigest()}.flac"
    try:
        downloaded, source_name, source_url = await _fetch_remote_audio(req.url)
        await asyncio.to_thread(_normalize_audio_only, downloaded, stable)
        _set_vocal_reference(owner, stable, source_name, source_url)
        return _vocal_reference_payload(req.song_id, req.track_hash)
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, f"Vocal-reference URL returned HTTP {exc.response.status_code}.")
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Could not download that vocal reference: {exc}")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    finally:
        if downloaded is not None:
            downloaded.unlink(missing_ok=True)


@app.get("/api/local/{track_hash}/cover")
async def local_cover(track_hash: str):
    row = _get_local_track(track_hash, include_cover=True)
    if not row or not row.get("cover_blob"):
        raise HTTPException(404, "No embedded cover art for this track.")
    return Response(content=row["cover_blob"], media_type=row.get("cover_mime") or "image/jpeg", headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.get("/api/local/{track_hash}/audio")
async def local_audio(track_hash: str):
    row = _get_local_track(track_hash)
    if not row:
        raise HTTPException(404, "Local track not found.")
    path = pathlib.Path(row["file_path"])
    if not path.is_file():
        raise HTTPException(404, "The temporary local audio file is no longer available. Re-open it to restore playback.")
    media_type = mimetypes.guess_type(row.get("original_name") or path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type)


# ---------------------------------------------------------------------------
# Queue API
# ---------------------------------------------------------------------------

class QueueAddRequest(BaseModel):
    type: str        # "sync" | "verify" | "auto"
    song_id: int = 0
    song_name: str
    lyrics: str = ""
    local_path: str = ""  # set when syncing a local file instead of an API song
    local_hash: str = ""  # decoded-audio SHA-256 for local persistence/task matching
    fast_auto: bool = False  # legacy compatibility only; Auto is always a full transcription
    word_timing: bool = True
    detect_interludes: bool = True
    inline_parenthetical_background: bool = True
    allow_overlapping_lyrics: bool = False
    interlude_threshold: float = 2.0
    preprocess_vocals: bool | None = None
    separator_model: str = ""
    separator_target: str = ""


@app.post("/api/queue")
async def queue_add(req: QueueAddRequest):
    if req.type not in ("sync", "verify", "auto", "transcribe"):
        raise HTTPException(400, f"Unknown task type '{req.type}'")

    # Snapshot model choices at enqueue time so a settings change made while a
    # task is waiting cannot silently change which model that task will use.
    sync_model = ALIGN_MODEL_SIZE
    transcribe_model = VERIFY_MODEL_SIZE
    preprocess_vocals = ADVANCED_SETTINGS["preprocess_vocals"] if req.preprocess_vocals is None else bool(req.preprocess_vocals)
    separator_model = req.separator_model or ADVANCED_SETTINGS["separator_model"]
    separator_target = req.separator_target or ADVANCED_SETTINGS["separator_target"]
    model_tasks: list[str] = []
    required_models = _task_model_requirements(req.type, sync_model, transcribe_model)
    if preprocess_vocals:
        required_models.append(separator_model)
    for model_id in dict.fromkeys(required_models):
        tid = await _enqueue_model_download(model_id)
        if tid:
            model_tasks.append(tid)

    task = QueueTask(
        id=str(uuid.uuid4())[:8],
        type=req.type,
        song_id=req.song_id,
        song_name=req.song_name,
        lyrics=req.lyrics,
        local_path=req.local_path,
        local_hash=req.local_hash.strip().lower(),
        fast_auto=req.fast_auto,
        word_timing=req.word_timing,
        detect_interludes=req.detect_interludes,
        inline_parenthetical_background=req.inline_parenthetical_background,
        allow_overlapping_lyrics=req.allow_overlapping_lyrics,
        interlude_threshold=max(0.0, req.interlude_threshold),
        preprocess_vocals=preprocess_vocals,
        separator_model=separator_model,
        separator_target=separator_target,
        sync_model=sync_model,
        transcribe_model=transcribe_model,
    )
    _tasks[task.id] = task
    await _queue_put_once(task)
    await _q_broadcast()
    return {"task_id": task.id, "model_tasks": model_tasks}


@app.get("/api/queue")
async def queue_state():
    active   = _active_task.to_dict() if _active_task else None
    pending  = [t.to_dict() for t in _tasks.values() if t.status in ("pending", "paused")]
    done_list = [t for t in _tasks.values() if t.status in ("done", "error", "cancelled")]
    history  = [t.to_dict() for t in sorted(done_list, key=lambda t: t.created_at)][-20:]
    return {"active": active, "pending": pending, "history": history}


@app.delete("/api/queue/history")
async def clear_queue_history():
    """Remove all completed/error/cancelled tasks from the queue."""
    to_remove = [tid for tid, t in _tasks.items()
                 if t.status in ("done", "error", "cancelled")]
    for tid in to_remove:
        del _tasks[tid]
    await _q_broadcast()
    return {"cleared": len(to_remove)}


@app.post("/api/queue/{task_id}/cancel")
async def queue_cancel(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    task.cancel_requested = True
    task.pause_requested = False
    if task.status == "pending":
        task.status = "cancelled"
    elif task.status == "paused":
        task.status = "cancelled"
        if task.type == "model_download":
            if task.model_id in MODEL_SPECS:
                await asyncio.to_thread(remove_managed_model, task.model_id)
            else:
                try:
                    item = get_catalog_model(task.model_id)
                except ValueError:
                    item = {}
                if item.get("source") == "hubertfa_release":
                    await asyncio.to_thread(remove_hubertfa)
                elif item.get("source") == "audio-separator":
                    asset = _UVR_MODEL_DIR / str(item.get("asset") or item.get("runtime_id") or "")
                    asset.unlink(missing_ok=True)
    elif task.status == "running":
        task.progress = {**task.progress, "msg": "Cancelling…"}
        task.status = "cancelling"
    await _q_broadcast()
    return {"cancelled": True}


@app.post("/api/queue/{task_id}/pause")
async def pause_queue_task(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found.")
    if task.type != "model_download":
        raise HTTPException(400, "Only model downloads can be paused.")
    if task.status == "paused":
        return task.to_dict()
    if task.status == "pending":
        task.status = "paused"
        task.pause_requested = False
    elif task.status in ("running", "cancelling"):
        task.pause_requested = True
        task.cancel_requested = False
        task.progress = {**task.progress, "stage": "pausing", "msg": "Pausing after the current download chunk…"}
    else:
        raise HTTPException(409, f"Task cannot be paused from state {task.status}.")
    await _q_broadcast()
    return task.to_dict()


@app.post("/api/queue/{task_id}/resume")
async def resume_queue_task(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found.")
    if task.type != "model_download":
        raise HTTPException(400, "Only model downloads can be resumed.")
    if task.status != "paused":
        raise HTTPException(409, f"Task cannot be resumed from state {task.status}.")
    task.pause_requested = False
    task.cancel_requested = False
    task.status = "pending"
    task.progress = {**task.progress, "stage": "queued", "msg": "Resuming model download…"}
    await _queue_put_once(task)
    await _q_broadcast()
    return task.to_dict()


@app.get("/api/queue/stream")
async def queue_stream():
    if _q_cond is None:
        raise HTTPException(503, "Server not ready yet")

    async def gen():
        yield f"data: {_q_state_json}\n\n"
        while True:
            async with _q_cond:
                try:
                    await asyncio.wait_for(_q_cond.wait(), timeout=25)
                    data = _q_state_json
                except asyncio.TimeoutError:
                    data = None
            if data is not None:
                yield f"data: {data}\n\n"
            else:
                yield ": keepalive\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.middleware("http")
async def no_cache_html(request: Request, call_next):
    # Chrome heuristically caches HTML served with only Last-Modified, which
    # keeps stale copies of the UI alive across edits — force revalidation.
    response = await call_next(request)
    if request.url.path.endswith(".html") or request.url.path in ("", "/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/", StaticFiles(directory="static", html=True), name="static")
