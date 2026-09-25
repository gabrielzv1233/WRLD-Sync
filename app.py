from dataclasses import dataclass, field as dc_field
from contextlib import asynccontextmanager
import xml.etree.ElementTree as ET
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
from collections import deque

from processing_progress import ProcessingProgress, RuntimeHistory
from youtube_lyrics import extract_youtube_lyrics
from gap_hints import build_alignment_chunks, find_alignment_anchor, parse_lyrics_gap_hints, should_emit_leading_interlude

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
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
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
_UVR_TQDM_RE = re.compile(
    r"(?P<pct>\d{1,3})%\|.*?\|\s*"
    r"(?P<done>[\d.]+)/(?P<total>[\d.]+)\s*"
    r"\[(?P<timing>[^\]]*)\]"
)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


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


class _UVRProgressSpy:
    """Capture audio-separator's generic tqdm iteration progress."""
    def __init__(self):
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self.silent = False

    def write(self, value: str) -> int:
        clean = _ANSI_ESCAPE_RE.sub("", str(value or ""))
        for chunk in re.split(r"[\r\n]", clean):
            match = _UVR_TQDM_RE.search(chunk)
            if not match:
                continue
            timing = match.group("timing").strip()
            elapsed_eta, _, rate = timing.partition(",")
            elapsed, separator, eta = elapsed_eta.partition("<")
            update = {
                "pct": max(0, min(100, int(match.group("pct")))),
                "done": float(match.group("done")),
                "total": float(match.group("total")),
                "elapsed": elapsed.strip() or "—",
                "eta": eta.strip() if separator else "—",
                "rate": rate.strip() or "—",
                "unit": "chunks",
            }
            with self._lock:
                self._latest = update
        return len(value)

    def flush(self): pass
    def isatty(self) -> bool: return False
    def fileno(self): raise io.UnsupportedOperation("fileno")

    def latest(self) -> dict | None:
        with self._lock:
            return dict(self._latest) if self._latest else None


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
_RUNTIME_HISTORY = RuntimeHistory(_CACHE_DIR / "processing-times.json")
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
    service = _yt_dlp_service(url) or "Remote media"
    metadata_args = ["--embed-metadata"]
    if service in {"YouTube", "YouTube Music"}:
        metadata_args += [
            "--parse-metadata", "%(title|)s:%(meta_title)s",
            "--parse-metadata", "%(uploader|)s:%(meta_artist)s",
        ]
    cmd[-1:-1] = metadata_args
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


def _stop_process_tree(proc) -> None:
    """Stop only this job's subprocess tree, including its audio encoders."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            proc.kill()
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait(timeout=3)


def _separate_vocals(
    source: pathlib.Path,
    model_id: str,
    cache_path: pathlib.Path,
    progress_spy: _UVRProgressSpy | None = None,
    cancelled=lambda: False,
) -> pathlib.Path:
    """Run UVR out of process so Cancel can interrupt native GPU inference."""
    if cancelled():
        raise InterruptedError("Cancelled")
    item = _separator_catalog_item(model_id)
    asset = str(item.get("asset") or item.get("runtime_id") or "")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep temporary output on the same filesystem for atomic publication.
    with tempfile.TemporaryDirectory(prefix=".uvr-", dir=cache_path.parent) as temp_dir:
        worker = pathlib.Path(__file__).parent / "scripts" / "uvr_worker.py"
        cmd = [sys.executable, "-u", str(worker), "--source", str(source.resolve()),
               "--model", asset, "--model-dir", str(_UVR_MODEL_DIR.resolve()),
               "--output-dir", str(pathlib.Path(temp_dir).resolve())]
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {"start_new_session": True}
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)
        tail = deque(maxlen=12)

        def read_output():
            pending = ""
            while chunk := proc.stdout.read1(4096):
                text = chunk.decode("utf-8", "replace")
                tail.append(text)
                pending = (pending + text)[-8192:]
                if progress_spy:
                    progress_spy.write(pending)
                if "\r" in pending or "\n" in pending:
                    pending = re.split(r"[\r\n]", pending)[-1]

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        try:
            while proc.poll() is None:
                if cancelled():
                    raise InterruptedError("Cancelled")
                time.sleep(.1)
            reader.join(timeout=2)
            if cancelled():
                raise InterruptedError("Cancelled")
            if proc.returncode:
                raise RuntimeError("UVR separation failed: " + "".join(tail)[-3000:].strip())
            manifest = pathlib.Path(temp_dir) / "result.txt"
            if not manifest.is_file():
                raise RuntimeError("UVR separation completed without a vocals stem")
            vocal = pathlib.Path(manifest.read_text(encoding="utf-8")).resolve()
            if not vocal.is_relative_to(pathlib.Path(temp_dir).resolve()) or not vocal.is_file() or not vocal.stat().st_size:
                raise RuntimeError("UVR returned an invalid vocals stem")
            if cancelled():
                raise InterruptedError("Cancelled")
            vocal.replace(cache_path)
            return cache_path
        finally:
            _stop_process_tree(proc)
            reader.join(timeout=2)
            proc.stdout.close()


def _uvr_progress_window(task) -> tuple[int, int]:
    """Reserve an overall queue-progress range for vocal preprocessing."""
    return {
        "sync": (8, 50),
        "verify": (8, 36),
        "transcribe": (8, 28),
        "auto": (8, 26),
    }.get(str(getattr(task, "type", "") or ""), (8, 30))


async def _prepare_analysis_audio(task, source_path: str) -> str:
    if task.cancel_requested:
        raise asyncio.CancelledError()
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
    progress_start, progress_end = _uvr_progress_window(task)
    task.progress = {
        "stage": "separating", "step": "preprocessing", "pct": progress_start,
        "msg": "Checking all-vocals stem cache…",
    }
    await _q_broadcast()
    audio_hash = await asyncio.to_thread(_audio_content_hash, pathlib.Path(source_path))
    if task.cancel_requested:
        raise asyncio.CancelledError()
    config = {"separator_model": separator_model, "separator_target": "all_vocals"}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    stem_dir = _STEM_CACHE_DIR / audio_hash / config_hash
    stem_path = stem_dir / "vocals.flac"
    metadata_path = stem_dir / "metadata.json"
    if not stem_path.is_file():
        label = get_catalog_model(separator_model).get("label", separator_model)
        task.progress = {
            "stage": "separating", "step": "preprocessing", "pct": progress_start,
            "msg": f"Loading UVR · {label}…",
        }
        await _q_broadcast()
        spy = _UVRProgressSpy()
        loop = asyncio.get_running_loop()
        stop = threading.Event()
        future = loop.run_in_executor(
            None,
            lambda: _separate_vocals(pathlib.Path(source_path), separator_model, stem_path, spy,
                                     lambda: task.cancel_requested or stop.is_set()),
        )
        started = time.perf_counter()
        try:
            while not future.done():
                uvr = spy.latest()
                if uvr:
                    phase_pct = uvr["pct"]
                    overall_pct = round(progress_start + (progress_end - progress_start) * phase_pct / 100)
                    message = f"Separating all vocals with UVR… {phase_pct}%"
                else:
                    overall_pct = progress_start
                    message = f"Loading UVR · {label}… {time.perf_counter() - started:.0f}s"
                task.progress = {
                    "stage": "separating", "step": "preprocessing", "pct": overall_pct,
                    "msg": "Cancelling UVR…" if task.cancel_requested else message,
                    "indeterminate": not bool(uvr),
                    **({"uvr": uvr} if uvr else {}),
                }
                await _q_broadcast()
                await asyncio.sleep(.25)
            await future
        except (asyncio.CancelledError, InterruptedError):
            stop.set()
            # Do not advance the queue until the child exits and its temporary files are gone.
            try:
                await asyncio.shield(future)
            except (InterruptedError, asyncio.CancelledError):
                pass
            raise asyncio.CancelledError()
        if task.cancel_requested:
            raise asyncio.CancelledError()
        task.progress = {
            "stage": "separating", "step": "preprocessing", "pct": progress_end,
            "msg": "Caching all-vocals stem…",
            **({"uvr": spy.latest()} if spy.latest() else {}),
        }
        await _q_broadcast()
        metadata_path.write_text(json.dumps({"audio_hash": audio_hash, **config}, indent=2), encoding="utf-8")
    else:
        task.progress = {
            "stage": "separating", "step": "preprocessing", "pct": progress_end,
            "msg": "Using cached all-vocals stem ✓",
        }
        await _q_broadcast()
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
        "gap_hints": list(getattr(task, "gap_hints", []) or []),
        "background_vocals": "parenthetical-inline-toggle",
        "alignment": {
            "original_split": True,
            "fast_mode": fast_mode,
            "token_step": 200 if fast_mode else 100,
            "nonspeech_skip": 2.0 if fast_mode else 5.0,
            "suppress_silence": True,
            "suppress_word_ts": True,
            "gap_strategy": "literal-seconds+asr-anchor-v2" if getattr(task, "gap_hints", []) else "none",
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
            cached_file = pathlib.Path(_audio_cache["file"])
            if cached_file.is_file() and cached_file.stat().st_size > 0:
                yield {"stage": "downloading", "pct": 100, "msg": "Audio already cached ✓"}
                yield {"_path": str(cached_file)}
                return
            _audio_cache = None

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
        async with httpx.AsyncClient(timeout=180, follow_redirects=True) as client:
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
    engine = HubertFAEngine(device_pref=DEVICE_PREF)
    active = " → ".join(engine.onnx_active_providers) or "none"
    available = ", ".join(engine.onnx_available_providers) or "none"
    CONSOLE.print(
        f"[green]HuBERT ONNX[/green] · preference {DEVICE_PREF} · active {active} "
        f"[dim](available: {available})[/dim]"
    )
    return engine


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
    gap_hints: list[dict] = dc_field(default_factory=list)
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
                "gap_hints": self.gap_hints,
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

def _gap_hint_seconds(hint: dict | None) -> float:
    """Read literal seconds, including experimental strength-era metadata."""
    if not hint:
        return 0.0
    if "seconds" in hint:
        try:
            return max(0.0, float(hint.get("seconds", 0.25)))
        except (TypeError, ValueError):
            return 0.25

    # Compatibility for cached results created before literal-second syntax.
    try:
        strength = max(1, min(7, int(hint.get("strength", 1) or 1)))
    except (TypeError, ValueError):
        strength = 1
    return min(12.0, 0.25 * (2 ** (strength - 1)))


def _serialize_gap_hint(hint) -> dict:
    return {
        "position": int(hint.position),
        "seconds": max(0.0, float(hint.seconds)),
        "interlude": str(hint.interlude),
        "source": str(hint.source),
    }


def _apply_gap_hints_to_lines(lines: list[dict], gap_hints: list[dict]) -> list[dict]:
    """Attach leading/between-line controls without turning them into lyrics."""
    if not lines or not gap_hints:
        return lines
    for hint in gap_hints:
        position = int(hint.get("position", -1))
        metadata = {
            "seconds": _gap_hint_seconds(hint),
            "interlude": str(hint.get("interlude") or "auto"),
            "source": str(hint.get("source") or "[...]"),
        }
        if position == 0:
            lines[0]["gap_before"] = metadata
        elif 0 < position < len(lines):
            lines[position - 1]["gap_after"] = metadata
    return lines


def _gap_hint_guard_seconds(hint: dict | None) -> float:
    """Literal minimum time before a post-gap lyric can be considered."""
    return _gap_hint_seconds(hint)


def _gap_hint_nonspeech_skip(hint: dict | None) -> float | None:
    """Use one gap-friendly stable-ts skip threshold independent of duration."""
    return 0.5 if hint else None


def _slice_alignment_audio(source_path: str, start_seconds: float) -> str:
    """Create an accurate 16 kHz mono remainder clip beginning at start_seconds."""
    start_seconds = max(0.0, float(start_seconds))
    if start_seconds <= 0.001:
        return source_path

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sliced_path = tmp.name

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", source_path,
        "-ss", f"{start_seconds:.3f}",
        "-map", "0:a:0",
        "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        sliced_path,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            **({"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}),
        )
    except Exception:
        pathlib.Path(sliced_path).unlink(missing_ok=True)
        raise
    if proc.returncode != 0 or not pathlib.Path(sliced_path).is_file() or not pathlib.Path(sliced_path).stat().st_size:
        pathlib.Path(sliced_path).unlink(missing_ok=True)
        raise RuntimeError(proc.stderr.strip() or "Could not prepare post-gap alignment audio.")
    return sliced_path


def _offset_timed_lines(lines: list[dict], offset: float) -> list[dict]:
    offset = max(0.0, float(offset))
    if offset <= 0.0:
        return lines

    shifted: list[dict] = []
    for raw in lines:
        line = dict(raw)
        line["start"] = round(float(raw.get("start", 0.0) or 0.0) + offset, 3)
        line["end"] = round(float(raw.get("end", 0.0) or 0.0) + offset, 3)
        words = []
        for raw_word in list(raw.get("words") or []):
            word = dict(raw_word)
            word["start"] = round(float(raw_word.get("start", 0.0) or 0.0) + offset, 3)
            word["end"] = round(float(raw_word.get("end", 0.0) or 0.0) + offset, 3)
            words.append(word)
        line["words"] = words
        shifted.append(line)
    return shifted


def _flatten_timed_words(lines: list[dict]) -> list[dict]:
    """Flatten timestamped ASR output into the word stream used by gap anchors."""
    words: list[dict] = []
    for line in lines:
        timed_words = list(line.get("words") or [])
        if timed_words:
            words.extend(timed_words)
            continue

        text = str(line.get("line") or "").strip()
        if not text:
            continue
        start = float(line.get("start", 0.0) or 0.0)
        end = float(line.get("end", start) or start)
        for token in re.findall(r"\S+", text):
            words.append({"word": token, "start": start, "end": end})

    return sorted(words, key=lambda item: float(item.get("start", 0.0) or 0.0))


def _align(
    model_obj,
    tmp_path: str,
    lyrics: str,
    fast_mode: bool = False,
    progress_callback=None,
    nonspeech_skip_override: float | None = None,
):
    # original_split=True keeps one output segment per input lyric line.
    # Optional fast alignment remains available internally for compatibility.
    # Gap-hinted chunks can lower nonspeech_skip so stable-ts is willing to
    # traverse instrumental/non-vocal sections instead of compressing lyrics
    # toward the beginning of the remainder clip.
    nonspeech_skip = (
        float(nonspeech_skip_override)
        if nonspeech_skip_override is not None
        else (2.0 if fast_mode else 5.0)
    )
    return model_obj.align(
        tmp_path, lyrics, language="en",
        original_split=True,
        nonspeech_skip=nonspeech_skip,
        fast_mode=fast_mode,
        token_step=200 if fast_mode else 100,
        suppress_silence=True,
        suppress_word_ts=True,
        verbose=False,
        progress_callback=progress_callback,
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


def _processing_audio_duration(path: str) -> float:
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=nw=1:nk=1', path],
            capture_output=True, text=True, timeout=5,
            **({'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {}),
        )
        duration = float(result.stdout.strip())
        return duration if 0 < duration < float('inf') else 0.0
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0.0


async def _run_model_job(task, path, model_id, runner, *, phase, label, start, end=98, live=False, broadcast=True):
    """Poll measured stage progress without releasing the inference lock prematurely."""
    if task.cancel_requested:
        raise asyncio.CancelledError()
    duration = await asyncio.to_thread(_processing_audio_duration, path)
    stop = threading.Event()
    engine = 'torch' if model_id in MODEL_SPECS else ENGINE_PREF
    key = f'{model_id}:{engine}:{_get_device(engine)}:words={task.word_timing}'
    async with _inference_lock:
        report = ProcessingProgress(_RUNTIME_HISTORY, key, duration,
                                    lambda: task.cancel_requested or stop.is_set())
        try:
            report.begin(phase, label, start, end)
        except InterruptedError:
            raise asyncio.CancelledError()
        future = asyncio.get_running_loop().run_in_executor(None, lambda: runner(report))
        try:
            while not future.done():
                task.progress = {**report.snapshot(), 'live': live}
                if broadcast:
                    await _q_broadcast()
                await asyncio.sleep(.25)
            result = await future
            report.finish()
            task.progress = {**report.snapshot(), 'live': live}
            if broadcast:
                await _q_broadcast()
            return result
        except asyncio.CancelledError:
            stop.set()
            try:
                await asyncio.shield(future)
            except (InterruptedError, asyncio.CancelledError):
                pass
            raise
        except InterruptedError:
            raise asyncio.CancelledError()


async def _legacy_model_events(path, model_id, runner, **options):
    """Share measured progress with the legacy direct SSE endpoints."""
    task = QueueTask(uuid.uuid4().hex, 'sync', 0, '', '')
    future = asyncio.create_task(_run_model_job(task, path, model_id, runner, broadcast=False, **options))
    try:
        while not future.done():
            if task.progress:
                yield task.progress
            await asyncio.sleep(.25)
        yield {'_result': await future}
    finally:
        if not future.done():
            task.cancel_requested = True
            try:
                await asyncio.shield(future)
            except (InterruptedError, asyncio.CancelledError):
                pass


async def _faster_stream_transcribe_worker(task: QueueTask, tmp_path: str) -> list[dict] | None:
    """Stream faster-whisper segments into QueueTask.live_lines as they decode.

    Returns None when the loaded model is not a faster-whisper model, allowing
    the caller to fall back to stable-ts/PyTorch transcription.
    """
    model_id = task.transcribe_model or VERIFY_MODEL_SIZE
    model_obj = await get_verify_model(model_id)
    transcribe_original = getattr(model_obj, "transcribe_original", None)
    if not callable(transcribe_original):
        return None

    def run(report):
        lines = []
        segments, info = transcribe_original(tmp_path, language="en", word_timestamps=True)
        duration = float(getattr(info, "duration", 0.0) or report.duration)
        report.update(0, duration)
        try:
            for segment in segments:
                report.check_cancel()
                line = _line_from_faster_segment(segment)
                if line:
                    lines.append(line)
                    task.live_lines = list(lines)
                report.update(float(segment.end), duration)
        finally:
            close = getattr(segments, "close", None)
            if close:
                close()
        return lines

    lines = await _run_model_job(
        task, tmp_path, model_id, run, phase="transcribing", label="Transcribing live…",
        start=42 if task.type == "verify" else 30, live=True,
    )
    return _sanitize_timed_lines(lines, inline_parenthetical_background=task.inline_parenthetical_background)


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

    def run_alignment(report):
        report.check_cancel()
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

    timestamps = await _run_model_job(
        task, tmp_path, task.sync_model or ALIGN_MODEL_SIZE, run_alignment,
        phase="aligning", label="Qwen forced alignment…", start=55,
    )
    return _sanitize_timed_lines(
        _align_items_to_lyric_lines(timestamps, lyrics),
        inline_parenthetical_background=task.inline_parenthetical_background,
    )


async def _hubertfa_sync_worker(task: QueueTask, tmp_path: str, lyrics: str) -> list[dict]:
    runtime = await get_align_model("hubert-fa-combined")

    def run(report):
        phases = {
            "preparing": (55, 58, "Preparing HuBERT phonemes…"),
            "inference": (58, 72, "HuBERT audio inference…"),
            "decoding": (72, 80, "Aligning phonemes…"),
            "candidates": (80, 92, "Scoring pronunciations…"),
            "finalizing": (92, 98, "Finalizing word alignment…"),
        }
        def update(phase, done=None, total=None):
            start, end, label = phases[phase]
            stage = f"aligning_{phase}"
            if report.phase != stage:
                report.begin(stage, label, start, end, total=total, unit="words")
            if total:
                report.update(done or 0, total)
        return runtime.align(tmp_path, lyrics, progress=update)

    timestamps = await _run_model_job(
        task, tmp_path, "hubert-fa-combined", run,
        phase="aligning_preparing", label="Preparing HuBERT phonemes…", start=55, end=58,
    )

    return _sanitize_timed_lines(
        _align_items_to_lyric_lines(timestamps, lyrics),
        inline_parenthetical_background=task.inline_parenthetical_background,
    )


async def _managed_transcribe_worker(task: QueueTask, tmp_path: str) -> list[dict] | None:
    model_id = task.transcribe_model or VERIFY_MODEL_SIZE
    if model_id not in MODEL_SPECS:
        return None
    runtime = await get_verify_model(model_id)
    label = get_managed_spec(model_id).label

    def run_qwen(report):
        report.check_cancel()
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

        report.begin("aligning", "Aligning Qwen transcription…", 76, 98)
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

    def run_parakeet(report):
        report.check_cancel()
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
    raw = await _run_model_job(
        task, tmp_path, model_id, runner, phase="transcribing", label=f"{label} transcription…",
        start=42 if task.type == "verify" else 30,
        end=76 if model_id.startswith("qwen3-asr-") else 98,
    )

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


async def _whisper_sync_worker(
    task: QueueTask,
    tmp_path: str,
    lyrics: str,
    fast_mode: bool = False,
    nonspeech_skip_override: float | None = None,
) -> list[dict]:
    """Use stable-ts callbacks rather than parsing terminal timing strings."""
    label = "Aligning" if lyrics else "Transcribing"
    model_id = (task.sync_model or ALIGN_MODEL_SIZE) if lyrics else (task.transcribe_model or VERIFY_MODEL_SIZE)
    model_obj = await (get_align_model(model_id) if lyrics else get_verify_model(model_id))

    def run(report):
        if lyrics:
            # stable-ts only advances its alignment callback when tqdm is enabled.
            # Capture that output while publishing the numeric callback directly.
            spy = _ProgressSpy()
            spy.silent = True
            original_stderr = sys.stderr
            sys.stderr = _TeeStderr(spy, original_stderr)
            try:
                return _align(
                    model_obj, tmp_path, lyrics,
                    fast_mode=fast_mode,
                    progress_callback=report.update,
                    nonspeech_skip_override=nonspeech_skip_override,
                )
            finally:
                sys.stderr = original_stderr
        return model_obj.transcribe(tmp_path, word_timestamps=True, verbose=None, progress_callback=report.update)

    result = await _run_model_job(
        task, tmp_path, model_id, run,
        phase="aligning" if lyrics else "transcribing", label=f"{label}…",
        start=55 if lyrics else 30,
    )
    return _sanitize_timed_lines(
        _lines_from_alignment(result, lyrics),
        inline_parenthetical_background=task.inline_parenthetical_background,
    )


async def _gap_anchor_transcribe_worker(task: QueueTask, tmp_path: str) -> list[dict]:
    """Build one timestamp map used to locate every lyric section after a gap."""
    sync_model = task.sync_model or ALIGN_MODEL_SIZE
    if sync_model in WHISPER_MODELS:
        model_id = sync_model
        model_obj = await get_align_model(model_id)
    else:
        # Qwen FA and HuBERT do not free-transcribe. Whisper base is small,
        # runtime-managed, and only loaded when the user actually uses a gap.
        model_id = "base"
        model_obj = await get_verify_model(model_id)

    def run(report):
        return model_obj.transcribe(
            tmp_path,
            word_timestamps=True,
            verbose=None,
            progress_callback=report.update,
        )

    result = await _run_model_job(
        task,
        tmp_path,
        model_id,
        run,
        phase="locating_gaps",
        label="Locating lyric gap anchors…",
        start=54,
        end=68,
    )
    return _sanitize_timed_lines(
        _lines_from_alignment(result, ""),
        inline_parenthetical_background=task.inline_parenthetical_background,
    )


async def _run_sync_model_once(
    task: QueueTask,
    tmp_path: str,
    lyrics: str,
    *,
    nonspeech_skip_override: float | None = None,
) -> list[dict]:
    sync_model = task.sync_model or ALIGN_MODEL_SIZE
    if sync_model == "qwen3-forced-aligner-0.6b":
        return await _qwen_sync_worker(task, tmp_path, lyrics)
    if sync_model == "hubert-fa-combined":
        return await _hubertfa_sync_worker(task, tmp_path, lyrics)
    return await _whisper_sync_worker(
        task, tmp_path, lyrics,
        nonspeech_skip_override=nonspeech_skip_override,
    )


async def _sync_with_gap_hints(task: QueueTask, tmp_path: str, parsed_lyrics) -> list[dict]:
    """Align marker-separated lyrics with hard forward-only audio barriers."""
    chunks = build_alignment_chunks(parsed_lyrics)
    if not chunks:
        return []

    # Preserve the original one-pass path exactly when there are no alignment
    # boundaries. A trailing marker alone affects TTML policy but has no lyrics
    # after it, so it does not need a second alignment pass.
    has_alignment_boundary = any(chunk.before_gap is not None for chunk in chunks)
    if not has_alignment_boundary:
        return await _run_sync_model_once(task, tmp_path, parsed_lyrics.text)

    duration = await asyncio.to_thread(_processing_audio_duration, tmp_path)
    anchor_lines = await _gap_anchor_transcribe_worker(task, tmp_path)
    anchor_words = _flatten_timed_words(anchor_lines)
    aligned: list[dict] = []
    previous_end = 0.0
    first_chunk_to_align = 0

    # When the lyrics do not start with a gap hint, do one normal full-text pass
    # and keep only the lines before the first marker. This preserves the
    # aligner's original global context for the pre-gap lyrics while discarding
    # every timestamp after the boundary that may have been greedily compressed.
    if chunks[0].before_gap is None and len(chunks) > 1:
        baseline = await _run_sync_model_once(task, tmp_path, parsed_lyrics.text)
        first_boundary = chunks[1].start_line
        initial_lines = baseline[:first_boundary]
        if len(initial_lines) < first_boundary:
            # Backend did not preserve enough input lines. Fall back to aligning
            # only the first chunk rather than manufacturing timestamps.
            initial_lines = await _run_sync_model_once(task, tmp_path, chunks[0].text)
        if not initial_lines:
            raise ValueError("Could not align lyrics before the first gap hint.")
        aligned.extend(initial_lines)
        previous_end = max(float(line.get("end", 0.0) or 0.0) for line in initial_lines)
        first_chunk_to_align = 1

    for index in range(first_chunk_to_align, len(chunks)):
        chunk = chunks[index]
        if task.cancel_requested:
            raise asyncio.CancelledError()

        hint = _serialize_gap_hint(chunk.before_gap) if chunk.before_gap is not None else None
        search_start = previous_end if index else 0.0
        if hint is not None:
            search_start += _gap_hint_guard_seconds(hint)
            anchor_start = find_alignment_anchor(
                chunk.text,
                anchor_words,
                start_at=search_start,
            )
            if anchor_start is not None:
                # Give the forced aligner a short lead-in before the located
                # lyric onset, while never crossing back through the hard guard.
                search_start = max(search_start, anchor_start - 0.6)

        if duration > 0 and search_start >= duration - 0.05:
            raise ValueError(
                f"Gap hint before lyric line {chunk.start_line + 1} moved alignment past the end of the audio."
            )

        slice_path = tmp_path
        cleanup_slice = False
        if search_start > 0.001:
            task.progress = {
                "stage": "gap_boundary",
                "msg": f"Applying lyric gap boundary {index + 1}/{len(chunks)}…",
                "step": "aligning",
                "pct": 54,
            }
            await _q_broadcast()
            slice_path = await asyncio.to_thread(_slice_alignment_audio, tmp_path, search_start)
            cleanup_slice = True

        try:
            chunk_lines = await _run_sync_model_once(
                task,
                slice_path,
                chunk.text,
                nonspeech_skip_override=_gap_hint_nonspeech_skip(hint),
            )
        finally:
            if cleanup_slice:
                pathlib.Path(slice_path).unlink(missing_ok=True)

        if search_start > 0.0:
            chunk_lines = _offset_timed_lines(chunk_lines, search_start)
        if not chunk_lines:
            raise ValueError(f"Could not align lyrics after gap hint before line {chunk.start_line + 1}.")

        aligned.extend(chunk_lines)
        previous_end = max(previous_end, max(float(line.get("end", 0.0) or 0.0) for line in chunk_lines))

    return aligned


async def _whisper_verify_worker(task: QueueTask, tmp_path: str, lyrics: str) -> list[dict]:
    """Free-transcribe + compare, using the same measured progress as Auto."""
    model_id = task.transcribe_model or VERIFY_MODEL_SIZE
    model_v = await get_verify_model(model_id)

    def run(report):
        return model_v.transcribe(
            tmp_path, verbose=None, word_timestamps=False,
            suppress_silence=False, regroup=False, progress_callback=report.update,
        )

    result = await _run_model_job(
        task, tmp_path, model_id, run, phase="transcribing",
        label="Transcribing for verification…", start=42, end=95,
    )
    return _verify_lines([line for line in lyrics.splitlines() if line.strip()], result.text or "")


async def _download_audio(task: QueueTask, song: dict) -> str:
    """Stream audio via ensure_audio, broadcasting progress. Returns tmp_path."""
    tmp_path = None
    async for ev in ensure_audio(song["path"]):
        if task.cancel_requested:
            raise asyncio.CancelledError()
        if "_path" in ev:
            tmp_path = ev["_path"]
        else:
            task.progress = {**ev, "pct": 2 + max(0, min(100, ev.get("pct", 0) or 0)) * .06, "step": "downloading"}
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
    source_lyrics = lyrics
    parsed_lyrics = parse_lyrics_gap_hints(lyrics)
    lyrics = parsed_lyrics.text
    task.gap_hints = [_serialize_gap_hint(hint) for hint in parsed_lyrics.gaps]
    task.lyrics = lyrics
    if not lyrics:
        raise ValueError("No lyric lines remain after parsing gap controls.")
    if task.cancel_requested:
        raise asyncio.CancelledError()
    tmp_path = await _prepare_analysis_audio(task, tmp_path)
    sync_model = task.sync_model or ALIGN_MODEL_SIZE
    label = get_managed_spec(sync_model).label if sync_model in MODEL_SPECS else f"Whisper {sync_model}"
    task.progress = {"stage": "loading", "msg": f"Loading {label}…", "step": "loading", "pct": 52}
    await _q_broadcast()
    lines = await _sync_with_gap_hints(task, tmp_path, parsed_lyrics)
    lines = _apply_gap_hints_to_lines(lines, task.gap_hints)
    task.result = {
        "lines": lines,
        "text": lyrics,
        "source_text": source_lyrics,
        "gap_hints": task.gap_hints,
    }
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
            if task.cancel_requested:
                raise asyncio.CancelledError()
            if task.status in ("running", "cancelling"):   # runner didn't set error/cancelled
                task.status = "done"
            if task.status == "done" and task.result and task.result.get("lines"):
                try:
                    _store_processed_lyrics(task)
                except sqlite3.Error as exc:
                    CONSOLE.print(f"[yellow]Could not save processed lyrics for {task.song_id}: {exc}[/yellow]")
        except asyncio.CancelledError:
            task.status = "cancelled"
            task.result = None
            if asyncio.current_task().cancelling():
                raise
        except Exception as exc:
            task.status = "cancelled" if task.cancel_requested else "error"
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

    loop = asyncio.get_running_loop()
    previous_exception_handler = loop.get_exception_handler()

    def handle_asyncio_exception(active_loop, context):
        exc = context.get("exception")
        if (
            sys.platform == "win32"
            and isinstance(exc, ConnectionResetError)
            and getattr(exc, "winerror", None) == 10054
        ):
            CONSOLE.print("[dim]Client connection reset during reload (WinError 10054).[/dim]")
            return
        if previous_exception_handler is not None:
            previous_exception_handler(active_loop, context)
        else:
            active_loop.default_exception_handler(context)

    loop.set_exception_handler(handle_asyncio_exception)
    proc = asyncio.create_task(_queue_processor())
    try:
        yield
    finally:
        loop.set_exception_handler(previous_exception_handler)
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


class SearchRequest(BaseModel):
    q: str
    page_size: int = 20


async def _search_catalog(q: str, page_size: int, response: Response) -> dict:
    response.headers["Cache-Control"] = "no-store"
    clean = str(q or "").strip()
    if not clean:
        return {"count": 0, "results": []}

    data = await jw_get(
        "/songs/",
        {"search": clean, "page_size": max(1, min(100, int(page_size)))},
    )
    raw_results = list(data.get("results") or []) if isinstance(data, dict) else []
    # Search cards only need lightweight catalog metadata. Do not tunnel full
    # lyrics/synced-lyrics/version payloads for every result.
    results = [
        {
            "id": row.get("id"),
            "name": row.get("name") or "",
            "track_titles": row.get("track_titles") or [],
            "image_url": row.get("image_url") or "",
            "era": row.get("era") or {},
            "category": row.get("category") or "",
            "length": row.get("length") or "",
            "path": row.get("path") or "",
        }
        for row in raw_results
        if isinstance(row, dict)
    ]
    return {
        "count": int(data.get("count", len(results)) or len(results)) if isinstance(data, dict) else len(results),
        "results": results,
    }


@app.get("/api/search")
async def search(q: str, response: Response, page_size: int = 20):
    return await _search_catalog(q, page_size, response)


@app.post("/api/search")
async def search_post(req: SearchRequest, response: Response):
    return await _search_catalog(req.q, req.page_size, response)


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


@app.get("/api/playback")
async def playback_audio(path: str):
    """Redirect catalog playback directly to the public audio host.

    This keeps large media bytes out of VS Code/dev tunnels. The frontend falls
    back to /api/stream when the upstream browser request itself fails.
    """
    clean_path = str(path or "").strip()
    if not clean_path:
        raise HTTPException(400, "An audio path is required.")
    upstream = BASE + "/files/download/?" + urllib.parse.urlencode({"path": clean_path})
    return RedirectResponse(
        upstream,
        status_code=307,
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/stream")
async def stream_audio(path: str):
    """Serve catalog audio as a local seekable file.

    The old route live-proxied the upstream response to the browser. That made
    playback depend on every reverse proxy/tunnel preserving Range requests and
    streaming semantics. Materializing through ensure_audio() gives playback
    the same known-good bytes used by Sync/Transcribe, then FileResponse handles
    browser byte ranges locally.
    """
    clean_path = str(path or "").strip()
    if not clean_path:
        raise HTTPException(400, "An audio path is required.")

    local_path = ""
    try:
        async for event in ensure_audio(clean_path):
            if "_path" in event:
                local_path = str(event["_path"])
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            exc.response.status_code,
            f"Audio download returned HTTP {exc.response.status_code}.",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Could not download catalog audio: {exc}") from exc

    file_path = pathlib.Path(local_path)
    if not local_path or not file_path.is_file() or file_path.stat().st_size <= 0:
        raise HTTPException(502, "Catalog audio download did not produce a usable file.")

    media_type = mimetypes.guess_type(clean_path)[0] or "audio/mpeg"
    return FileResponse(
        file_path,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
        },
    )


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
        cleaned_line = {
            "line": text,
            "start": start,
            "end": end,
            "words": valid_words,
        }
        for gap_key in ("gap_before", "gap_after"):
            gap_meta = raw.get(gap_key)
            if isinstance(gap_meta, dict):
                cleaned_line[gap_key] = {
                    "seconds": _gap_hint_seconds(gap_meta),
                    "interlude": str(gap_meta.get("interlude") or "auto"),
                    "source": str(gap_meta.get("source") or "[...]"),
                }
        cleaned.append(cleaned_line)

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
    leading_hint = lines[0].get("gap_before") if isinstance(lines[0].get("gap_before"), dict) else {}
    leading_policy = str(leading_hint.get("interlude") or "auto")
    if leading_hint and should_emit_leading_interlude(
        lines[0]["start"],
        leading_policy,
        detect_interludes=detect_interludes,
    ):
        ET.SubElement(body, f"{{{TTML_NS}}}div", {
            "begin": _ttml_time(0.0),
            "end": _ttml_time(lines[0]["start"]),
            f"{{{ITUNES_NS}}}song-part": "Instrumental",
        })

    lyric_div = new_lyric_div(lines[0])
    active_block_end = lines[0]["end"]
    for i, line in enumerate(lines):
        active_block_end = max(active_block_end, line["end"])
        add_line(lyric_div, line, active_block_end)
        if i + 1 >= len(lines):
            continue

        nxt = lines[i + 1]
        gap_start = active_block_end
        gap_end = nxt["start"]
        gap_hint = line.get("gap_after") if isinstance(line.get("gap_after"), dict) else {}
        interlude_policy = str(gap_hint.get("interlude") or "auto")
        if interlude_policy == "force":
            emit_interlude = gap_end > gap_start
        elif interlude_policy == "forbid":
            emit_interlude = False
        else:
            emit_interlude = detect_interludes and gap_end - gap_start >= threshold
        if emit_interlude:
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

            def run(report):
                return model.transcribe(tmp_path, verbose=None, word_timestamps=False,
                                        suppress_silence=False, regroup=False, progress_callback=report.update)

            async for event in _legacy_model_events(tmp_path, ALIGN_MODEL_SIZE, run,
                                                    phase="transcribing", label="Transcribing…", start=42):
                if "_result" in event:
                    result = event["_result"]
                else:
                    yield sse(event)

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

            lyrics = (req.lyrics or song.get("lyrics") or "").strip()
            label = "Aligning lyrics to audio" if lyrics else "Transcribing audio"

            def run(report):
                spy = _ProgressSpy()
                spy.silent = True
                orig = sys.stderr
                sys.stderr = _TeeStderr(spy, orig)
                try:
                    if lyrics:
                        return _align(model, tmp_path, lyrics, progress_callback=report.update)
                    return model.transcribe(tmp_path, word_timestamps=True, verbose=None, progress_callback=report.update)
                finally:
                    sys.stderr = orig

            async for event in _legacy_model_events(tmp_path, ALIGN_MODEL_SIZE, run,
                                                    phase="aligning" if lyrics else "transcribing",
                                                    label=label, start=55 if lyrics else 30):
                if "_result" in event:
                    result = event["_result"]
                else:
                    yield sse(event)
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
        "source_url": source_url,
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
    service = _yt_dlp_service(url) or "Remote media"
    exe = shutil.which("yt-dlp")
    cmd = [exe] if exe else [sys.executable, "-m", "yt_dlp"]
    cmd += [
        "--no-playlist", "--no-progress", "--no-warnings",
        "--max-filesize", "2G",
        "-f", _yt_dlp_format(quality),
        "-x", "--audio-format", "flac",
        "--embed-metadata",
    ]
    if service in {"YouTube", "YouTube Music"}:
        cmd += [
            "--parse-metadata", "%(title|)s:%(meta_title)s",
            "--parse-metadata", "%(uploader|)s:%(meta_artist)s",
        ]
    cmd += [
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
        meta = await asyncio.to_thread(_register_local_track, path, name, source_url)
        if _yt_dlp_service(source_url) in {"YouTube", "YouTube Music"}:
            try:
                meta.update(await asyncio.to_thread(extract_youtube_lyrics, source_url))
                if not meta.get("lyrics"):
                    meta["lyrics_notice"] = "No YouTube subtitles available. Add lyrics or use Transcribe."
            except Exception:
                meta["lyrics_notice"] = "YouTube subtitles could not be retrieved. Add lyrics or use Transcribe."
        return meta
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


class LocalRestoreRequest(BaseModel):
    source: str


@app.post("/api/local-restore")
async def restore_local_track(req: LocalRestoreRequest):
    source = str(req.source or "").strip()
    if not source:
        raise HTTPException(400, "A local source is required.")

    with _db_connect() as conn:
        row = conn.execute(
            """
            SELECT track_hash, original_name, source_url, file_path, title, artist,
                   duration, cover_mime,
                   CASE WHEN cover_blob IS NULL THEN 0 ELSE 1 END AS has_cover
            FROM local_tracks
            WHERE track_hash = ? OR file_path = ? OR source_url = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (source.lower(), source, source),
        ).fetchone()

    if row is None:
        raise HTTPException(404, "That local source is not registered in WRLD Sync.")

    data = dict(row)
    path = pathlib.Path(data["file_path"])
    if not path.is_file():
        raise HTTPException(
            404,
            "The registered local audio file is no longer available. Re-open the original file.",
        )

    track_hash = str(data["track_hash"])
    return {
        "track_hash": track_hash,
        "path": str(path),
        "name": data["original_name"],
        "source_url": data["source_url"] or "",
        "title": data["title"],
        "artist": data["artist"],
        "duration": float(data["duration"] or 0.0),
        "duration_label": _duration_label(data["duration"]),
        "category": "local_file",
        "cover_url": f"/api/local/{track_hash}/cover" if data["has_cover"] else "",
        "audio_url": f"/api/local/{track_hash}/audio",
        "processed": _get_local_processed(track_hash),
    }


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
    media_type = (
        mimetypes.guess_type(path.name)[0]
        or mimetypes.guess_type(row.get("original_name") or "")[0]
        or "application/octet-stream"
    )
    return FileResponse(path, media_type=media_type, headers={"Accept-Ranges": "bytes"})


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
