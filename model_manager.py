from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import tempfile
import threading
import time
import urllib.parse

import httpx
from dataclasses import dataclass, field
from typing import Callable, Iterable

from model_catalog import load_catalog


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
MANIFEST_NAME = ".wrld-model.json"

ProgressCallback = Callable[[dict], None]
CancelCallback = Callable[[], bool]
PauseCallback = Callable[[], bool]


class DownloadPaused(InterruptedError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    repo_id: str
    folder: str
    kind: str
    provider: str
    revision: str | None = None
    allow_patterns: tuple[str, ...] = field(default_factory=tuple)
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""
    runtime_adapter: str = ""

    @property
    def path(self) -> pathlib.Path:
        return MODELS_DIR / self.folder


def _managed_specs_from_catalog() -> dict[str, ModelSpec]:
    specs: dict[str, ModelSpec] = {}
    for item in load_catalog()["models"]:
        if item.get("source") != "huggingface":
            continue
        model_id = str(item["id"])
        specs[model_id] = ModelSpec(
            id=model_id,
            label=str(item.get("label") or model_id),
            repo_id=str(item.get("repo_id") or ""),
            folder=str(item.get("folder") or model_id),
            kind=str(item.get("kind") or (item.get("tasks") or ["model"])[0]),
            provider=str(item.get("provider") or "huggingface"),
            revision=item.get("revision"),
            allow_patterns=tuple(item.get("allow_patterns") or ()),
            dependencies=tuple(item.get("dependencies") or ()),
            description=str(item.get("description") or ""),
            runtime_adapter=str(item.get("runtime_adapter") or ""),
        )
    return specs


MODEL_SPECS: dict[str, ModelSpec] = _managed_specs_from_catalog()

_remote_manifest_cache: dict[str, dict] = {}
_remote_manifest_lock = threading.Lock()
_verified_this_process: set[str] = set()


def get_spec(model_id: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown managed model: {model_id}") from exc


def _emit(cb: ProgressCallback | None, **payload) -> None:
    if cb:
        cb(payload)


def _matches_pattern(path: str, patterns: Iterable[str]) -> bool:
    from fnmatch import fnmatch
    return any(fnmatch(path, pat) for pat in patterns)


def _remote_manifest(spec: ModelSpec) -> dict:
    """Fetch file sizes + authoritative LFS SHA-256 values from Hugging Face."""
    cache_key = f"{spec.repo_id}@{spec.revision or 'main'}"
    with _remote_manifest_lock:
        cached = _remote_manifest_cache.get(cache_key)
    if cached:
        return cached

    from huggingface_hub import HfApi

    info = HfApi().model_info(
        spec.repo_id,
        revision=spec.revision,
        files_metadata=True,
    )
    files: list[dict] = []
    for sibling in info.siblings:
        rel = str(getattr(sibling, "rfilename", "") or "")
        if not rel or (spec.allow_patterns and not _matches_pattern(rel, spec.allow_patterns)):
            continue
        size = int(getattr(sibling, "size", 0) or 0)
        sha256 = None
        lfs = getattr(sibling, "lfs", None)
        if lfs:
            if isinstance(lfs, dict):
                sha256 = lfs.get("sha256")
                size = int(lfs.get("size") or size or 0)
            else:
                sha256 = getattr(lfs, "sha256", None)
                size = int(getattr(lfs, "size", 0) or size or 0)
        files.append({
            "path": rel,
            "size": size,
            "sha256": sha256,
            "blob_id": getattr(sibling, "blob_id", None),
        })

    manifest = {
        "repo_id": spec.repo_id,
        "revision_requested": spec.revision,
        "revision": getattr(info, "sha", None) or spec.revision or "main",
        "files": files,
        "total_size": sum(x["size"] for x in files),
    }
    with _remote_manifest_lock:
        _remote_manifest_cache[cache_key] = manifest
    return manifest


def _sha256_file(path: pathlib.Path, cancel: CancelCallback | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            if cancel and cancel():
                raise InterruptedError("Cancelled")
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _git_blob_sha1_file(path: pathlib.Path, cancel: CancelCallback | None = None) -> str:
    """Hash a normal Git/HF blob, including Git's `blob <size>\0` header."""
    size = path.stat().st_size
    h = hashlib.sha1()
    h.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as f:
        while True:
            if cancel and cancel():
                raise InterruptedError("Cancelled")
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _verify_candidate(
    candidate: pathlib.Path,
    remote: dict,
    progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
) -> tuple[bool, dict[str, str]]:
    """Verify every required file against HF metadata.

    Large LFS/Xet weights use the repository's SHA-256. Normal Git-tracked files
    use their Git blob SHA-1 (which includes the `blob <size>\0` header).
    """
    files = remote.get("files") or []
    verified_hashes: dict[str, str] = {}

    for item in files:
        path = candidate / item["path"]
        if not path.is_file():
            return False, {}
        expected_size = int(item.get("size") or 0)
        if expected_size and path.stat().st_size != expected_size:
            return False, {}

    hash_items = [x for x in files if x.get("sha256") or x.get("blob_id")]
    for idx, item in enumerate(hash_items, 1):
        if cancel and cancel():
            raise InterruptedError("Cancelled")
        rel = item["path"]
        _emit(
            progress,
            stage="verifying",
            msg=f"Verifying {pathlib.Path(rel).name}…",
            pct=(idx - 1) / max(1, len(hash_items)) * 100,
        )
        path = candidate / rel
        if item.get("sha256"):
            digest = _sha256_file(path, cancel)
            verified_hashes[rel] = digest
            if digest.lower() != str(item["sha256"]).lower():
                return False, {}
        else:
            digest = _git_blob_sha1_file(path, cancel)
            if digest.lower() != str(item["blob_id"]).lower():
                return False, {}
            # Keep a conventional SHA-256 in our local manifest after verifying
            # the authoritative Git blob identity.
            if path.stat().st_size <= 32 * 1024 * 1024:
                verified_hashes[rel] = _sha256_file(path, cancel)

    _emit(progress, stage="verifying", msg="Hashes verified", pct=100)
    return True, verified_hashes


def _write_manifest(target: pathlib.Path, spec: ModelSpec, remote: dict, hashes: dict[str, str], source: str) -> None:
    all_hashes = dict(hashes)
    # Hash small non-LFS config/tokenizer files too. This makes future local corruption
    # detectable without pretending Git blob IDs are SHA-256 values.
    for item in remote.get("files") or []:
        rel = item["path"]
        path = target / rel
        if rel not in all_hashes and path.is_file() and path.stat().st_size <= 32 * 1024 * 1024:
            all_hashes[rel] = _sha256_file(path)
    data = {
        "model_id": spec.id,
        "repo_id": spec.repo_id,
        "revision": remote.get("revision"),
        "verified_at": time.time(),
        "source": source,
        "files": [
            {
                "path": item["path"],
                "size": item.get("size", 0),
                "sha256": all_hashes.get(item["path"]) or item.get("sha256"),
                "mtime_ns": (target / item["path"]).stat().st_mtime_ns if (target / item["path"]).is_file() else 0,
            }
            for item in remote.get("files") or []
        ],
    }
    (target / MANIFEST_NAME).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _quick_installed(spec: ModelSpec) -> bool:
    target = spec.path
    manifest_path = target / MANIFEST_NAME
    if not manifest_path.is_file():
        return False
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if data.get("repo_id") != spec.repo_id:
            return False
        for item in data.get("files", []):
            p = target / item["path"]
            if not p.is_file():
                return False
            stat = p.stat()
            size = int(item.get("size") or 0)
            if size and stat.st_size != size:
                return False
            recorded_mtime = int(item.get("mtime_ns") or 0)
            if recorded_mtime and stat.st_mtime_ns != recorded_mtime:
                return False
        return True
    except Exception:
        return False


def is_installed(model_id: str) -> bool:
    return _quick_installed(get_spec(model_id))


def verify_installed(model_id: str, progress: ProgressCallback | None = None) -> bool:
    spec = get_spec(model_id)
    if model_id in _verified_this_process and _quick_installed(spec):
        return True
    if not spec.path.is_dir():
        return False
    try:
        remote = _remote_manifest(spec)
        ok, hashes = _verify_candidate(spec.path, remote, progress)
        if ok:
            _write_manifest(spec.path, spec, remote, hashes, "existing-project")
            _verified_this_process.add(model_id)
        return ok
    except Exception:
        return False


def _hf_cache_roots() -> list[pathlib.Path]:
    roots: list[pathlib.Path] = []
    for env_name in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        val = os.environ.get(env_name)
        if val:
            roots.append(pathlib.Path(val).expanduser())
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(pathlib.Path(hf_home).expanduser() / "hub")
    roots.append(pathlib.Path.home() / ".cache" / "huggingface" / "hub")
    if os.environ.get("LOCALAPPDATA"):
        roots.append(pathlib.Path(os.environ["LOCALAPPDATA"]) / "huggingface" / "hub")
    # Preserve order and remove duplicates.
    out: list[pathlib.Path] = []
    seen: set[str] = set()
    for r in roots:
        key = os.path.normcase(str(r.resolve(strict=False)))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _candidate_dirs(spec: ModelSpec, remote: dict) -> list[pathlib.Path]:
    candidates: list[pathlib.Path] = []
    repo_cache_name = "models--" + spec.repo_id.replace("/", "--")
    wanted_revision = str(remote.get("revision") or "")

    for root in _hf_cache_roots():
        repo_dir = root / repo_cache_name
        snapshots = repo_dir / "snapshots"
        if snapshots.is_dir():
            exact = snapshots / wanted_revision
            if wanted_revision and exact.is_dir():
                candidates.append(exact)
            try:
                candidates.extend(sorted((p for p in snapshots.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True))
            except OSError:
                pass

    # Common user model folders. Search only a few levels and only for the exact
    # known folder/repo basename so startup cannot turn into a whole-drive crawl.
    roots: list[pathlib.Path] = []
    home = pathlib.Path.home()
    roots.extend([home / "Models", home / "models", home / "AI", home / "Downloads", home / "Documents"])
    # A common real-world case is keeping an older checkout beside the current
    # project. Look directly under sibling projects' models folders without
    # recursively crawling an entire development tree.
    try:
        for sibling in PROJECT_ROOT.parent.iterdir():
            if sibling.is_dir() and sibling.resolve(strict=False) != PROJECT_ROOT.resolve(strict=False):
                roots.append(sibling / "models")
    except OSError:
        pass
    custom = os.environ.get("WRLD_MODEL_SEARCH_PATHS", "")
    roots.extend(pathlib.Path(p).expanduser() for p in custom.split(os.pathsep) if p.strip())
    if os.name == "nt":
        for drive in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            base = pathlib.Path(f"{drive}:\\")
            if base.exists():
                roots.extend([base / "Models", base / "AI" / "models"])

    target_names = {spec.folder.lower(), spec.repo_id.rsplit("/", 1)[-1].lower()}
    for root in roots:
        if not root.is_dir():
            continue
        if root.name.lower() in target_names:
            candidates.append(root)
        try:
            for child in root.iterdir():
                if child.is_dir() and child.name.lower() in target_names:
                    candidates.append(child)
                elif child.is_dir() and (
                    root.name.lower() in {"models", "model", "ai", "checkpoints"}
                    or child.name.lower() in {"models", "model", "ai", "checkpoints", "huggingface", "hf", "qwen", "nvidia"}
                ):
                    try:
                        for grandchild in child.iterdir():
                            if grandchild.is_dir() and grandchild.name.lower() in target_names:
                                candidates.append(grandchild)
                    except OSError:
                        pass
        except OSError:
            pass

    out: list[pathlib.Path] = []
    seen: set[str] = set()
    for c in candidates:
        key = os.path.normcase(str(c.resolve(strict=False)))
        if key not in seen and c.resolve(strict=False) != spec.path.resolve(strict=False):
            seen.add(key)
            out.append(c)
    return out


def _copy_verified_candidate(
    source: pathlib.Path,
    target: pathlib.Path,
    remote: dict,
    progress: ProgressCallback | None,
    cancel: CancelCallback | None,
) -> None:
    files = remote.get("files") or []
    total = sum(int(x.get("size") or 0) for x in files) or 1
    done = 0
    target.mkdir(parents=True, exist_ok=True)
    for item in files:
        if cancel and cancel():
            raise InterruptedError("Cancelled")
        rel = item["path"]
        src = source / rel
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as rf, dst.open("wb") as wf:
            while True:
                if cancel and cancel():
                    raise InterruptedError("Cancelled")
                chunk = rf.read(8 * 1024 * 1024)
                if not chunk:
                    break
                wf.write(chunk)
                done += len(chunk)
                _emit(progress, stage="copying", msg=f"Copying existing {pathlib.Path(rel).name}…", pct=min(99.0, done / total * 100))
    _emit(progress, stage="copying", msg="Existing model copied", pct=100)


def _dir_download_bytes(path: pathlib.Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def ensure_model(
    model_id: str,
    progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
    pause: PauseCallback | None = None,
) -> pathlib.Path:
    """Ensure a managed model exists in ./models, copying a verified cache before downloading."""
    spec = get_spec(model_id)
    target = spec.path

    # A forced launcher exit can leave a partial temp model behind. The queue is
    # serial, so stale temp directories for this model are safe to remove before
    # starting a new acquisition attempt.
    for pattern in (f".{spec.folder}-copy-*",):
        for stale in MODELS_DIR.glob(pattern):
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)

    if _quick_installed(spec):
        # Full hash check once per server process, not on every status poll.
        if model_id in _verified_this_process:
            _emit(progress, stage="ready", msg=f"{spec.label} already installed", pct=100)
            return target
        _emit(progress, stage="verifying", msg=f"Checking {spec.label}…", pct=0)
        if verify_installed(model_id, progress):
            return target

    _emit(progress, stage="checking", msg=f"Checking Hugging Face metadata for {spec.label}…", pct=0)
    remote = _remote_manifest(spec)

    # Prefer an already downloaded copy from common HF/model directories.
    candidates = _candidate_dirs(spec, remote)
    for idx, candidate in enumerate(candidates, 1):
        if cancel and cancel():
            raise InterruptedError("Cancelled")
        _emit(progress, stage="searching", msg=f"Checking existing copy {idx}/{len(candidates)}…", pct=(idx - 1) / max(1, len(candidates)) * 100)
        try:
            ok, hashes = _verify_candidate(candidate, remote, progress, cancel)
        except (OSError, InterruptedError):
            if cancel and cancel():
                raise
            continue
        if not ok:
            continue
        tmp_target = pathlib.Path(tempfile.mkdtemp(prefix=f".{spec.folder}-copy-", dir=MODELS_DIR))
        try:
            _copy_verified_candidate(candidate, tmp_target, remote, progress, cancel)
            # Verify after the copy too, so a flaky disk/copy cannot silently poison the project model.
            ok2, hashes2 = _verify_candidate(tmp_target, remote, progress, cancel)
            if not ok2:
                raise RuntimeError("Copied model failed SHA-256 verification")
            _write_manifest(tmp_target, spec, remote, hashes2 or hashes, f"copied:{candidate}")
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            tmp_target.replace(target)
            _verified_this_process.add(model_id)
            _emit(progress, stage="ready", msg=f"Reused existing {spec.label}", pct=100, source="copy")
            return target
        except Exception:
            shutil.rmtree(tmp_target, ignore_errors=True)
            raise

    # Download into a persistent project-local partial directory. Each file is
    # resumed with HTTP Range so pausing or a launcher restart does not discard
    # multi-GB progress.
    partial_root = MODELS_DIR / ".partials" / spec.folder
    partial_root.mkdir(parents=True, exist_ok=True)
    revision = str(remote.get("revision") or spec.revision or "main")
    files = list(remote.get("files") or [])
    total = sum(int(x.get("size") or 0) for x in files) or 1

    def interrupt_state() -> str | None:
        if cancel and cancel():
            return "cancel"
        if pause and pause():
            return "pause"
        return None

    try:
        # Keep a running byte counter instead of recursively rescanning a multi-GB
        # partial tree after every 1 MiB chunk. Existing completed/partial bytes
        # are counted once when their file is reached and then incremented in-memory.
        done_total = 0
        timeout = httpx.Timeout(30.0, read=180.0, write=30.0, pool=30.0)
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            for item in files:
                state = interrupt_state()
                if state == "pause":
                    raise DownloadPaused("Paused")
                if state == "cancel":
                    raise InterruptedError("Cancelled")

                rel = str(item["path"])
                expected = int(item.get("size") or 0)
                final = partial_root / rel
                part = final.with_name(final.name + ".part")
                final.parent.mkdir(parents=True, exist_ok=True)

                # A complete file from a previous run can be reused directly.
                if final.is_file() and (not expected or final.stat().st_size == expected):
                    done_total += final.stat().st_size
                    _emit(progress, stage="downloading", msg=f"Reusing {pathlib.Path(rel).name}…", pct=min(99.0, done_total / total * 100), done_bytes=done_total, total_bytes=total)
                    continue
                if final.exists():
                    final.unlink(missing_ok=True)

                existing = part.stat().st_size if part.is_file() else 0
                if expected and existing > expected:
                    part.unlink(missing_ok=True)
                    existing = 0
                done_total += existing

                url_path = "/".join(urllib.parse.quote(piece, safe="") for piece in rel.split("/"))
                url = f"https://huggingface.co/{spec.repo_id}/resolve/{urllib.parse.quote(revision, safe='')}/{url_path}"
                headers = {"Range": f"bytes={existing}-"} if existing else {}
                with client.stream("GET", url, headers=headers) as response:
                    response.raise_for_status()
                    # Some origins ignore Range. Restart only this file if so.
                    if existing and response.status_code != 206:
                        part.unlink(missing_ok=True)
                        done_total -= existing
                        existing = 0
                        response.close()
                        with client.stream("GET", url) as restarted:
                            restarted.raise_for_status()
                            with part.open("wb") as out:
                                for chunk in restarted.iter_bytes(1024 * 1024):
                                    state = interrupt_state()
                                    if state == "pause":
                                        raise DownloadPaused("Paused")
                                    if state == "cancel":
                                        raise InterruptedError("Cancelled")
                                    out.write(chunk)
                                    done_total += len(chunk)
                                    _emit(progress, stage="downloading", msg=f"Downloading {pathlib.Path(rel).name}…", pct=min(99.0, done_total / total * 100), done_bytes=done_total, total_bytes=total)
                    else:
                        mode = "ab" if existing else "wb"
                        with part.open(mode) as out:
                            for chunk in response.iter_bytes(1024 * 1024):
                                state = interrupt_state()
                                if state == "pause":
                                    raise DownloadPaused("Paused")
                                if state == "cancel":
                                    raise InterruptedError("Cancelled")
                                out.write(chunk)
                                done_total += len(chunk)
                                _emit(progress, stage="downloading", msg=f"Downloading {pathlib.Path(rel).name}…", pct=min(99.0, done_total / total * 100), done_bytes=done_total, total_bytes=total)

                if expected and part.stat().st_size != expected:
                    raise RuntimeError(f"Incomplete download for {rel}: expected {expected} bytes, got {part.stat().st_size}")
                part.replace(final)

        # Ignore any old transfer bookkeeping and verify every required model file
        # before publishing the install.
        shutil.rmtree(partial_root / ".cache", ignore_errors=True)
        _emit(progress, stage="verifying", msg=f"Verifying downloaded {spec.label}…", pct=0)
        ok, hashes = _verify_candidate(partial_root, remote, progress, cancel)
        if not ok:
            raise RuntimeError(f"Downloaded {spec.label} failed repository verification")
        _write_manifest(partial_root, spec, remote, hashes, "download-resumed")
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial_root.replace(target)
        _verified_this_process.add(model_id)
        _emit(progress, stage="ready", msg=f"{spec.label} ready", pct=100, source="download")
        return target
    except DownloadPaused:
        _emit(progress, stage="paused", msg=f"Paused {spec.label}", pct=min(99.0, _dir_download_bytes(partial_root) / total * 100))
        raise
    except InterruptedError:
        # Explicit destructive cancel removes partial bytes. Pause has its own
        # path above and deliberately preserves them.
        shutil.rmtree(partial_root, ignore_errors=True)
        raise



def remove_model(model_id: str) -> dict:
    spec = get_spec(model_id)
    removed = False
    if spec.path.exists():
        shutil.rmtree(spec.path, ignore_errors=True)
        removed = True
    partial = MODELS_DIR / ".partials" / spec.folder
    if partial.exists():
        shutil.rmtree(partial, ignore_errors=True)
        removed = True
    _verified_this_process.discard(model_id)
    return {"model_id": model_id, "removed": removed}

def catalog_status() -> list[dict]:
    catalog = {str(item["id"]): dict(item) for item in load_catalog()["models"]}
    out = []
    for spec in MODEL_SPECS.values():
        item = catalog.get(spec.id, {"id": spec.id, "label": spec.label})
        item.update({
            "repo_id": spec.repo_id,
            "kind": spec.kind,
            "provider": spec.provider,
            "description": item.get("description") or spec.description,
            "installed": _quick_installed(spec),
            "path": str(spec.path),
            "dependencies": list(spec.dependencies),
            "managed": True,
        })
        out.append(item)
    return out
