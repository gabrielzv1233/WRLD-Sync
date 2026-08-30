from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
MANIFEST_NAME = ".wrld-model.json"

ProgressCallback = Callable[[dict], None]
CancelCallback = Callable[[], bool]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    repo_id: str
    folder: str
    kind: str  # asr | aligner
    provider: str  # qwen | parakeet
    revision: str | None = None
    allow_patterns: tuple[str, ...] = field(default_factory=tuple)
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""

    @property
    def path(self) -> pathlib.Path:
        return MODELS_DIR / self.folder


MODEL_SPECS: dict[str, ModelSpec] = {
    "qwen3-asr-0.6b": ModelSpec(
        id="qwen3-asr-0.6b",
        label="Qwen3-ASR 0.6B",
        repo_id="Qwen/Qwen3-ASR-0.6B-hf",
        folder="qwen3-asr-0.6b",
        kind="asr",
        provider="qwen",
        allow_patterns=("*.json", "*.safetensors", "*.jinja"),
        dependencies=("qwen3-forced-aligner-0.6b",),
        description="Fast Qwen song/singing transcription with Qwen word alignment.",
    ),
    "qwen3-asr-1.7b": ModelSpec(
        id="qwen3-asr-1.7b",
        label="Qwen3-ASR 1.7B",
        repo_id="Qwen/Qwen3-ASR-1.7B-hf",
        folder="qwen3-asr-1.7b",
        kind="asr",
        provider="qwen",
        allow_patterns=("*.json", "*.safetensors", "*.jinja"),
        dependencies=("qwen3-forced-aligner-0.6b",),
        description="Higher-accuracy Qwen transcription tuned for singing and songs with BGM.",
    ),
    "qwen3-forced-aligner-0.6b": ModelSpec(
        id="qwen3-forced-aligner-0.6b",
        label="Qwen3 Forced Aligner 0.6B",
        repo_id="Qwen/Qwen3-ForcedAligner-0.6B-hf",
        folder="qwen3-forced-aligner-0.6b",
        kind="aligner",
        provider="qwen",
        allow_patterns=("*.json", "*.safetensors", "*.jinja"),
        description="Qwen forced alignment for existing lyrics.",
    ),
    "parakeet-tdt-0.6b-v3": ModelSpec(
        id="parakeet-tdt-0.6b-v3",
        label="Parakeet TDT 0.6B v3",
        repo_id="nvidia/parakeet-tdt-0.6b-v3",
        folder="parakeet-tdt-0.6b-v3",
        kind="asr",
        provider="parakeet",
        # Download only the native Transformers checkpoint, not the duplicate
        # NeMo/GGUF weights that live in the same repository.
        allow_patterns=("config.json", "generation_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"),
        description="Fast Parakeet TDT transcription with native token-duration timestamps.",
    ),
}

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
) -> pathlib.Path:
    """Ensure a managed model exists in ./models, copying a verified cache before downloading."""
    spec = get_spec(model_id)
    target = spec.path

    # A forced launcher exit can leave a partial temp model behind. The queue is
    # serial, so stale temp directories for this model are safe to remove before
    # starting a new acquisition attempt.
    for pattern in (f".{spec.folder}-download-*", f".{spec.folder}-copy-*"):
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

    # Download into a temporary project-local directory, then verify and atomically publish.
    from huggingface_hub import snapshot_download

    tmp_target = pathlib.Path(tempfile.mkdtemp(prefix=f".{spec.folder}-download-", dir=MODELS_DIR))
    errors: list[BaseException] = []

    def _download() -> None:
        try:
            snapshot_download(
                repo_id=spec.repo_id,
                revision=str(remote.get("revision") or spec.revision or "main"),
                local_dir=str(tmp_target),
                allow_patterns=list(spec.allow_patterns) or None,
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=_download, name=f"download-{model_id}", daemon=True)
    worker.start()
    total = int(remote.get("total_size") or 0)
    try:
        while worker.is_alive():
            if cancel and cancel():
                raise InterruptedError("Cancelled")
            done = _dir_download_bytes(tmp_target)
            pct = min(99.0, (done / total * 100) if total else 0.0)
            _emit(progress, stage="downloading", msg=f"Downloading {spec.label}…", pct=pct, done_bytes=done, total_bytes=total)
            worker.join(0.25)
        if errors:
            raise errors[0]

        # snapshot_download(local_dir=...) adds transfer bookkeeping that is not
        # part of the model itself. Keep project model folders clean.
        shutil.rmtree(tmp_target / ".cache", ignore_errors=True)
        _emit(progress, stage="verifying", msg=f"Verifying downloaded {spec.label}…", pct=0)
        ok, hashes = _verify_candidate(tmp_target, remote, progress, cancel)
        if not ok:
            raise RuntimeError(f"Downloaded {spec.label} failed SHA-256 verification")
        _write_manifest(tmp_target, spec, remote, hashes, "download")
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        tmp_target.replace(target)
        _verified_this_process.add(model_id)
        _emit(progress, stage="ready", msg=f"{spec.label} ready", pct=100, source="download")
        return target
    except Exception:
        shutil.rmtree(tmp_target, ignore_errors=True)
        raise


def catalog_status() -> list[dict]:
    out = []
    for spec in MODEL_SPECS.values():
        out.append({
            "id": spec.id,
            "label": spec.label,
            "repo_id": spec.repo_id,
            "kind": spec.kind,
            "provider": spec.provider,
            "description": spec.description,
            "installed": _quick_installed(spec),
            "path": str(spec.path),
            "dependencies": list(spec.dependencies),
        })
    return out
