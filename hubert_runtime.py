"""Project-local HuBERT FA installer + English singing alignment runtime.

The runtime runs the HuBERT ONNX network once per audio source, then reuses its
logits to score conservative pronunciation alternatives. Visible lyric text is
never passed through the aligner as pronunciation text and is restored exactly
from the phoneme plan after alignment.
"""
from __future__ import annotations

from hubert_phonemes import build_phoneme_plan, PhonemePlan
from dataclasses import dataclass, replace
from collections.abc import Callable
import urllib.parse
import importlib
import tempfile
import hashlib
import pathlib
import zipfile
import shutil
import httpx
import json
import sys

ROOT = pathlib.Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODEL_ID = "hubert-fa-combined"
INSTALL_DIR = MODELS_DIR / MODEL_ID
PARTIAL_DIR = MODELS_DIR / ".partials" / MODEL_ID
SOURCE_DIR = INSTALL_DIR / "runtime"
BUNDLE_DIR = INSTALL_DIR / "bundle"

HUBERTFA_TAG = "v0.0.7"
SOURCE_URL = f"https://github.com/wolfgitpr/HubertFA/archive/refs/tags/{HUBERTFA_TAG}.zip"
MODEL_ASSET_NAME = "1218_hfa_model_new_dict.zip"
MODEL_URL = f"https://github.com/wolfgitpr/HubertFA/releases/download/{HUBERTFA_TAG}/{MODEL_ASSET_NAME}"
RELEASE_API_URL = f"https://api.github.com/repos/wolfgitpr/HubertFA/releases/tags/{HUBERTFA_TAG}"

ProgressCallback = Callable[[dict], None]
CancelCallback = Callable[[], bool]
PauseCallback = Callable[[], bool]


class HubertDownloadPaused(InterruptedError):
    pass


@dataclass(frozen=True)
class HubertInstall:
    root: pathlib.Path
    runtime_dir: pathlib.Path
    model_path: pathlib.Path
    dictionary_path: pathlib.Path | None


def _emit(cb: ProgressCallback | None, **payload) -> None:
    if cb:
        cb(payload)


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_one(root: pathlib.Path, name: str) -> pathlib.Path | None:
    matches = list(root.rglob(name))
    return matches[0] if matches else None


def get_hubert_install() -> HubertInstall | None:
    if not INSTALL_DIR.is_dir():
        return None
    runtime = SOURCE_DIR
    onnx = _find_one(BUNDLE_DIR, "model.onnx")
    if not runtime.joinpath("onnx_infer.py").is_file() or onnx is None:
        return None
    dictionary = (
        _find_one(BUNDLE_DIR, "ds_cmudict-07b.txt")
        or _find_one(BUNDLE_DIR, "dictionary.txt")
        or _find_one(runtime, "ds_cmudict-07b.txt")
    )
    return HubertInstall(INSTALL_DIR, runtime, onnx, dictionary)


def hubert_installed() -> bool:
    return get_hubert_install() is not None




def _release_asset_metadata(client: httpx.Client) -> tuple[str, str | None]:
    """Resolve the pinned release asset and GitHub-provided SHA-256 when available.

    GitHub release metadata may expose an asset ``digest`` such as
    ``sha256:<hex>``. We use that authoritative digest when present instead of
    baking an unverified checksum into WRLD Sync. Network/API failure falls back
    to the pinned browser-download URL and final install-structure validation.
    """
    try:
        response = client.get(RELEASE_API_URL, headers={"Accept": "application/vnd.github+json"})
        response.raise_for_status()
        release = response.json()
        for asset in release.get("assets") or []:
            if str(asset.get("name") or "") != MODEL_ASSET_NAME:
                continue
            url = str(asset.get("browser_download_url") or MODEL_URL)
            digest = str(asset.get("digest") or "")
            if digest.lower().startswith("sha256:"):
                value = digest.split(":", 1)[1].strip().lower()
                if len(value) == 64 and all(ch in "0123456789abcdef" for ch in value):
                    return url, value
            return url, None
    except Exception:
        pass
    return MODEL_URL, None


def _download_resumable(
    client: httpx.Client,
    url: str,
    path: pathlib.Path,
    *,
    label: str,
    expected_sha256: str | None,
    weight_start: float,
    weight_end: float,
    progress: ProgressCallback | None,
    cancel: CancelCallback | None,
    pause: PauseCallback | None,
) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    existing = part.stat().st_size if part.is_file() else 0

    def interrupted() -> None:
        if cancel and cancel():
            raise InterruptedError("Cancelled")
        if pause and pause():
            raise HubertDownloadPaused("Paused")

    interrupted()
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with client.stream("GET", url, headers=headers) as r:
        r.raise_for_status()
        # If the origin ignores Range, this response already contains the whole
        # file from byte zero. Restart only this local file and consume it.
        if existing and r.status_code != 206:
            part.unlink(missing_ok=True)
            existing = 0
        total = existing + int(r.headers.get("content-length") or 0)
        mode = "ab" if existing else "wb"
        done = existing
        with part.open(mode) as out:
            for chunk in r.iter_bytes(1024 * 1024):
                interrupted()
                out.write(chunk)
                done += len(chunk)
                frac = done / total if total else 0.0
                _emit(progress, stage="downloading", msg=f"Downloading {label}…", pct=weight_start + frac * (weight_end - weight_start), done_bytes=done, total_bytes=total)
    part.replace(path)
    if expected_sha256:
        _emit(progress, stage="verifying", msg=f"Verifying {label}…", pct=weight_end)
        actual = _sha256(path)
        if actual.lower() != expected_sha256.lower():
            path.unlink(missing_ok=True)
            raise RuntimeError(f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    return path


def _extract_single_root(archive: pathlib.Path, destination: pathlib.Path) -> None:
    temp = pathlib.Path(tempfile.mkdtemp(prefix=".hubert-extract-", dir=MODELS_DIR))
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(temp)
        entries = [x for x in temp.iterdir() if x.name != "__MACOSX"]
        source = entries[0] if len(entries) == 1 and entries[0].is_dir() else temp
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source == temp:
            destination.mkdir(parents=True, exist_ok=True)
            for item in entries:
                shutil.move(str(item), destination / item.name)
        else:
            shutil.move(str(source), destination)
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def ensure_hubertfa(
    progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
    pause: PauseCallback | None = None,
) -> pathlib.Path:
    install = get_hubert_install()
    if install:
        _emit(progress, stage="ready", msg="HuBERT FA combined already installed", pct=100)
        return install.root

    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    source_zip = PARTIAL_DIR / f"HubertFA-{HUBERTFA_TAG}.zip"
    model_zip = PARTIAL_DIR / "1218_hfa_model_new_dict.zip"
    try:
        timeout = httpx.Timeout(30.0, read=180.0, write=30.0, pool=30.0)
        with httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": "WRLD-Sync/HubertFA"}) as client:
            resolved_model_url, resolved_model_sha256 = _release_asset_metadata(client)
            _download_resumable(client, SOURCE_URL, source_zip, label="HuBERT FA runtime", expected_sha256=None, weight_start=0, weight_end=10, progress=progress, cancel=cancel, pause=pause)
            _download_resumable(client, resolved_model_url, model_zip, label="HuBERT FA ONNX bundle", expected_sha256=resolved_model_sha256, weight_start=10, weight_end=88, progress=progress, cancel=cancel, pause=pause)
        if cancel and cancel():
            raise InterruptedError("Cancelled")
        if pause and pause():
            raise HubertDownloadPaused("Paused")
        _emit(progress, stage="installing", msg="Installing HuBERT FA…", pct=90)
        staging = pathlib.Path(tempfile.mkdtemp(prefix=".hubert-install-", dir=MODELS_DIR))
        try:
            runtime_dest = staging / "runtime"
            bundle_dest = staging / "bundle"
            _extract_single_root(source_zip, runtime_dest)
            _extract_single_root(model_zip, bundle_dest)
            onnx = _find_one(bundle_dest, "model.onnx")
            if onnx is None:
                raise RuntimeError("HuBERT FA release bundle did not contain model.onnx")
            required = [onnx.parent / "config.json", onnx.parent / "vocab.json", onnx.parent / "VERSION"]
            missing = [str(x.name) for x in required if not x.is_file()]
            if missing:
                raise RuntimeError(f"HuBERT FA ONNX bundle is missing: {', '.join(missing)}")
            manifest = {
                "id": MODEL_ID,
                "tag": HUBERTFA_TAG,
                "source_url": SOURCE_URL,
                "model_url": resolved_model_url,
                "model_sha256": resolved_model_sha256,
            }
            (staging / ".wrld-model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            if INSTALL_DIR.exists():
                shutil.rmtree(INSTALL_DIR, ignore_errors=True)
            staging.replace(INSTALL_DIR)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        shutil.rmtree(PARTIAL_DIR, ignore_errors=True)
        if not hubert_installed():
            raise RuntimeError("HuBERT FA install validation failed")
        _emit(progress, stage="ready", msg="HuBERT FA combined ready", pct=100)
        return INSTALL_DIR
    except HubertDownloadPaused:
        # Keep the last real download percentage in the queue UI. The task
        # runner marks status=paused, so overwriting progress with 0% here would
        # make a correctly resumable partial download look as if it was lost.
        raise
    except InterruptedError:
        shutil.rmtree(PARTIAL_DIR, ignore_errors=True)
        raise


def remove_hubertfa() -> dict:
    removed = False
    for path in (INSTALL_DIR, PARTIAL_DIR):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            removed = True
    return {"model_id": MODEL_ID, "removed": removed}


class HubertFAEngine:
    """Pinned HuBERT FA ONNX engine with audio-scored pronunciation candidates."""

    def __init__(self, install: HubertInstall | None = None):
        self.install = install or get_hubert_install()
        if self.install is None:
            raise RuntimeError("HuBERT FA combined is not installed")
        runtime_str = str(self.install.runtime_dir)
        if runtime_str not in sys.path:
            sys.path.insert(0, runtime_str)
        # Import after adding the pinned runtime source directory.
        onnx_mod = importlib.import_module("onnx_infer")
        decoder_mod = importlib.import_module("tools.decoder")
        self.AlignmentDecoder = decoder_mod.AlignmentDecoder
        self.inference = onnx_mod.InferenceOnnx(self.install.model_path)
        self.inference.load_config()
        self.inference.init_decoder()
        self.inference.load_model()
        self.sample_rate = int(self.inference.mel_cfg["sample_rate"])
        self.hop_size = int(self.inference.mel_cfg["hop_size"])
        self.language_prefix = bool(self.inference.vocab.get("language_prefix"))
        self.language = "en"

    def _sequence(self, plan: PhonemePlan):
        ph_seq = ["SP"]
        word_seq: list[str] = []
        ph_to_word = [-1]
        for word_pos, word in enumerate(plan.words):
            word_seq.append(word.text)
            for phone in word.phones:
                ph_seq.append(f"{self.language}/{phone}" if self.language_prefix else phone)
                ph_to_word.append(word_pos)
            if ph_seq[-1] != "SP":
                ph_seq.append("SP")
                ph_to_word.append(-1)
        return ph_seq, word_seq, ph_to_word

    def _decoder(self):
        return self.AlignmentDecoder(self.inference.vocab, self.sample_rate, self.hop_size)

    def _decode(self, plan: PhonemePlan, frame_logits, edge_logits, wav_length: float):
        ph_seq, word_seq, ph_to_word = self._sequence(plan)
        vocab = self.inference.vocab.get("vocab", {})
        missing = sorted({phone for phone in ph_seq if phone not in vocab})
        if missing:
            raise RuntimeError(f"HuBERT FA model vocabulary does not contain: {', '.join(missing)}")
        decoder = self._decoder()
        words, confidence = decoder.decode(
            ph_frame_logits=frame_logits,
            ph_edge_logits=edge_logits,
            wav_length=wav_length,
            ph_seq=ph_seq,
            word_seq=word_seq,
            ph_idx_to_word_idx=ph_to_word,
        )
        return words, float(confidence), decoder

    def _score_word_candidates(self, plan: PhonemePlan, base_words, frame_logits, edge_logits, progress=None) -> None:
        """Score alternates locally around each base word using the same audio logits.

        The canonical full-song pass gives a time window. Each word's CMU/G2P/
        controlled singing candidates are then decoded against that local audio
        window. An alternate must beat the current score by a small margin, which
        prevents tiny confidence noise from selecting gratuitous re-articulation.
        """
        import numpy as np

        by_index = {i: word for i, word in enumerate(base_words)}
        frame_seconds = self.hop_size / self.sample_rate
        for index, lyric_word in enumerate(plan.words):
            if progress:
                progress('candidates', index, len(plan.words))
            if len(lyric_word.candidates) <= 1:
                continue
            base = by_index.get(lyric_word.index)
            if base is None:
                continue
            left = max(0.0, float(base.start) - 0.18)
            right = max(left + 0.08, float(base.end) + 0.18)
            start_frame = max(0, int(left / frame_seconds))
            end_frame = min(frame_logits.shape[-1], max(start_frame + 2, int(right / frame_seconds)))
            if end_frame <= start_frame + 1:
                continue
            local_frame = frame_logits[:, :, start_frame:end_frame]
            local_edge = edge_logits[:, start_frame:end_frame]
            local_len = (end_frame - start_frame) * frame_seconds
            best_idx, best_score = lyric_word.chosen, float("-inf")
            original_choice = lyric_word.chosen
            for idx, candidate in enumerate(lyric_word.candidates):
                if progress:
                    progress('candidates', index, len(plan.words))
                # Score with a detached one-word copy so trying an alternate never
                # mutates the global word/range map until a winner is selected.
                tiny_word = replace(lyric_word, chosen=idx, phoneme_start=0, phoneme_end=len(candidate.phones))
                tiny = PhonemePlan(plan.raw_text, [tiny_word], list(candidate.phones))
                try:
                    _, score, _ = self._decode(tiny, local_frame, local_edge, local_len)
                except Exception:
                    continue
                # Small complexity penalty makes repeated-vowel variants earn their keep.
                score -= max(0, len(candidate.phones) - len(lyric_word.candidates[0].phones)) * 0.003
                if score > best_score + 0.002:
                    best_idx, best_score = idx, score
            lyric_word.chosen = best_idx if best_score != float("-inf") else original_choice

        if progress:
            progress('candidates', len(plan.words), len(plan.words))
        # Rebuild global ranges after local candidate choices changed phone counts.
        flattened: list[str] = []
        for word in plan.words:
            word.phoneme_start = len(flattened)
            flattened.extend(word.phones)
            word.phoneme_end = len(flattened)
        plan.phones[:] = flattened

    def align(self, audio_path: str | pathlib.Path, lyrics: str, progress=None) -> list[dict]:
        import librosa
        import numpy as np

        if progress:
            progress('preparing')
        plan = build_phoneme_plan(
            lyrics,
            dictionary_path=self.install.dictionary_path,
            include_singing_variants=True,
        )
        if not plan.words:
            return []
        wav, _ = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        wav = np.asarray(wav, dtype=np.float32)
        wav_length = len(wav) / self.sample_rate
        if progress:
            progress('inference')
        results = self.inference.run_onnx(self.inference.model, {"waveform": [wav]})
        frame_logits = results["ph_frame_logits"]
        edge_logits = results["ph_edge_logits"]

        if progress:
            progress('decoding')
        base_words, _, _ = self._decode(plan, frame_logits, edge_logits, wav_length)
        self._score_word_candidates(plan, base_words, frame_logits, edge_logits, progress=progress)
        if progress:
            progress('finalizing')
        final_words, confidence, _ = self._decode(plan, frame_logits, edge_logits, wav_length)

        out: list[dict] = []
        for idx, word in enumerate(final_words):
            if idx >= len(plan.words):
                break
            original = plan.words[idx]
            out.append({
                "text": original.text,
                "start_time": float(word.start),
                "end_time": float(word.end),
                "alignment_confidence": confidence,
                "pronunciation_source": original.candidates[original.chosen].source,
                "phones": list(original.phones),
            })
        return out
