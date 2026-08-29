#!/usr/bin/env python3
"""
WRLD Sync launcher.

Sets up the virtual environment and dependencies if needed, then starts the
server and opens it in your browser. Safe to run repeatedly — subsequent
runs skip setup steps that are already done.

Usage:
    python launch.py
    python launch.py --port 8080
    python launch.py --no-browser
    python launch.py --auto-update       # pull updates without asking
    python launch.py --no-update-check   # skip the update check entirely
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def _enable_windows_vt() -> bool:
    """Older/plain cmd.exe windows don't interpret \\033[ ANSI codes unless
    ENABLE_VIRTUAL_TERMINAL_PROCESSING is explicitly turned on for the console
    handle -- without this, users see literal escape-code garbage instead of
    colored text. Returns whether it succeeded."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        STD_OUTPUT_HANDLE = -11
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        if not kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING):
            return False
        return True
    except Exception:
        return False


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not (sys.stdout.isatty() or os.environ.get("FORCE_COLOR") == "1"):
        return False
    if platform.system() == "Windows" and not _enable_windows_vt():
        return False
    return True


USE_COLOR = _supports_color()


def paint(text: str, *codes: str) -> str:
    if not USE_COLOR:
        return text
    return "".join(codes) + text + C.RESET


def step(text: str) -> None:
    print(f"\n{paint('->', C.BOLD, C.CYAN)} {paint(text, C.BOLD)}")


def ok(text: str) -> None:
    print(f"  {paint('OK', C.GREEN, C.BOLD)} {text}")


def warn(text: str) -> None:
    print(f"  {paint('!', C.YELLOW, C.BOLD)} {paint(text, C.YELLOW)}")


def err(text: str) -> None:
    print(f"  {paint('X', C.RED, C.BOLD)} {paint(text, C.RED)}")


def fail(text: str) -> None:
    err(text)
    sys.exit(1)


def confirm(prompt: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    try:
        reply = input(f"  {paint('?', C.MAGENTA, C.BOLD)} {prompt} {suffix} ").strip().lower()
    except (EOFError, OSError):
        return default_yes
    if not reply:
        return default_yes
    return reply in ("y", "yes")


def human_time(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}m{seconds:02d}s"


def port_number(value: str) -> int:
    """An argparse type for TCP ports that can actually be bound."""
    try:
        port = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("port must be a number") from e
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_python_version() -> None:
    if sys.version_info < (3, 10):
        fail(
            f"Python 3.10+ is required (found {platform.python_version()}). "
            "Install a newer Python from https://python.org/downloads/ and try again."
        )
    ok(f"Python {platform.python_version()}")


def _git(args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)


GIT_REMOTE_URL = "https://github.com/leanwrldd/juicewrldapi-whisper.git"


def has_git() -> bool:
    return shutil.which("git") is not None


def ensure_git() -> bool:
    """Returns True if git is available (already installed, or freshly installed via winget)."""
    if has_git():
        return True
    if platform.system() != "Windows":
        warn("git not found. Install it with your OS package manager to enable auto-updates.")
        return False
    if not shutil.which("winget"):
        warn("git not found, and winget isn't available to install it automatically. "
             "Install Git for Windows manually: https://git-scm.com/download/win")
        return False

    warn("git not found — installing via winget (one-time)...")
    t0 = time.time()
    result = subprocess.run(
        ["winget", "install", "--id", "Git.Git", "-e",
         "--accept-package-agreements", "--accept-source-agreements"],
    )
    if result.returncode != 0:
        warn("Automatic git install failed. Install it manually: https://git-scm.com/download/win")
        return False

    _refresh_path_from_registry()
    if has_git():
        ok(f"git installed in {human_time(time.time() - t0)}.")
        return True
    warn(f"git installed in {human_time(time.time() - t0)}, but not detected on PATH yet in this "
         "window — restart this launcher once (a fresh terminal will have it).")
    return False


def _looks_like_this_project(path: Path) -> bool:
    return (path / "app.py").exists() and (path / "requirements.txt").exists() and (path / "launch.py").exists()


def offer_git_conversion(auto_yes: bool) -> None:
    """If this folder isn't a git checkout but is clearly a copy of this
    project (e.g. downloaded as a GitHub ZIP), offer to replace it with a
    proper git clone in place so future runs can actually auto-update."""
    if not _looks_like_this_project(ROOT):
        return
    if not ensure_git():
        return

    warn("This folder isn't a git checkout (likely downloaded as a ZIP), so it can never "
         "auto-update as-is. It can be converted into a real git checkout in place.")
    # auto_yes (only true with --auto-update) skips the prompt entirely. Otherwise
    # always ask -- but default the answer to Yes, since this is a safe, reversible
    # operation (the old folder is backed up, never deleted) and most users just
    # press Enter rather than typing an explicit y/n.
    if not auto_yes and not confirm("Convert this folder into a git checkout now?", default_yes=True):
        ok("Skipping — update checks will keep saying 'not a git checkout'.")
        return

    tmp_dir = ROOT.parent / f"{ROOT.name}.gitconv"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    step("Cloning a fresh git checkout")
    t0 = time.time()
    result = subprocess.run(["git", "clone", "--quiet", GIT_REMOTE_URL, str(tmp_dir)])
    if result.returncode != 0:
        warn("Clone failed — leaving the current folder untouched.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return
    ok(f"Cloned in {human_time(time.time() - t0)}.")

    # Carry over local preferences that only exist on disk, not in git.
    for name in (".env", ".model_pref", ".model_pref_align", ".model_pref_verify", ".device_pref", ".engine_pref"):
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, tmp_dir / name)

    backup_dir = ROOT.parent / f"{ROOT.name}.bak"
    if backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)

    try:
        ROOT.rename(backup_dir)
    except OSError as e:
        warn(f"Couldn't move the old folder aside ({e}).")
        ok(f"Your new git checkout is ready at: {tmp_dir} — run start.bat from there instead.")
        sys.exit(0)

    try:
        tmp_dir.rename(ROOT)
    except OSError as e:
        backup_dir.rename(ROOT)  # put the original back, nothing lost
        warn(f"Couldn't move the new clone into place ({e}) — restored the original folder. "
             f"The git clone is still available at: {tmp_dir}")
        return

    ok(f"This folder is now a proper git checkout. The old copy is backed up at: {backup_dir}")
    step("Restarting with the git checkout")
    os.execv(sys.executable, [sys.executable, str(ROOT / "launch.py"), *sys.argv[1:]])


def check_for_updates(auto_update: bool, skip: bool) -> None:
    step("Checking for updates")
    if skip:
        ok("Skipped (--no-update-check).")
        return

    if not (ROOT / ".git").exists():
        ok("Not a git checkout, skipping update check.")
        offer_git_conversion(auto_update)
        return

    try:
        if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
            warn("git not found on PATH, skipping update check.")
            return
    except FileNotFoundError:
        warn("git not found on PATH, skipping update check.")
        return

    status = _git(["status", "--porcelain"])
    if status.returncode != 0:
        warn("Couldn't read git status, skipping update check.")
        return
    if status.stdout.strip():
        warn("You have local changes — skipping auto-update to avoid conflicts.")
        return

    try:
        fetch = _git(["fetch", "--quiet", "origin"], timeout=15)
    except subprocess.TimeoutExpired:
        warn("Timed out reaching GitHub — continuing with the current version.")
        return
    if fetch.returncode != 0:
        warn("Couldn't reach GitHub to check for updates (offline?). Continuing with the current version.")
        return

    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    local = _git(["rev-parse", "HEAD"]).stdout.strip()
    remote_result = _git(["rev-parse", f"origin/{branch}"])
    if remote_result.returncode != 0 or not remote_result.stdout.strip():
        ok(f"No 'origin/{branch}' to compare against, skipping.")
        return
    remote = remote_result.stdout.strip()

    if local == remote:
        ok("You're up to date.")
        return

    behind = _git(["rev-list", "--count", f"{local}..{remote}"]).stdout.strip()
    warn(f"Update available: {behind} new commit(s) on '{branch}'.")

    do_pull = auto_update or confirm("Download and apply the update now?")
    if not do_pull:
        ok("Skipping update for this run.")
        return

    t0 = time.time()
    pull = subprocess.run(["git", "pull", "--ff-only", "origin", branch], cwd=ROOT)
    if pull.returncode != 0:
        warn("git pull failed — continuing with the current version. You may need to update manually.")
        return
    ok(f"Updated in {human_time(time.time() - t0)}.")

    step("Restarting with the updated code")
    os.execv(sys.executable, [sys.executable, str(ROOT / "launch.py"), *sys.argv[1:]])


def _try_enable_long_paths() -> None:
    """PyTorch's package ships absurdly deep nested files (license notices under
    third_party/kineto/.../DCGM/testing/python3/libs_3rdparty/...), which combined
    with a long project folder path can blow past Windows' legacy 260-character
    MAX_PATH limit and make pip fail outright with WinError 206. NTFS long-path
    support (available since Windows 10 1607) fixes this. Best-effort only --
    requires admin to write HKLM, so this silently does nothing if we're not
    elevated; _install_failure_hint() below explains the manual fallback."""
    if platform.system() != "Windows":
        return
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
            0, winreg.KEY_SET_VALUE,
        ) as key:
            winreg.SetValueEx(key, "LongPathsEnabled", 0, winreg.REG_DWORD, 1)
    except OSError:
        pass


def _install_failure_hint() -> None:
    if platform.system() != "Windows":
        return
    warn("If the error above mentions 'WinError 206' or 'filename or extension is too "
         "long', that's Windows' classic 260-character path limit -- PyTorch ships some "
         "very deeply nested files. Two fixes:")
    warn("  1. Move this folder somewhere with a shorter path (e.g. C:\\WRLDSync) and run "
         "start.bat again.")
    warn("  2. Or enable long path support (needs to be run as Administrator once):")
    warn('     reg add "HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" '
         "/v LongPathsEnabled /t REG_DWORD /d 1 /f")


def ensure_venv() -> None:
    step("Checking virtual environment")
    if venv_python().exists():
        ok(f".venv already set up ({venv_python()})")
        return
    warn(".venv not found, creating it...")
    t0 = time.time()
    result = subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)])
    if result.returncode != 0:
        fail("Failed to create the virtual environment.")
    ok(f"Created .venv in {human_time(time.time() - t0)}")


def detect_uv() -> tuple[str | None, str | None]:
    """Return uv's absolute path and version label when it can be launched.

    Detection happens once per launcher run. Keeping the resolved path also
    means a later Windows PATH refresh cannot accidentally hide uv.
    """
    executable = shutil.which("uv")
    if not executable:
        return None, None
    try:
        result = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None, None
    if result.returncode != 0:
        return None, None
    version = (result.stdout or result.stderr or "").strip() or "uv"
    return executable, version


def _install_packages(args: list[str], uv_executable: str | None) -> subprocess.CompletedProcess:
    """Install packages into this project's venv, preferring uv when available."""
    if uv_executable:
        command = [
            uv_executable, "pip", "install", "--python", str(venv_python()), *args,
        ]
        try:
            return subprocess.run(command, cwd=ROOT)
        except OSError as e:
            # A successful `uv --version` followed by a launch error generally
            # means the executable was moved or blocked between the two calls.
            # Resolution/network failures return normally and must not retry a
            # multi-gigabyte PyTorch download with pip.
            warn(f"uv could not be launched ({e}) — falling back to pip.")

    return subprocess.run(
        [str(venv_python()), "-m", "pip", "install", *args], cwd=ROOT,
    )


def requirements_satisfied() -> bool:
    """Cheap check: were the current requirements previously installed?
    Avoids re-running the slow install/resolve on every single launch."""
    marker = VENV_DIR / ".requirements_installed"
    if not marker.exists():
        return False
    try:
        return marker.read_text().strip() == _requirements_fingerprint()
    except OSError:
        return False


def _requirements_fingerprint() -> str:
    import hashlib
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def has_nvidia_gpu() -> bool:
    try:
        return subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5
        ).returncode == 0
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


def gpu_compute_capability() -> float | None:
    """Compute capability (e.g. 6.1 for a GT 1030) of GPU 0, or None if it
    can't be determined. The cu128 PyTorch build only ships kernels for
    Turing and newer (sm_75+), so older cards need to be steered to CPU
    instead of downloading a multi-hundred-MB wheel that'll just silently
    fall back to CPU anyway at inference time."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        first_line = result.stdout.strip().splitlines()[0].strip()
        return float(first_line)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


# Lowest compute capability the cu128 PyTorch wheels ship kernels for (Turing+).
MIN_SUPPORTED_COMPUTE_CAP = 7.5
# Lowest compute capability the older cu118 wheels ship kernels for (Maxwell+) --
# below this, no current PyTorch build has kernels for the card at all.
MIN_LEGACY_COMPUTE_CAP = 5.0


def torch_install_info() -> tuple[str | None, str | None]:
    """Return (torch version, compiled CUDA version). CUDA is None for a CPU wheel."""
    result = subprocess.run(
        [str(venv_python()), "-c",
         "import torch; print(torch.__version__); print(torch.version.cuda or '')"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None, None
    lines = result.stdout.splitlines()
    version = lines[0].strip() if lines else None
    cuda = lines[1].strip() if len(lines) > 1 else ""
    return version or None, cuda or None


def torch_cuda_build() -> str | None:
    return torch_install_info()[1]


def torch_runtime_cuda_available() -> bool:
    result = subprocess.run(
        [str(venv_python()), "-c", "import torch; print(int(torch.cuda.is_available()))"],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "1"


def ensure_gpu_torch(uv_executable: str | None) -> None:
    """Ensure an NVIDIA machine does not stay stuck on a CPU-only Torch wheel.

    uv/pip considers an already-installed CPU build to satisfy the package name
    ``torch`` even when a CUDA-specific index is supplied. When that situation is
    detected we explicitly reinstall from the CUDA index, then verify the wheel
    reports both a CUDA build and a usable GPU before claiming success.
    """
    step("Checking for GPU acceleration")
    if not venv_python().exists():
        ok("Skipping (.venv not created yet).")
        return
    if not has_nvidia_gpu():
        ok("No NVIDIA GPU detected — CUDA packages are not needed.")
        return

    cap = gpu_compute_capability()
    too_old_for_cu128 = cap is not None and cap < MIN_SUPPORTED_COMPUTE_CAP
    too_old_for_anything = cap is not None and cap < MIN_LEGACY_COMPUTE_CAP
    version, build = torch_install_info()

    if build:
        if build.startswith("12") and too_old_for_cu128 and not too_old_for_anything:
            warn(
                f"PyTorch {version} has CUDA {build}, but this GPU ({cap}) needs the "
                "legacy CUDA 11.8 wheel. Reinstalling once..."
            )
            index_url = "https://download.pytorch.org/whl/cu118"
        else:
            if torch_runtime_cuda_available():
                ok(f"GPU PyTorch ready: {version} + CUDA {build}.")
            else:
                warn(
                    f"PyTorch {version} is a CUDA {build} build, but torch.cuda.is_available() "
                    "is false. Keeping it installed; Faster-Whisper can still use CUDA "
                    "independently through CTranslate2 if that runtime is available."
                )
            return
    else:
        if too_old_for_anything:
            warn(
                f"NVIDIA GPU detected, but compute capability {cap} is too old for the "
                "supported PyTorch CUDA wheels. Keeping CPU PyTorch."
            )
            return
        index_url = (
            "https://download.pytorch.org/whl/cu118"
            if too_old_for_cu128
            else "https://download.pytorch.org/whl/cu128"
        )
        if version:
            warn(
                f"NVIDIA GPU detected, but PyTorch {version} is CPU-only. Replacing it "
                f"with the {'CUDA 11.8' if too_old_for_cu128 else 'CUDA 12.8'} wheel..."
            )
        else:
            warn(
                f"NVIDIA GPU detected — installing the "
                f"{'CUDA 11.8' if too_old_for_cu128 else 'CUDA 12.8'} PyTorch build..."
            )

    t0 = time.time()
    install_args = []
    # This is the important bit for an existing +cpu wheel: merely changing
    # --index-url does NOT make uv replace an already-satisfied torch package.
    if version is not None:
        install_args.append("--force-reinstall")
    install_args += ["torch", "torchaudio", "--index-url", index_url]

    result = _install_packages(install_args, uv_executable)
    if result.returncode != 0:
        _install_failure_hint()
        warn("GPU PyTorch install failed — leaving the existing Torch installation in place.")
        return

    new_version, new_build = torch_install_info()
    if not new_build:
        warn(
            "The install command completed, but Torch is still CPU-only. This Python/PyTorch "
            "combination may not have a matching CUDA wheel yet."
        )
        return

    if torch_runtime_cuda_available():
        ok(
            f"GPU PyTorch ready: {new_version} + CUDA {new_build} "
            f"({human_time(time.time() - t0)})."
        )
    else:
        warn(
            f"Installed PyTorch {new_version} + CUDA {new_build}, but CUDA runtime detection "
            "still failed. Check the NVIDIA driver if the Torch backend is needed."
        )


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _refresh_path_from_registry() -> None:
    """Winget (and installers in general) update PATH in the registry, but an
    already-running process keeps its own cached copy in os.environ -- pull
    the current machine + user PATH values and merge them in, the same way
    Windows would on a fresh process/shell without needing a full restart."""
    if platform.system() != "Windows":
        return
    import winreg
    registry_paths = []
    locations = (
        (winreg.HKEY_LOCAL_MACHINE,
         r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
        (winreg.HKEY_CURRENT_USER, "Environment"),
    )
    for hive, location in locations:
        try:
            with winreg.OpenKey(hive, location) as key:
                registry_paths.append(winreg.QueryValueEx(key, "Path")[0])
        except OSError:
            pass

    # Preserve process-only entries (portable tools, activated shells, etc.)
    # while appending paths an installer just added to the registry.
    values = [os.environ.get("PATH", ""), *registry_paths]
    os.environ["PATH"] = _merge_path_values(values)


def _merge_path_values(values: list[str]) -> str:
    entries = []
    seen = set()
    for value in values:
        for entry in value.split(os.pathsep):
            if not entry:
                continue
            key = os.path.normcase(os.path.normpath(entry))
            if key in seen:
                continue
            seen.add(key)
            entries.append(entry)
    return os.pathsep.join(entries)


def ensure_ffmpeg() -> None:
    """Whisper shells out to ffmpeg to decode audio; without it, syncing fails
    deep inside a background task with an opaque '[WinError 2] The system
    cannot find the file specified' instead of a clear message here."""
    step("Checking for ffmpeg")
    if has_ffmpeg():
        ok("ffmpeg found.")
        return

    if platform.system() != "Windows":
        warn("ffmpeg not found on PATH — Whisper needs it to decode audio. "
             "Install it with your OS package manager (e.g. 'brew install ffmpeg' on Mac).")
        return

    if not shutil.which("winget"):
        warn("ffmpeg not found, and winget isn't available to install it automatically. "
             "Install it manually: https://ffmpeg.org/download.html")
        return

    warn("ffmpeg not found — installing via winget (one-time)...")
    t0 = time.time()
    result = subprocess.run(
        ["winget", "install", "--id", "Gyan.FFmpeg", "-e",
         "--accept-package-agreements", "--accept-source-agreements"],
    )
    if result.returncode != 0:
        warn("Automatic ffmpeg install failed. Install it manually with "
             "'winget install ffmpeg' (or https://ffmpeg.org/download.html), "
             "then restart this launcher.")
        return

    _refresh_path_from_registry()

    if has_ffmpeg():
        ok(f"ffmpeg installed in {human_time(time.time() - t0)}.")
    else:
        warn(f"ffmpeg installed in {human_time(time.time() - t0)}, but not detected on PATH "
             "yet in this window — restart this launcher once (a fresh terminal will have it).")


def install_requirements(uv_executable: str | None) -> None:
    step("Checking Python dependencies")
    if requirements_satisfied():
        ok("Dependencies already installed and up to date.")
        return

    warn("Installing dependencies (first run, or requirements.txt changed)...")
    warn("This includes PyTorch/Whisper and can take several minutes.")
    t0 = time.time()
    result = _install_packages(
        ["-r", str(REQUIREMENTS)],
        uv_executable,
    )
    if result.returncode != 0:
        _install_failure_hint()
        fail("Dependency installation failed — see the installer output above.")

    marker = VENV_DIR / ".requirements_installed"
    marker.write_text(_requirements_fingerprint())
    ok(f"Dependencies installed in {human_time(time.time() - t0)}.")


def _pids_listening_on(port: int) -> list[int]:
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-NetTCPConnection -LocalPort {port} -State Listen "
             "-ErrorAction SilentlyContinue).OwningProcess"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    pids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return sorted(set(pids))


def _process_command_line(pid: int) -> str | None:
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-CimInstance Win32_Process -Filter \"ProcessId = {pid}\" "
             "-ErrorAction SilentlyContinue).CommandLine"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _is_wrld_sync_server(command_line: str | None) -> bool:
    if not command_line:
        return False
    command = command_line.casefold().replace("/", "\\")
    expected_python = str(venv_python()).casefold().replace("/", "\\")
    padded = f" {command} "
    return (expected_python in command and
            " -m uvicorn " in padded and
            " app:app " in padded)


def stop_existing_server(port: int) -> None:
    step("Checking for an already-running server")
    if platform.system() != "Windows":
        ok("Skipping (not Windows) — if a server is already running on the "
           "target port, starting a new one below will fail loudly.")
        return

    pids = [p for p in _pids_listening_on(port) if p != os.getpid()]
    if not pids:
        ok("No existing server was running on this port.")
        return

    unknown = [p for p in pids if not _is_wrld_sync_server(_process_command_line(p))]
    if unknown:
        joined = ", ".join(str(p) for p in unknown)
        fail(f"Port {port} is already in use by another process (PID {joined}). "
             f"Close it or choose another port with --port.")

    for pid in pids:
        try:
            result = subprocess.run(
                ["taskkill", "/f", "/pid", str(pid)], capture_output=True, timeout=5,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            fail(f"Couldn't stop the previous WRLD Sync server on port {port} (PID {pid}).")
        if result.returncode != 0:
            fail(f"Couldn't stop the previous WRLD Sync server on port {port} (PID {pid}).")
    ok(f"Stopped {len(pids)} process(es) previously listening on port {port}.")
    time.sleep(0.5)  # give the OS a moment to release the socket


def wait_for_server(port: int, proc: subprocess.Popen, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1.5):
                return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.3)
    return False


def print_banner(port: int, uv_version: str | None) -> None:
    installer = f"{uv_version} (accelerated)" if uv_version else "pip (uv not available)"
    print()
    print(paint("=" * 58, C.MAGENTA))
    print(f"  {paint('WRLD Sync', C.BOLD, C.MAGENTA)}")
    print(f"  {paint('Local lyrics syncing workspace', C.DIM)}")
    print(paint("-" * 58, C.DIM))
    print(f"  Address    http://127.0.0.1:{port}")
    print(f"  Installer  {installer}")
    print(paint("=" * 58, C.MAGENTA))


def _stop_server_process(proc: subprocess.Popen) -> None:
    """Stop Uvicorn without letting native Whisper/executor threads hang Ctrl+C."""
    if proc.poll() is not None:
        return

    try:
        if platform.system() == "Windows":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(proc.pid, signal.SIGINT)
    except (OSError, ValueError):
        pass

    try:
        proc.wait(timeout=3)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        proc.terminate()
        proc.wait(timeout=2)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass

    try:
        proc.kill()
    except OSError:
        return
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


def run_server(port: int, open_browser: bool) -> int:
    step("Starting the server")
    url = f"http://127.0.0.1:{port}"
    popen_kwargs = {"cwd": ROOT}
    if platform.system() == "Windows":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        [str(venv_python()), "-m", "uvicorn", "app:app", "--host", "127.0.0.1",
         "--port", str(port), "--no-access-log", "--log-level", "warning"],
        **popen_kwargs,
    )

    try:
        ready = wait_for_server(port, proc)
        if ready:
            ok(f"Server is ready at {url}")
        elif proc.returncode is not None:
            err(f"Server exited before it was ready (exit code {proc.returncode}).")
            return proc.returncode
        else:
            warn("Server didn't respond within 60s — it may still be starting. "
                 "The browser will stay closed; check the logs above.")

        if ready and open_browser:
            if webbrowser.open(url):
                ok("Opened in your browser.")
            else:
                warn(f"Couldn't open a browser automatically — visit {url} manually.")

        status = "WRLD Sync is running." if ready else "WRLD Sync is still starting."
        status_color = C.GREEN if ready else C.YELLOW
        print(f"\n{paint(status, status_color, C.BOLD)}")
        print(f"{paint(url, C.CYAN)}  |  Press Ctrl+C here to stop it.\n")
        return proc.wait()
    except KeyboardInterrupt:
        print()
        step("Shutting down")
        _stop_server_process(proc)
        ok("Stopped.")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set up and launch WRLD Sync.")
    parser.add_argument("--port", type=port_number, default=8000, help="Port to serve on (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Don't automatically open a browser tab")
    parser.add_argument("--auto-update", action="store_true", help="Pull updates automatically without asking")
    parser.add_argument("--no-update-check", action="store_true", help="Don't check for updates at all")
    args = parser.parse_args(argv)

    uv_executable, uv_version = detect_uv()
    print_banner(args.port, uv_version)

    check_for_updates(auto_update=args.auto_update, skip=args.no_update_check)

    step("Checking Python")
    ensure_python_version()

    _try_enable_long_paths()
    ensure_venv()
    ensure_gpu_torch(uv_executable)
    ensure_ffmpeg()
    install_requirements(uv_executable)
    stop_existing_server(args.port)

    return run_server(args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    sys.exit(main())
