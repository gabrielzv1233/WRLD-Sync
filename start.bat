@echo off
setlocal
cd /d "%~dp0"

rem Prefer a normal installed Python so Ctrl+C reaches launch.py directly.
rem uv is still a fallback/bootstrap path when Python is not installed globally.
where py >nul 2>&1
if not errorlevel 1 (
  py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
  if not errorlevel 1 (
    py -3 launch.py %*
    exit /b %errorlevel%
  )
)

where python >nul 2>&1
if not errorlevel 1 (
  python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
  if not errorlevel 1 (
    python launch.py %*
    exit /b %errorlevel%
  )
)

where uv >nul 2>&1
if not errorlevel 1 (
  uv --version >nul 2>&1
  if not errorlevel 1 (
    uv run --no-project --python ">=3.10" launch.py %*
    exit /b %errorlevel%
  )
)

echo.
echo   WRLD Sync needs uv or Python 3.10+.
echo   Recommended: winget install --id astral-sh.uv -e
echo   Or install Python from https://python.org/downloads/
echo   and check "Add python.exe to PATH" during setup.
exit /b 1
