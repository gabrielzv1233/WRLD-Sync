@echo off
setlocal
cd /d "%~dp0"

rem WRLD Sync pins its ML environment to Python 3.13 for dependency compatibility.
rem Prefer an installed 3.13 so Ctrl+C reaches launch.py directly; uv can bootstrap 3.13 otherwise.
where py >nul 2>&1
if not errorlevel 1 (
  py -3.13 -c "import sys" >nul 2>&1
  if not errorlevel 1 (
    py -3.13 launch.py %*
    exit /b %errorlevel%
  )
)

where uv >nul 2>&1
if not errorlevel 1 (
  uv --version >nul 2>&1
  if not errorlevel 1 (
    uv run --no-project --python "3.13" launch.py %*
    exit /b %errorlevel%
  )
)

where python >nul 2>&1
if not errorlevel 1 (
  python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)" >nul 2>&1
  if not errorlevel 1 (
    python launch.py %*
    exit /b %errorlevel%
  )
)

echo.
echo   WRLD Sync needs uv or Python 3.13.
echo   Recommended: winget install --id astral-sh.uv -e
echo   Or install Python from https://python.org/downloads/
echo   and check "Add python.exe to PATH" during setup.
exit /b 1
