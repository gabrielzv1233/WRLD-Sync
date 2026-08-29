@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 goto try_python
uv --version >nul 2>&1
if errorlevel 1 goto try_python
uv run --no-project --python ">=3.10" launch.py %*
set "launcher_exit=%errorlevel%"
goto done

:try_python
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 goto try_py
python launch.py %*
set "launcher_exit=%errorlevel%"
goto done

:try_py
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 goto missing_python
py -3 launch.py %*
set "launcher_exit=%errorlevel%"
goto done

:missing_python
echo.
echo   WRLD Sync needs uv or Python 3.10+.
echo   Recommended: winget install --id astral-sh.uv -e
echo   Or install Python from https://python.org/downloads/
echo   and check "Add python.exe to PATH" during setup.
set "launcher_exit=1"

:done
echo.
pause
exit /b %launcher_exit%
