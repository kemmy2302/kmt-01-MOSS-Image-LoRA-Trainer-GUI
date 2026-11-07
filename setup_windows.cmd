@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

where git >nul 2>nul || (echo [ERROR] Git for Windows was not found.& pause & exit /b 1)
if not exist ".venv\Scripts\python.exe" (
  where py >nul 2>nul && py -3.10 -m venv .venv >nul 2>nul
)
if not exist ".venv\Scripts\python.exe" (
  where python >nul 2>nul || goto :python_missing
  python -c "import sys,struct; raise SystemExit(0 if sys.version_info[:2] == (3,10) and struct.calcsize('P') == 8 else 1)" >nul 2>nul || goto :python_missing
  python -m venv .venv || goto :failed
)
".venv\Scripts\python.exe" -c "import sys,struct; raise SystemExit(0 if sys.version_info[:2] == (3,10) and struct.calcsize('P') == 8 else 1)" >nul 2>nul || goto :python_incompatible
echo Using Python:
".venv\Scripts\python.exe" --version
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip || goto :failed
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 || goto :failed
if not exist "sd-scripts\.git" git clone https://github.com/kohya-ss/sd-scripts.git sd-scripts || goto :failed
git -C sd-scripts fetch --quiet || goto :failed
git -C sd-scripts checkout 04ca541746b7667b27eab230db067579f682895e || goto :failed
pushd sd-scripts || goto :failed
python -m pip install -r requirements.txt
set "SD_REQUIREMENTS_RESULT=%ERRORLEVEL%"
popd
if not "%SD_REQUIREMENTS_RESULT%"=="0" goto :failed
python -m pip install -r requirements.txt || goto :failed
if not exist "config.json" copy /y "config.example.json" "config.json" >nul
echo.
echo Setup complete. Edit config.json, then run start_gui.cmd.
pause
exit /b 0

:python_missing
echo.
echo [ERROR] 64-bit Python 3.10 was not found.
echo Install Python 3.10, or make a 64-bit Python 3.10 available as py -3.10 or python in PATH.
pause
exit /b 1

:python_incompatible
echo.
echo [ERROR] The existing .venv is not using 64-bit Python 3.10.
echo Delete the .venv folder and run setup_windows.cmd again.
pause
exit /b 1

:failed
echo.
echo [ERROR] Setup failed. Review the error above, fix it, and run this file again.
pause
exit /b 1
