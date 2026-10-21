@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul || (echo [ERROR] Python Launcher was not found. Install 64-bit Python 3.10 with the launcher enabled.& pause & exit /b 1)
py -3.10 --version >nul 2>nul || (echo [ERROR] 64-bit Python 3.10 was not found.& pause & exit /b 1)
where git >nul 2>nul || (echo [ERROR] Git for Windows was not found.& pause & exit /b 1)
if not exist ".venv\Scripts\python.exe" py -3.10 -m venv .venv || goto :failed
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip || goto :failed
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 || goto :failed
if not exist "sd-scripts\.git" git clone https://github.com/kohya-ss/sd-scripts.git sd-scripts || goto :failed
git -C sd-scripts fetch --quiet || goto :failed
git -C sd-scripts checkout 04ca541746b7667b27eab230db067579f682895e || goto :failed
python -m pip install -r "sd-scripts\requirements.txt" || goto :failed
python -m pip install -r requirements.txt || goto :failed
if not exist "config.json" copy /y "config.example.json" "config.json" >nul
echo.
echo Setup complete. Edit config.json, then run start_gui.cmd.
pause
exit /b 0

:failed
echo.
echo [ERROR] Setup failed. Review the error above, fix it, and run this file again.
pause
exit /b 1
