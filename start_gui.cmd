@echo off
setlocal
cd /d "%~dp0"
if not exist "config.json" (
  copy /y "config.example.json" "config.json" >nul
  echo config.json was created from config.example.json.
  echo Edit the model and output paths before starting the GUI.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Python environment not found. Run setup_windows.cmd first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" trainer_app.py
if errorlevel 1 (
  echo.
  echo The GUI stopped with an error. Review the message above.
  pause
)
