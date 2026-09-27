@echo off
title Racing Telemetry Overlay - keep this window open (closing it stops any render)
cd /d "%~dp0"
python -c "import sys" >nul 2>nul
if errorlevel 1 (
  echo Python was not found on this PC.
  echo It can be installed with winget, or from python.org ^(tick "Add python.exe to PATH"^).
  choice /M "Install Python 3.12 with winget now"
  if errorlevel 2 goto end
  winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
  echo.
  echo Done. Close this window and run start_ui.bat again.
  goto end
)
python telemetry_ui.py
:end
pause
