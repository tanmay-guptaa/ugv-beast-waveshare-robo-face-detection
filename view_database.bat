@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe view_database.py
) else (
    python view_database.py
)
pause
