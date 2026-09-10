@echo off
echo ============================================
echo   UGV Beast Face Detection Dashboard
echo   Starting Streamlit on port 8501
echo ============================================
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: .venv not found. Run install.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
streamlit run Dashboard.py --server.port 8501

pause
