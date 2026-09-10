@echo off
echo ============================================
echo   UGV Beast Dashboard — Install Dependencies
echo ============================================
echo.

cd /d "%~dp0"

echo [1/3] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [2/3] Upgrading pip...
python -m pip install --upgrade pip

echo [3/3] Installing packages (this may take a few minutes)...
echo NOTE: face-recognition/dlib may take long or need Visual C++ on Windows.
echo       The dashboard works without it - using OpenCV Haar cascade fallback.
echo.

pip install streamlit opencv-python numpy Pillow requests psutil plotly pandas

echo.
echo ============================================
echo  Optional: install face_recognition for full
echo  128-D embedding recognition:
echo    pip install face-recognition dlib
echo  (Requires CMake + Visual C++ Build Tools)
echo ============================================
echo.

echo [4/4] Running one-time demo setup...
python setup_demo.py

echo.
echo  DONE! Run start_dashboard.bat to launch.
pause
