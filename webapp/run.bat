@echo off
echo ============================================================
echo Starting Squat Form Analyzer Web Application
echo ============================================================
echo.

REM Check if model exists
if not exist "..\models\pose_iter_440000.caffemodel" (
    echo WARNING: OpenPose model not found!
    echo Please run: python ..\download_openpose_model.py
    echo.
    pause
    exit /b 1
)

echo Model found! Starting server...
echo.
echo Access the application at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
echo ============================================================

python app.py
