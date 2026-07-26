@echo off
setlocal

cd /d "%~dp0"

where docker >nul 2>&1
if errorlevel 1 (
    echo Error: Docker was not found in PATH.
    exit /b 1
)

py -3.12 -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo Error: Python 3.12 was not found by the Windows Python launcher.
    echo Install Python 3.12, then try again.
    exit /b 1
)

if not exist "venv312\Scripts\python.exe" (
    echo Creating Python 3.12 virtual environment...
    py -3.12 -m venv venv312
    if errorlevel 1 exit /b 1

    echo Installing backend dependencies...
    "venv312\Scripts\python.exe" -m pip install -r apps\api\requirements.txt
    if errorlevel 1 exit /b 1
)

if not exist "apps\web\node_modules" (
    echo Error: Frontend dependencies are not installed.
    echo Run npm install from apps\web, then try again.
    exit /b 1
)

echo Starting PostgreSQL...
docker compose up -d postgres
if errorlevel 1 (
    echo Error: PostgreSQL could not be started. Make sure Docker Desktop is running.
    exit /b 1
)

echo Starting backend and frontend...
start "SquatSpot Backend" /D "%~dp0apps\api" cmd /k ""%~dp0venv312\Scripts\python.exe" -m uvicorn app.main:app --reload"
start "SquatSpot Frontend" /D "%~dp0apps\web" cmd /k "npm run dev"

echo.
echo SquatSpot is starting:
echo   Frontend:  http://localhost:3000
echo   API:       http://localhost:8000
echo   OpenAPI:   http://localhost:8000/docs
echo   PostgreSQL localhost:5432

endlocal
