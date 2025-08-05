@echo off
echo 🤖 AI Learning Quest - Starting...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist ai_learning_game.py (
    echo ❌ ai_learning_game.py not found in current directory
    echo Please run this script from the project root directory
    pause
    exit /b 1
)

echo ✅ Python found
echo 🔧 Installing/updating dependencies...
python -m pip install -r requirements.txt

echo.
echo 🚀 Launching AI Learning Quest...
echo 📱 The app will open in your browser automatically.
echo 🛑 Press Ctrl+C in this window to stop the application.
echo.

REM Run the game
python run_game.py

echo.
echo 👋 Thanks for playing AI Learning Quest!
pause
