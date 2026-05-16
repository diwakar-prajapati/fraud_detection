@echo off
title Fraud Detection System Launcher
echo ============================================================
echo    FRAUD DETECTION SYSTEM - AUTO LAUNCHER
echo ============================================================
echo.

cd /d D:\Project\fraud_detection_complete

echo [1/3] Starting API Server...
start "API Server" cmd /k "title API Server && .venv\Scripts\activate && uvicorn application.bank_api:app --reload --port 8000"

timeout /t 3 /nobreak >nul

echo [2/3] Starting Dashboard...
start "Dashboard" cmd /k "title Dashboard && .venv\Scripts\activate && streamlit run application/dashboard.py"

timeout /t 2 /nobreak >nul

echo [3/3] Starting Test Monitor (Optional)...
start "Test Monitor" cmd /k "title Test Monitor && .venv\Scripts\activate && echo Ready to test! Use: python test_api_fixed.py"

echo.
echo ============================================================
echo ✅ ALL SERVICES STARTED!
echo ============================================================
echo.
echo 📡 API Server: http://localhost:8000
echo 📚 API Docs: http://localhost:8000/docs
echo 🎨 Dashboard: http://localhost:8501
echo.
echo 💡 To stop: Close the terminal windows
echo ============================================================