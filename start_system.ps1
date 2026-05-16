# start_system.ps1
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   FRAUD DETECTION SYSTEM - AUTO LAUNCHER" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$ProjectPath = "D:\Project\fraud_detection_complete"
Set-Location $ProjectPath

Write-Host "`n[1/3] Starting API Server..." -ForegroundColor Yellow
Start-Process -NoNewWindow -FilePath "cmd.exe" -ArgumentList "/k title API Server && .venv\Scripts\activate && uvicorn application.bank_api:app --reload --port 8000"

Start-Sleep -Seconds 3

Write-Host "[2/3] Starting Dashboard..." -ForegroundColor Yellow
Start-Process -NoNewWindow -FilePath "cmd.exe" -ArgumentList "/k title Dashboard && .venv\Scripts\activate && streamlit run application/dashboard.py"

Start-Sleep -Seconds 2

Write-Host "[3/3] Opening Browser..." -ForegroundColor Yellow
Start-Sleep -Seconds 3
Start-Process "http://localhost:8501"

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "✅ ALL SERVICES STARTED!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "`n📡 API Server: http://localhost:8000" -ForegroundColor White
Write-Host "📚 API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "🎨 Dashboard: http://localhost:8501" -ForegroundColor White
Write-Host "`n💡 To stop: Close the terminal windows" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Green