$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

Write-Host "Creating Python 3.12 environment (.venv312)..." -ForegroundColor Cyan
if (-not (Test-Path ".venv312\Scripts\python.exe")) {
    py -3.12 -m venv .venv312
}

$PythonExe = Join-Path $RepoRoot ".venv312\Scripts\python.exe"

Write-Host "Installing project dependencies (CrewAI + LangGraph included for Python 3.12)..." -ForegroundColor Cyan
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -r requirements.txt

Write-Host "Validating CrewAI availability..." -ForegroundColor Cyan
& $PythonExe -c "from src.crew_runtime import is_crewai_available; print('CrewAI available:', is_crewai_available())"

Write-Host "Done. Run CrewAI-primary pipeline with:" -ForegroundColor Green
Write-Host "  .venv312\Scripts\python.exe -m src.main --runtime auto --ollama" -ForegroundColor Green
