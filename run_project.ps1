param(
    [int]$Port = 8501,
    [string]$OllamaModel = "llama3.2",
    [switch]$SkipDependencyInstall
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$PythonExe312 = Join-Path $RepoRoot ".venv312\Scripts\python.exe"
$PythonExe314 = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (Test-Path $PythonExe312) {
    $PythonExe = $PythonExe312
} elseif (Test-Path $PythonExe314) {
    $PythonExe = $PythonExe314
} else {
    throw "Virtual environment not found. Create one with: py -3.12 -m venv .venv312 (recommended)"
}

if (-not $SkipDependencyInstall) {
    Write-Host "Installing dependencies..." -ForegroundColor Cyan
    & $PythonExe -m pip install -r requirements.txt
}

$ollamaCmd = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollamaCmd) {
    $fallback = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path $fallback) {
        $ollamaCmd = $fallback
    }
}
if (-not $ollamaCmd) {
    throw "Ollama is not installed. Install with: winget install -e --id Ollama.Ollama"
}

function Test-OllamaUp {
    try {
        $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 10
        return $true
    } catch {
        return $false
    }
}

if (-not (Test-OllamaUp)) {
    Write-Host "Starting Ollama server..." -ForegroundColor Yellow
    Start-Process -FilePath $ollamaCmd -ArgumentList "serve" -WindowStyle Hidden
    $ready = $false
    foreach ($i in 1..25) {
        if (Test-OllamaUp) {
            $ready = $true
            break
        }
        Start-Sleep -Seconds 2
    }

    if (-not $ready) {
        Write-Host "Retrying Ollama startup with default launch mode..." -ForegroundColor Yellow
        Start-Process -FilePath $ollamaCmd -WindowStyle Hidden
        foreach ($i in 1..15) {
            if (Test-OllamaUp) {
                $ready = $true
                break
            }
            Start-Sleep -Seconds 2
        }
    }

    if (-not $ready) {
        throw "Ollama server did not come online on http://localhost:11434"
    }
}

Write-Host "Ensuring model '$OllamaModel' is available..." -ForegroundColor Cyan
& $ollamaCmd show $OllamaModel *> $null
if ($LASTEXITCODE -ne 0) {
    & $ollamaCmd pull $OllamaModel
}

Write-Host "Launching Streamlit on port $Port..." -ForegroundColor Green
& $PythonExe -m streamlit run src/app_frontend.py --server.port $Port
