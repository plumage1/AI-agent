param(
    [switch]$SkipRedis,
    [switch]$InstallDeps,
    [switch]$NoReload
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$EnvFile = Join-Path $RepoRoot ".env"
$EnvExample = Join-Path $RepoRoot ".env.example"
$RequirementsFile = Join-Path $RepoRoot "requirements.txt"

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Write-Warn([string]$Message) {
    Write-Host "Warning: $Message" -ForegroundColor Yellow
}

function Test-CommandAvailable([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Ensure-EnvFile {
    if (Test-Path $EnvFile) {
        return
    }

    if (-not (Test-Path $EnvExample)) {
        throw "Missing .env.example in project root."
    }

    Copy-Item $EnvExample $EnvFile
    Write-Step "Created .env from .env.example"
}

function Ensure-Venv {
    if (Test-Path $VenvPython) {
        return
    }

    if (-not (Test-CommandAvailable "python")) {
        throw "Python is not installed or not available in PATH."
    }

    Write-Step "Creating virtual environment"
    & python -m venv .venv
}

function Test-CoreDependenciesInstalled {
    & $VenvPython -c "import fastapi, uvicorn, dotenv" *> $null
    return $LASTEXITCODE -eq 0
}

function Ensure-Dependencies {
    if (-not (Test-Path $RequirementsFile)) {
        throw "Missing requirements.txt in project root."
    }

    if ($InstallDeps -or -not (Test-CoreDependenciesInstalled)) {
        Write-Step "Installing Python dependencies"
        & $VenvPython -m pip install -r $RequirementsFile
    }
}

function Test-CodexCliProvider {
    if (-not (Test-Path $EnvFile)) {
        return $false
    }

    $match = Select-String -Path $EnvFile -Pattern '^LLM_PROVIDER=(.+)$' | Select-Object -First 1
    if (-not $match) {
        return $false
    }

    return $match.Matches[0].Groups[1].Value.Trim() -eq "codex_cli"
}

function Warn-IfCodexMissing {
    $usesCodexCli = Test-CodexCliProvider
    if ($usesCodexCli -and -not (Test-CommandAvailable "codex")) {
        Write-Warn "codex command was not found. Chat endpoints may fail until Codex CLI is installed or .env is updated."
    }
}

function Use-LocalFallbackMode {
    $env:REDIS_ENABLED = "false"
    $env:LANGGRAPH_CHECKPOINTER_BACKEND = "memory"
    Write-Warn "Using in-memory fallback mode for this run."
}

function Try-StartRedis {
    if ($SkipRedis) {
        Write-Warn "Skipping Redis startup by request."
        Use-LocalFallbackMode
        return
    }

    if (-not (Test-CommandAvailable "docker")) {
        Write-Warn "Docker was not found. Continuing without Redis."
        Use-LocalFallbackMode
        return
    }

    try {
        docker info *> $null
    }
    catch {
        Write-Warn "Docker daemon is not running. Continuing without Redis."
        Use-LocalFallbackMode
        return
    }

    Write-Step "Starting Redis container"
    try {
        docker compose up -d redis | Out-Host
    }
    catch {
        Write-Warn "Redis container could not be started. Continuing without Redis."
        Use-LocalFallbackMode
    }
}

Write-Step "Preparing project"
Ensure-EnvFile
Ensure-Venv
Ensure-Dependencies
Warn-IfCodexMissing
Try-StartRedis

Write-Step "Starting FastAPI"
Write-Host "UI:   http://127.0.0.1:8000/ui"
Write-Host "Docs: http://127.0.0.1:8000/docs"

$UvicornArgs = @(
    "-m",
    "uvicorn",
    "api.main:app",
    "--host",
    "127.0.0.1",
    "--port",
    "8000"
)

if (-not $NoReload) {
    $UvicornArgs += "--reload"
}

& $VenvPython @UvicornArgs
