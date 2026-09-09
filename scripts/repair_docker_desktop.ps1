param(
    [string]$DockerUser = "ROHITH-JOBSYME\codexsandboxoffline",
    [string]$DockerRunDirectory = "C:\Users\VSC\AppData\Local\Docker\run",
    [switch]$SkipGroupUpdate,
    [switch]$SkipStaleSocketCleanup
)

$ErrorActionPreference = "Stop"

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-IsAdmin)) {
    throw "Run this script from an Administrator PowerShell window."
}

if (-not $SkipGroupUpdate) {
    $members = net localgroup docker-users 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "The local docker-users group was not found. Reinstall or repair Docker Desktop first."
    }

    if ($members -notmatch [regex]::Escape($DockerUser)) {
        net localgroup docker-users $DockerUser /add
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to add $DockerUser to docker-users."
        }
        Write-Host "Added $DockerUser to docker-users."
    } else {
        Write-Host "$DockerUser is already in docker-users."
    }
}

if (-not $SkipStaleSocketCleanup) {
    Get-Process "Docker Desktop", "com.docker.backend", "docker", "dockerd" -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue

    $service = Get-Service com.docker.service -ErrorAction SilentlyContinue
    if ($service -and $service.Status -ne "Stopped") {
        Stop-Service com.docker.service -Force -ErrorAction SilentlyContinue
    }

    $staleSocket = Join-Path $DockerRunDirectory "dockerInference"
    if (Test-Path -LiteralPath $staleSocket) {
        Remove-Item -LiteralPath $staleSocket -Force
        Write-Host "Removed stale Docker inference socket: $staleSocket"
    } else {
        Write-Host "No stale Docker inference socket found."
    }
}

$dockerDesktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
if (-not (Test-Path -LiteralPath $dockerDesktop)) {
    throw "Docker Desktop executable not found at $dockerDesktop."
}

Start-Process -FilePath $dockerDesktop
Write-Host "Docker Desktop launched. Sign out/in or restart Codex if group membership was changed."
