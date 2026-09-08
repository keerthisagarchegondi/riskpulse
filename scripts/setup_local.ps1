param(
    [switch]$StartServices,
    [switch]$SkipComposeValidation
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Resolve-Python {
    $candidates = @()
    if ($env:PYTHON_BIN) {
        $candidates += $env:PYTHON_BIN
    }
    $candidates += @(
        "python",
        "python3",
        "py",
        (Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe")
    )

    foreach ($candidate in $candidates) {
        try {
            $command = Get-Command $candidate -ErrorAction Stop
            & $command.Source --version *> $null
            return $command.Source
        } catch {
            if (Test-Path $candidate) {
                & $candidate --version *> $null
                return $candidate
            }
        }
    }

    throw "Python 3.11+ was not found. Install Python or set PYTHON_BIN."
}

$python = Resolve-Python
$argsList = @("scripts/setup_local.py")
if ($StartServices) {
    $argsList += "--start-services"
}
if ($SkipComposeValidation) {
    $argsList += "--skip-compose-validation"
}

Push-Location $Root
try {
    & $python @argsList
} finally {
    Pop-Location
}
