$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvPath = if ($env:LUPI_VENV) { $env:LUPI_VENV } else { "C:\tmp\squatspot-lupi-py313" }
py -3.13 -m venv $EnvPath
if ($LASTEXITCODE -ne 0) { throw "Python 3.13 environment creation failed ($LASTEXITCODE)" }
& (Join-Path $EnvPath "Scripts\python.exe") -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed ($LASTEXITCODE)" }
& (Join-Path $EnvPath "Scripts\python.exe") -m pip install -r (Join-Path $Here "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "dependency installation failed ($LASTEXITCODE)" }
Write-Host "Ready. Activate with: $EnvPath\Scripts\Activate.ps1"
