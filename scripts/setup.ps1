#Requires -Version 5.1
<#
.SYNOPSIS
    Konfiguruje srodowisko deweloperskie projektu RemoteDroids na Windows.

.PARAMETER SkipInstall
    Pomija instalacje zaleznosci pip.

.PARAMETER RecreateVenv
    Usuwa i odtwarza virtualenv od zera przed instalacja.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
    powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -RecreateVenv
    powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -SkipInstall
#>
[CmdletBinding()]
param(
    [switch]$SkipInstall,
    [switch]$RecreateVenv
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Przejdz do katalogu glownego projektu (rodzic katalogu scripts/)
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
Write-Host "Katalog projektu: $ProjectRoot" -ForegroundColor Cyan

# --- Sprawdz Python ---
$PythonCmd = $null
foreach ($candidate in @('python', 'python3')) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
        $ver = & $candidate --version 2>&1
        if ($ver -match 'Python 3\.(\d+)') {
            if ([int]$Matches[1] -ge 13) {
                $PythonCmd = $candidate
                Write-Host "Znaleziono $ver" -ForegroundColor Green
                break
            }
        }
    }
}
if (-not $PythonCmd) {
    Write-Error "Nie znaleziono Pythona 3.13+. Zainstaluj Python i dodaj go do PATH."
}

# --- Virtualenv ---
$VenvDir = Join-Path $ProjectRoot '.venv'

if ($RecreateVenv -and (Test-Path $VenvDir)) {
    Write-Host "Usuwam istniejacy .venv..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $VenvDir
}

if (-not (Test-Path $VenvDir)) {
    Write-Host "Tworzenie virtualenv w .venv..." -ForegroundColor Cyan
    & $PythonCmd -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Write-Error "Nie udalo sie utworzyc virtualenv." }
}
else {
    Write-Host "Virtualenv juz istnieje (.venv)." -ForegroundColor Green
}

$PipExe  = Join-Path $VenvDir 'Scripts\pip.exe'
$PyExe   = Join-Path $VenvDir 'Scripts\python.exe'

# --- Instalacja zaleznosci ---
if (-not $SkipInstall) {
    Write-Host "Aktualizacja pip..." -ForegroundColor Cyan
    & $PyExe -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { Write-Error "Nie udalo sie zaktualizowac pip." }

    Write-Host "Instalacja zaleznosci deweloperskich (requirements-dev.txt)..." -ForegroundColor Cyan
    & $PipExe install -r (Join-Path $ProjectRoot 'requirements-dev.txt')
    if ($LASTEXITCODE -ne 0) { Write-Error "Instalacja zaleznosci nie powiodla sie." }

    Write-Host "Zaleznosci zainstalowane pomyslnie." -ForegroundColor Green
}
else {
    Write-Host "Pomijam instalacje zaleznosci (-SkipInstall)." -ForegroundColor Yellow
}

# --- Uruchomienie serwera ---
Write-Host ""
Write-Host "=== Konfiguracja zakonczona - uruchamiam serwer ===" -ForegroundColor Green
Write-Host "Uruchamianie: $PyExe src\server\init.py" -ForegroundColor Cyan
& $PyExe (Join-Path $ProjectRoot 'src\server\init.py')
