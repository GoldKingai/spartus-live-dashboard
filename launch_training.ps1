# Spartus Training AI — Launch Script
# Run this in PowerShell: .\launch_training.ps1
# Or right-click > Run with PowerShell

param(
    [int]$Weeks = 530,
    [switch]$Resume,
    [switch]$NoDashboard,
    [switch]$NoTensorBoard
)

$pythonPath = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
$scriptPath = Join-Path $PSScriptRoot "scripts\train.py"
$tbPath = Join-Path $PSScriptRoot "venv\Scripts\tensorboard.exe"
$logDir = Join-Path $PSScriptRoot "storage\logs\tensorboard"

Write-Host ""
Write-Host "  SPARTUS TRADING AI" -ForegroundColor Cyan
Write-Host "  ==================" -ForegroundColor DarkCyan
Write-Host ""

# Launch TensorBoard (detailed metrics in browser)
if (-not $NoTensorBoard) {
    Write-Host "  Starting TensorBoard dashboard..." -ForegroundColor Gray
    $tbProcess = Start-Process -FilePath $tbPath `
        -ArgumentList "--logdir", $logDir, "--port", "6006", "--reload_interval", "5" `
        -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 2
    Start-Process "http://localhost:6006"
    Write-Host "  TensorBoard: http://localhost:6006 (opened in browser)" -ForegroundColor Green
    Write-Host ""
}

Write-Host "  Weeks: $Weeks | Resume: $Resume" -ForegroundColor Gray
Write-Host "  Qt dashboard will open automatically." -ForegroundColor DarkGray
Write-Host ""

# Build training args
$trainArgs = @($scriptPath, "--weeks", $Weeks)
if ($Resume) { $trainArgs += "--resume" }
if ($NoDashboard) { $trainArgs += "--no-dashboard" }

# Launch training (Qt dashboard opens as native window)
Set-Location $PSScriptRoot
& $pythonPath @trainArgs

# Cleanup TensorBoard on exit
if (-not $NoTensorBoard -and $tbProcess -and -not $tbProcess.HasExited) {
    Write-Host ""
    Write-Host "  TensorBoard still running at http://localhost:6006" -ForegroundColor Gray
    Write-Host "  Press T to stop TensorBoard, or any other key to leave it running..." -ForegroundColor Gray
    $key = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    if ($key.Character -eq 't' -or $key.Character -eq 'T') {
        Stop-Process -Id $tbProcess.Id -Force -ErrorAction SilentlyContinue
        Write-Host "  TensorBoard stopped." -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "  Training complete. Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
