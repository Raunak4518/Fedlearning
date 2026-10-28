# One-time setup of a Windows PC with an NVIDIA GPU for the FedECS-LT runs. No administrator rights needed.
# Started by setup_windows.bat (double-click). Safe to run again.
#  - checks the NVIDIA driver (560 or newer)
#  - installs Python 3.12.10 for this user only, if no Python 3.12 is found (python.org installer, InstallAllUsers=0)
#  - creates %USERPROFILE%\fedecs_env with PyTorch 2.14.1 (CUDA 12.6) and the fixed versions of requirements.txt
#  - checks that PyTorch sees the GPU
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Windows PowerShell 5.1 downloads far faster without the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$Here = $PSScriptRoot
$Venv = Join-Path $env:USERPROFILE "fedecs_env"
$Torch = @("torch==2.14.1", "torchvision==0.29.1")
$PyUrl = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"   # the last Windows installer of 3.12

function Say($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Stop-Setup($m) { Write-Host "`nSTOP: $m" -ForegroundColor Red; Read-Host "Press Enter to close"; exit 1 }

Say "NVIDIA driver"
$smi = (Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source
if (-not $smi -and (Test-Path "$env:SystemRoot\System32\nvidia-smi.exe")) { $smi = "$env:SystemRoot\System32\nvidia-smi.exe" }
if (-not $smi) { Stop-Setup "nvidia-smi not found: install the NVIDIA driver for your GPU from nvidia.com (560 or newer)." }
$q = & $smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
Write-Host $q
$drv = [int](($q | Select-Object -First 1).Split(",")[2].Trim().Split(".")[0])
if ($drv -lt 560) { Stop-Setup "driver $drv is too old for PyTorch 2.14 (needs 560 or newer): update it from nvidia.com." }

Say "Python 3.12"
$py = $null
foreach ($c in @((Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
                 "$env:ProgramFiles\Python312\python.exe")) { if (-not $py -and (Test-Path $c)) { $py = $c } }
if (-not $py) {
    try { $v = & py -3.12 -c "import sys; print(sys.executable)" 2>$null; if ($LASTEXITCODE -eq 0) { $py = $v.Trim() } } catch {}
}
if (-not $py) {
    $inst = Join-Path $env:TEMP "python-3.12.10-amd64.exe"
    Write-Host "Downloading Python 3.12.10 (about 25 MB) ..."
    Invoke-WebRequest -Uri $PyUrl -OutFile $inst -UseBasicParsing
    Write-Host "Installing it for this user only (no administrator rights) ..."
    Start-Process -Wait -FilePath $inst -ArgumentList "/quiet", "InstallAllUsers=0", "PrependPath=0",
        "Include_launcher=0", "Include_test=0", "Include_doc=0"
    $py = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
    if (-not (Test-Path $py)) { Stop-Setup "Python did not install into $py. Install Python 3.12 from python.org yourself ('Install for me only'), then run this again." }
}
& $py --version

Say "Python environment $Venv (10-20 minutes, about 5 GB)"
if (-not (Test-Path "$Venv\Scripts\python.exe")) { & $py -m venv $Venv }
$vpy = "$Venv\Scripts\python.exe"
& $vpy -m pip install -q --upgrade pip wheel setuptools
if ($LASTEXITCODE -ne 0) { Stop-Setup "pip could not update itself (internet?)." }
& $vpy -m pip install -q @Torch --index-url "https://download.pytorch.org/whl/cu126"
if ($LASTEXITCODE -ne 0) { Stop-Setup "PyTorch did not install (internet, or disk space: about 5 GB)." }
& $vpy -m pip install -q -r (Join-Path $Here "requirements.txt")
if ($LASTEXITCODE -ne 0) { Stop-Setup "the packages of requirements.txt did not install; send a screenshot of the error." }

Say "Check"
$env:PYTHONUTF8 = "1"
& $vpy -c @"
import sys, flwr, ray, torch, torchvision, sklearn, reportlab
print('Python', sys.version.split()[0], '| torch', torch.__version__, '| Flower', flwr.__version__, '| Ray', ray.__version__)
assert torch.cuda.is_available(), 'PyTorch does not see the GPU'
p = torch.cuda.get_device_properties(0)
print(f'GPU: {p.name}, {p.total_memory / 2**30:.1f} GB')
x = torch.randn(512, 512, device='cuda'); print('CUDA works:', bool(torch.isfinite(x @ x).all()))
"@
if ($LASTEXITCODE -ne 0) { Stop-Setup "the check failed (message above). If it names a missing DLL, install the 'Microsoft Visual C++ Redistributable' (x64) or ask for help." }

Write-Host "`nSetup finished. Next, in the folder fedecs_lt\local (README.md):" -ForegroundColor Green
Write-Host "  1. double-click run_check.bat   (once per PC, 30-90 minutes: what fits on this GPU)"
Write-Host "  2. double-click list_jobs.bat   (the jobs and which ones fit here)"
Write-Host "  3. run_job.bat JOB_NAME         (in a Command Prompt opened in that folder)"
Read-Host "Press Enter to close"
