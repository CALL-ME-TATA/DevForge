# Start (or resume) training in a Linux Docker container on this machine.
# Needs: Docker Desktop (WSL2 backend) running + NVIDIA driver. Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File training\start_training.ps1 [-Hours 8] [-Extra "--lr 1.5e-4 --ent 0.005"]
param([double]$Hours = 8, [string]$Extra = "")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

docker info *> $null
if ($LASTEXITCODE -ne 0) { throw "Docker is not running. Start Docker Desktop and retry." }

$cores = (Get-CimInstance Win32_Processor | Measure-Object NumberOfLogicalProcessors -Sum).Sum
$envs = [Math]::Min(48, 2 * $cores)
Write-Host "Logical cores: $cores -> $envs parallel games"

if (-not (docker images -q doom-train)) {
    Write-Host "Building training image (downloads ~4 GB, 5-15 min)..."
    docker build -t doom-train -f training/Dockerfile training
    if ($LASTEXITCODE -ne 0) { throw "image build failed" }
}
docker run --rm --gpus all doom-train python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "GPU not visible inside Docker (update NVIDIA driver / enable WSL2 GPU)" }

New-Item -ItemType Directory -Force training/ckpt | Out-Null
if (Test-Path training/ckpt/latest.pt) { Write-Host "Resuming from training/ckpt/latest.pt" }
docker rm -f doomrun *> $null
docker run -d --name doomrun --restart on-failure --gpus all --shm-size 4g -v "${root}:/w" doom-train `
    sh -c "python -u training/train.py --envs $envs --hours $Hours --resume $Extra 2>&1 | grep --line-buffered -v -e nested -e UserWarning -e 'output = torch' | tee -a training/ckpt/train.log"
if ($LASTEXITCODE -ne 0) { throw "failed to start container" }
Write-Host "Training started. Watch progress:  Get-Content training\ckpt\train.log -Wait -Tail 5"
