<# 
    Author: Safa Obuz
    Date: 5/9/2025
    One-click Isaac Lab install and setup
#>

# Exit on error
$ErrorActionPreference = "Stop"

# Paths
$ScriptDir = $PSScriptRoot #.../revolution/setup
$RepoRoot = (Resolve-Path "$ScriptDir\..").Path
$EnvDir = Join-Path $RepoRoot "isaac_env"
$LabDir = Join-Path $RepoRoot "IsaacLab"

# Create Venv
if (Test-Path $EnvDir) { Remove-Item $EnvDir -Recurse -Force }
& py -3.10 -m venv $EnvDir
& (Join-Path $EnvDir "Scripts\Activate.ps1")

# Upgrade pip
python -m pip install --upgrade pip

#This is directly from the Isaac Lab docs, with these exact version numbers
Write-Host "Installing Torch..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

Write-Host "Now checking if CUDA is enabled..."
python -c "import torch, platform, sys; \
print(f'Torch {torch.__version__} CUDA available:', torch.cuda.is_available()); \
print('Device count:', torch.cuda.device_count()); \
print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU detected')"

Write-Host "Installing Isaac Sim..."
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

Write-Host "Cloning + Installing Isaac Lab..."
if (-not (Test-Path $LabDir)) {
    git clone https://github.com/isaac-sim/IsaacLab.git $LabDir
}
Set-Location $LabDir
& ".\isaaclab.bat" --install
Set-Location $RepoRoot

Write-Host "Environment setup complete."
Write-Host "Activate the environment next time with:"
Write-Host "    & '$(Join-Path $EnvDir "Scripts\Activate.ps1")'"