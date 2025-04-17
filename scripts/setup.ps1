if (Test-Path .\venv) {
    Remove-Item -Recurse -Force .\venv
}

py -3.12 -m venv venv

.\venv\Scripts\Activate.ps1

.\venv\Scripts\python.exe -m pip install --upgrade pip

pip install -r requirements.txt

# Install CUDA-enabled PyTorch (adjust the command to match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

Write-Host "Environment setup complete. To activate the environment in the future, run: .\venv\Scripts\Activate.ps1"
Write-Host "Now checking if CUDA is enabled..."

.\venv\Scripts\python.exe -c "import torch; print('CUDA device count:', torch.cuda.device_count()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'No CUDA GPU detected')"




# Exit on error
$ErrorActionPreference = "Stop"

# Desired Python version
$pythonVersion = "3.10.10"

# Create virtual environment
python -m venv isaac_env
.\isaac_env\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Isaac Lab and dependencies
pip install isaaclab[isaacsim,all]==2.0.2 --extra-index-url https://pypi.nvidia.com

# Confirm installation
Write-Host "Isaac Lab installation complete."
