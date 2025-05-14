# TO DO... Verify this actually does what it needs to. 

# 1) Open an elevated PowerShell (Win-X → “Windows Terminal (Admin)”).

# 2) Find the path to NVIDIA’s Vulkan ICD file automatically
#    – it’s always called “nvidia_icd.json”, but the folder name changes
#      with every driver update.

$nvIcd = Get-ChildItem -Path "C:\Windows\System32\DriverStore\FileRepository" `
                       -Filter "nvidia_icd.json" -Recurse `
                       -ErrorAction SilentlyContinue |
          Select-Object -First 1 -ExpandProperty FullName

if (-not $nvIcd) {
    Write-Host "❌  Could not locate nvidia_icd.json – is the NVIDIA driver installed?" -ForegroundColor Red
    return
}

Write-Host "✅  NVIDIA ICD found at:`n $nvIcd" -ForegroundColor Green

# 3) Tell THIS PowerShell session to use ONLY that ICD
$env:VK_ICD_FILENAMES = $nvIcd

# 4) (Optional) Confirm the override
Write-Host "`nCurrent VK_ICD_FILENAMES = $env:VK_ICD_FILENAMES`n"

# 5) Launch Isaac Lab from the same shell.
#    Replace the path below if your repo lives elsewhere.

cd "C:\Code\revolution\IsaacLab"
python scripts\reinforcement_learning\skrl\train.py `
       task=Isaac-Ant-v0 `
       renderer=Storm        # ← much lighter than RTX Real-Time; comment out if you want RTX
