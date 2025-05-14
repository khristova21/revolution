# Installation

TODO: Add FFMPEG insatllation to the script, so we can record videos of the agents.

Simply run the setup script on Windows or Linux for a streamlined install. Note: 50-series Nvidia cards [require a nightly build of PyTorch](https://developer.nvidia.com/cuda-gpus) as of 5/13/2025. This is incorporated into the script.

# Common Issues

This is how to debug common issues that comes with the Isaac Lab installation process. The setup scripts attempts to streamline the process, but there may be a few kinks. Feel free to follow the official [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html#quick-installation-guide) for custom installation (for instance, using conda or miniconda versus our script which uses pip).

## Linux & Windows Issues

### Black Viewport

Activate the virtual enviroment `isaac_env` and run `isaacsim` to verify the simulator runs as expected.  
Ensure the simulation is ran on your discrete graphics processing unit, and verify you meet the [minimum requirements](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html). Use the Isaac Sim Compatibility Checker if you are unsure.  
If you are training an agent using one of the provided [reinforcement learning libraries](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html), ensure you are referring to the correct task. For instance, the RL-games library may refer to the "ant" agent as `Ant` and SKRL or RSL-RL as something else. However, Isaac Lab provides a wrapper script `train.py` (lives in different directories) and refers to the ant agent as `Isaac-Ant-v0` irrespective of the library being used. If you provide the incorrect name for the task, the viewport will be black until the program exits. 

### Could not find Isaac Sim VSCode settings

Create `.vscode/settings.json` in the root of the project (not isaac_env)  
Isaac Lab also has their own `.vscode` folder. See for reference
```
{
  // Use the interpreter in the virtual-env
  "python.defaultInterpreterPath": "${workspaceFolder}/isaac_env/python.exe",

  // Tell Pylance where the Omniverse / Isaac modules live
  "python.analysis.extraPaths": [
    "${workspaceFolder}/isaac_env/Lib/site-packages",
    "${workspaceFolder}/isaac_env/Lib/site-packages/isaacsim"
  ]
}
```

### Missing ICD

## Linux

### Missing ICD

TODO:  
[Nvidia ICD JSON missing Linux](https://github.com/NVIDIA/nvidia-container-toolkit/issues/767)  
[Black window missing ICD.json](https://forums.developer.nvidia.com/t/isaac-gym-black-window-missing-nvidia-icd-json-segmentation-fault/210175)  
[nvidia-vulkan-container](https://github.com/Babbleshack/nvidia-vulkan-container/blob/master/nvidia_icd.json) (5 years old commit, may be different now)  

`nvidia_icd.json`
```
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.1.99"
    }
}
```

## Windows

### Shell Doesn't Allow Script to Run
Run the shell with administrator privileges  
Modify the ExecutionPolicy for your current shell (will revert once the session is closed)
```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```
### 

### VK_EXT_memory_budget
Disable or remove the [OpenCL™, OpenGL®, and Vulkan® Compatibility Pack](https://apps.microsoft.com/detail/9nqpsl29bfff?hl=en-US&gl=US) from Windows  
See [Memory Errors Issue](https://github.com/isaac-sim/IsaacLab/discussions/2046) for reference

### Nvidia ICD JSON missing
TODO: Run nvidia_driver_fix.ps1