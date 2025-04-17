# revolution

# CS 486 - Reinforcement Learning Project
## Spring 2025

Team Members:
 - Krisi Hristova <kh3339@drexel.edu>
 - Safa Obuz <seo52@drexel.edu>
 - Francisco Cruz-Urbanc <fjc59@drexel.edu>

# Installation

Run `setup.ps1` if on Windows or `setup.sh` if on Linux (Note Mac is not supported as there is no CUDA support)

If in VSCode, ***press cmd + shift + P***, search ***"Python: Select Interpreter"*** and select the Python interpreter that belongs to your venv folder.
- Linux: ***CS435FinalProject/venv/bin/python***
- Windows: ***CS435FinalProject\venv\Scripts\python.exe***

For additional details see:
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html

# Project Structure

    revolution_project/
    ├── assets/
    │   ├── urdf/
    │   │   ├── generation_01/
    │   │   │   ├── agent_v1.urdf
    │   │   │   ├── agent_v2.urdf
    │   │   │   └── ...
    │   │   ├── generation_02/
    │   │   │   ├── agent_v3.urdf
    │   │   │   └── agent_v4.urdf
    │   │   │   └── ...
    │   │   └── final_designs/
    │   ├── meshes/
    │   │   ├── limbs/
    │   │   └── body_parts/
    │   └── textures/
    ├── configs/
    │   ├── sim_config.yaml          # General Isaac Sim config 
    │   ├── training_config.yaml     # RL algorithm hyperparameters 
    │   └── evolution_config.yaml    # Evolutionary strategy parameters
    ├── scripts/
    │   ├── setup.sh                 # Linux setup script
    │   ├── setup.ps1                # Windows setup script
    ├── src/
    │   ├── train.py                 # Main script to train a single agent morphology
    │   ├── evolve.py                # Script managing evolution: running training selecting morphologies
    │   ├── evaluate.py              # Script for evaluating morphologies and selecting survivors
    │   └── utils/
    │       ├── morphology.py        # URDF swapping/loading utilities
    │       └── logging.py           # Integration with Weights & Biases or custom logger
    ├── outputs/
    │   ├── logs/                    # Training logs (W&B, CSV, ect)
    │   ├── checkpoints/             # Saved policy/model checkpoints per morphology
    │   └── evaluations/             # Morphology performance evaluations and metrics
    ├── docs/
    │   ├── reports/
    │   │   ├── pre_proposal.pdf
    │   │   ├── proposal.pdf     
    │   │   └── final_report.pdf
    │   └── references/
    │       └── bibliography.bib
    │       └── ieee_fullname.bst    # Details how to format tex citations
    ├── isaac_env/                   # Python virtual environment created via venv
    ├── .gitignore                   # Ignore checkpoints, logs, large binary files
    ├── requirements.txt             # Python packages (Isaac Lab, Gymnasium,W&B, Tianshou, MuJoCo, etc.)
    └── README.md                    # Instructions for running, setup, and reproducing experiments
