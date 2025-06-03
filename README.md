# revolution

# CS 486 - Reinforcement Learning Project
## Spring 2025

Team Members:
 - Krisi Hristova <kh3339@drexel.edu>
 - Safa Obuz <seo52@drexel.edu>
 - Francisco Cruz-Urbanc <fjc59@drexel.edu>

# Installation

Run `setup.ps1` if on Windows or `setup.sh` if on Linux (Note Mac is not supported as there is no CUDA support)

If in VSCode, ***press cmd + shift + P***, search ***"Python: Select Interpreter"*** and select the venv folder (or manually source via the command line)

Additionally, run `pip install requirement.txt` after running the installation script

For additional details see:  
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html

# Project Structure

    revolution_project/
    ├── assets/
    │   ├── urdf/
    │   │   ├── design_01/
    │   │   ├── design_02/
    ├── configs/
    │   ├── sim_config.yaml          # General Isaac Sim config 
    │   ├── training_config.yaml     # RL algorithm hyperparameters 
    │   └── evolution_config.yaml    # Evolutionary strategy parameters
    ├── setup/
    │   ├── setup.sh                 # Linux setup script
    │   ├── setup.ps1                # Windows setup script
    ├── src/
    │   ├── train.py                 # Main script to train a single agent morphology
    │   ├── evolve.py                # Script managing evolution: running training selecting morphologies
    │   ├── evaluate.py              # Script for evaluating morphologies and selecting survivors
    │   └── utils/
    │       ├── morphology.cpp       # URDF swapping/loading utilities
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
    ├── requirements.txt             # Python packages (Isaac Lab, Gymnasium, W&B, Tianshou, MuJoCo, etc.)
    └── README.md                    # Instructions for running, setup, and reproducing experiments

# Graphs
Analysis graphs from training are located in `/figures/graphs/`.

# Custom Morphologies
Custom URDF-based morphologies are located in `/assets/urdf/`.

Images of some of our custom morphologies are located in `/figures/morphologies/`.

### Morphologies specifically used for training and analysis:
- caterpillar.urdf
- stompy_4.urdf
- frankie_jr_1.3.urdf
- frankie_jr.urdf

### Morphology created by simple evolution:
The four-legged `stompy` morphology was created via *evolution* by adding a new limb at each run through. The collection of morphologies is located in `/assets/urdf/stompy/stompy-capsules/`. 

Each `stompy_<numLegs>.urdf` file represents a torso with a different number of legs that was sequentially developed by passing one `stompy` URDF file through `evolve` which attempts to add a limb at a corner of the robot each iteration. This was performed for 5 iterations resulting in 5 different evolved `stompy` morphologies.



## Extra
**Link to visualize URDFs:** http://urdf.robotsfan.com/ 

**Usage**: Upload a URDF file to visualize its structure while making initial custom morphologies before using for training. 