# Author: Safa Obuz
# Date: 5/22/2025
# Purpose: Setting Up Enviroment (for now, eventually here will reside the full training loop)

import argparse

# Isaac Lab APIs
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Training script to train pool of agents")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Once the simulation app runs, we can import 
from isaaclab.sim import SimulationCfg, SimulationContext

def runSimlation():
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimlationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    sim.reset()
    print("Starting simulation")

    while simulation_app.is_running():
        sim.set()

if __name__ == "__main__":
    runSimlation()
    simulation_app.close()