# Author: Safa Obuz
# Date: 5/29/2025
# Purpose: Setting Up Enviroment (for now, eventually here will reside the full training loop)

# Python Modules
import argparse, os, contextlib, torch, time
import numpy as np
from typing import Type

# Parsing CLI arguments
parser = argparse.ArgumentParser(description="Training script to train pool of agents")
# Simulation App arguments
parser.add_argument("--width", type=int, default=1280, help="Width of the viewport and generated images. Defaults to 1280")
parser.add_argument("--height", type=int, default=720, help="Height of the viewport and generated images. Defaults to 720")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn")
# Miscellanous
parser.add_argument("--urdf", type=str, default="assets/urdf/test2.urdf", help="Path to URDF to load")
parser.add_argument("--wiggle", type=bool, default=False, help="Start enviroment where agents move randomly")

# Start simulation App before importing APIs
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab API
import isaaclab.sim as sim_utils # for helper functions
from isaaclab.sim import SimulationCfg, SimulationContext # used frequently
 
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.configclass import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

# Nvidia Omniverse AP
import isaacsim.core.utils.prims as prim_utils
import omni.client, omni.usd
import carb

def configure_agent_minimally(usd_path: str, prim_path: str) -> ArticulationCfg:
    """Configure agent with minimal settings
    Args: 
        usd_path: Path to the USD file
        prim_path: Path to the primitive in the stage (e.g., "{ENV_REGEX_NS}/Agent")
    Returns:
        Articulation Config for the agent
    """
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path), # Generates SpawnerCfg
        # Dictionary of Actuator Configs. 'None' will force defaults found in USD asset
        actuators={"joints": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
    )

def configure_agent(usd_path: str, prim_path: str) -> ArticulationCfg:
    """Configure agent with moderate settings
    Args: 
        usd_path: Path to the USD file
        prim_path: Path to the primitive in the stage (e.g., "{ENV_REGEX_NS}/Agent")
    Returns:
        Articulation Config for the agent
    """
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
        ),
        # init_state=Articulation.Cfg.InitialStateCfg(joint_pos={},pos=(x,y,z))
        actuators={"joints": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
    )

def design_scene() -> None:
    """Designs the scene by spawning ground plane, light, and agents from USD file"""
    # Ground Plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # Scene Light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75)
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # Import URDF
    try:
        usd_path = importURDF(args_cli.urdf, "assets/usd")
        print(f"Successfully converted URDF to {usd_path}")
        
        # Now spawn the agent using the actual USD file
        cfg_agent = sim_utils.UsdFileCfg(
            usd_path=usd_path
        )
        cfg_agent.func("/World/Agents", cfg_agent, translation=(0.0, 0.0, 1.0))
    
    except Exception as e:
        print(f"Error importing URDF: {e}")
    
def runSimulationWiggle(sim: SimulationContext, scene: InteractiveScene) -> None:
    """Test simulation environment where agent moves randomly
    Args:
        sim: Config for currently running simulation started in main
        scene: 
    """
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    start_real_time = time.time()

    num_agents = args_cli.num_envs
    num_joints = scene["agent"].num_joints
    device = scene["agent"].device # Every tensor needs to run on CUDA all together
        
    print(f"[INFO] Number of joints: {num_joints}")
    print(f"[INFO] Joint names: {scene['agent'].data.joint_names}")
    print(f"[INFO] Physics timestep: {sim_dt}")

    frequencies = 1.0 + 0.4 * torch.rand(num_agents, num_joints, device=device)  # 1.0 - 1.4 Hz
    amplitudes = 1.0 + 1.0 * torch.rand(num_agents, num_joints, device=device)   # 1.0 - 2.0 rad/s
    phases = 2 * torch.pi * torch.rand(num_agents, num_joints, device=device)    # 0 - 2π phase offset

    reset_interval = 10.0 # Reset every 10 *simulation* seconds
    last_reset_time = 0.0
        
    while simulation_app.is_running():
        if sim_time - last_reset_time >= reset_interval:
            last_reset_time = sim_time

            root_agent_state = scene["agent"].data.default_root_state.clone()
            root_agent_state[:, :3] += scene.env_origins

            scene["agent"].write_root_pose_to_sim(root_agent_state[:, :7])
            scene["agent"].write_root_velocity_to_sim(root_agent_state[:, 7:])

            joint_pos, joint_vel = (
                scene["agent"].data.default_joint_pos.clone(),
                scene["agent"].data.default_joint_vel.clone(),
            )
            scene["agent"].write_joint_state_to_sim(joint_pos, joint_vel)

            scene.reset()
            print(f"[INFO] Resetting agent at t={sim_time:.2f}")

        # Wiggle - apply random joint actions
        # Using sine waves at different frequencies for each joint
        phase_angles = 2 * torch.pi * frequencies * sim_time + phases
        velocities = amplitudes * torch.sin(phase_angles)
        
        # set_joint_effort_target(efforts) needs damping, stiff = 0
        scene["agent"].set_joint_velocity_target(velocities)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        if count % int(1.0 / sim_dt) == 0:
            real_time_elapsed = time.time() - start_real_time
            real_time_factor = sim_time / real_time_elapsed
            print(f"[INFO] Sim time: {sim_time:.1f}s, Real time: {real_time_elapsed:.1f}s, RTF: {real_time_factor:.2f}x")

def runSimlation() -> None:
    """Main simulation loop with URDF loading"""

    # Config
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)

    design_scene()

    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Reset simulation to apply changes
    sim.reset()
    print("Starting simulation. Ctrl+C to stop")

    while simulation_app.is_running():
        sim.step()

def importURDF(urdf_path: str, usd_dir: str) -> str:
    """
    Imports a URDF into an Isaac Lab Scene that is currently running
    Args:
        urdf_path: Path to the URDF file
        usd_dir: Directory where the USD file should be saved
    Returns: 
        Path to the converted USD file
    """

    # If the file path is relative, make it absolute
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)
    if not os.path.isabs(usd_dir):
        usd_dir = os.path.abspath(usd_dir)
    
    # Backslash turned to forward slash on Windows for omniverse API
    forward_slash_urdf_path = urdf_path.replace(os.sep, "/")
    forward_slash_usd_dir = usd_dir.replace(os.sep, "/") # If needed

    # We need to interface with Omniverse Nucleus Server which keeps track of OpenUSD data
    # omni.client.stat returns Tuple[Result, ListEntry] where Result.OK means url/file was found
    result, _ = omni.client.stat(forward_slash_urdf_path)
    if not (result == omni.client.Result.OK): # If Nucleus does not find the file
        raise ValueError(f"URDF file not found: {urdf_path}")
    
    # Generate USD filename based on URDF filename (Seems reasonable?)
    urdf_filename = os.path.basename(urdf_path)
    usd_filename = os.path.splitext(urdf_filename)[0] + ".usd"
    os.makedirs(os.path.dirname(usd_dir), exist_ok=True)
    
    # Stored in instanceable format by default, basically a wrapper to isaacsim.asset.importer.urdf
    # Instancing allows for better performance when loading the same agent in many training enviroments
    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=usd_dir,
        usd_file_name=usd_filename,
        fix_base=False,
        merge_fixed_joints=False,
        force_usd_conversion=True,
        root_link_name="segment_0",
        self_collision=False,                 # For stability
        replace_cylinders_with_capsules=True, # Ensures capsule *collider* for physics (not visual) 
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=100.0,
                damping=1.0,
            ),
            target_type="velocity",
            drive_type="force" # defaults to this, but very important that it is "force"
        )
    )

    print("URDF Converter Configuration:")
    print_dict(urdf_converter_cfg.to_dict(), nesting=0)
    
    # Convert URDF to USD
    urdf_converter = UrdfConverter(urdf_converter_cfg)
    return urdf_converter.usd_path

def create_simple_scene_cfg(urdf_path: str) -> Type[InteractiveSceneCfg]:
    """Factory function to create a scene configuration class with a given URDF
    Args:
        urdf_path: Path to the URDF file to convert
    Returns:
        A class type that inherits from InteractiveSceneCfg
    """
    try:
        usd_path = importURDF(urdf_path, "assets/usd")
        print(f"Successfully converted URDF to {usd_path}")
    except Exception as e:
        print(f"Error importing URDF: {e}")
        raise

    @configclass    
    class SimpleSceneCfg(InteractiveSceneCfg):
        """Designs the scene in a modular way"""
        # NOTE: Add entities in the order we want them to be added to the scene
        # Ground Plane
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        
        # Scene Light
        dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )

        agent = configure_agent_minimally(usd_path, "{ENV_REGEX_NS}/Agent")

    return SimpleSceneCfg

@configclass
class ActionsCfg:
    """Action specifications for simple caterpillar environment"""
    # I want to start using forces, maybe? Still thinking about it.
    # Also, My agents are uniform but have different morphologies, I want the actions to be all efforts (torques) across their joints for training
    joint_efforts = mdp.JointEffortActionCfg(asset_name="agent", joint_names=[".*"], scale=0.5)

@configclass
class ObservationsCfg:
    """Observation specifications for the environment"""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group"""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
    
    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events"""

    # on startup
    randomize_agent_startup = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("agent", joint_names=["*."]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5,0.5),
        }
    )

    randomize_agent_reset = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("agent", joint_names=["*."]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5,0.5),
        }
    )

@configclass
class ManagerSceneCfg(ManagerBasedEnv):
    """Configuration for a manager based environment"""
    SceneCfg = create_simple_scene_cfg(args_cli.urdf)
    scene = SceneCfg(args_cli.num_envs, env_spacing=2.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        self.decimation = 4
        self.sim.dt = 0.005 # 200Hz


if __name__ == "__main__":
    if args_cli.wiggle:
        sim_cfg = SimulationCfg(device=args_cli.device)
        sim = SimulationContext(sim_cfg)
        sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
        
        # Create the scene config
        SimpleSceneCfg = create_simple_scene_cfg(args_cli.urdf)

        # Instantiate it
        scene_cfg = SimpleSceneCfg(args_cli.num_envs, env_spacing=2.0)
        scene = InteractiveScene(scene_cfg)

        sim.reset()
        print("Starting simulation. Ctrl+C to stop")
        runSimulationWiggle(sim, scene)
        simulation_app.close()
    else:
        runSimlation()
        simulation_app.close()