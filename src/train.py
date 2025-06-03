# Author: Safa Obuz
# Date: 6/2/2025
# Purpose: Full training loop

import argparse
import imageio
import os
import sys
from datetime import datetime

# ─── Parse CLI arguments ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train custom agent with skrl PPO")
parser.add_argument("--width", type=int, default=1280, help="Width of the viewport")
parser.add_argument("--height", type=int, default=720, help="Height of the viewport")
parser.add_argument("--num_envs", type=int, default=4000, help="Number of environments to spawn")
parser.add_argument("--urdf", type=str, default="assets/urdf/test.urdf", help="Path to URDF to load")
parser.add_argument("--start_height", nargs=2, type=float, default=[0.3, 0.31], help="Height at which to load the agent in environment")

# Training specific arguments
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum training iterations")
parser.add_argument("--checkpoint_interval", type=int, default=50, help="Save checkpoint every N iterations")
parser.add_argument("--log_interval", type=int, default=10, help="Log metrics every N iterations")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training")
parser.add_argument("--video_interval", type=int, default=100, help="Record video every N iterations")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded videos in steps")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
parser.add_argument("--experiment_name", type=str, default="custom_agent", help="Name for the experiment")

# Add IsaacLab AppLauncher args
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras if recording video
if args_cli.video:
    args_cli.enable_cameras = True

# Launch the app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import everything else after app launch
import torch
import numpy as np
from typing import Type
from pathlib import Path
import gymnasium as gym

# skrl imports
import skrl
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
#from sequential import SequentialTrainer
from skrl.trainers.torch.sequential import SequentialTrainer
from skrl.utils import set_seed
from skrl.envs.wrappers.torch import wrap_env
from skrl.resources.schedulers.torch import KLAdaptiveLR

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg
)
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.io import dump_pickle, dump_yaml
import isaaclab.envs.mdp as mdp
from isaaclab.sensors import CameraCfg, Camera

import omni.client
import omni.usd

import observations
import rewards

import wandb
from torch.utils.tensorboard import SummaryWriter

def importURDF(urdf_path: str, usd_dir: str) -> str:
    """
    Imports a URDF into an Isaac Lab Scene that is currently running.
    Returns the path to the newly created USD.
    """
    # Make paths absolute
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)
    if not os.path.isabs(usd_dir):
        usd_dir = os.path.abspath(usd_dir)

    forward_urdf = urdf_path.replace(os.sep, "/")
    forward_usd_dir = usd_dir.replace(os.sep, "/")

    # Check existence on Nucleus (or local)
    result, _ = omni.client.stat(forward_urdf)
    if result != omni.client.Result.OK:
        raise ValueError(f"URDF file not found: {urdf_path}")

    # Build USD filename
    urdf_filename = os.path.basename(urdf_path)
    usd_filename = os.path.splitext(urdf_filename)[0] + ".usd"
    os.makedirs(os.path.dirname(usd_dir), exist_ok=True)

    # Configure the URDF converter
    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=usd_dir,
        usd_file_name=usd_filename,
        fix_base=False,
        merge_fixed_joints=False,
        force_usd_conversion=True,
        root_link_name="torso",
        make_instanceable=True,
        self_collision=False,
        collision_from_visuals = True,
        link_density=1000.0,
        replace_cylinders_with_capsules=True,
        joint_drive=None,  # we'll rely on default joint‐drive settings
    )

    urdf_converter = UrdfConverter(urdf_converter_cfg)
    return urdf_converter.usd_path

def configure_agent(usd_path: str, prim_path: str) -> ArticulationCfg:
    """Return a basic ArticulationCfg that spawns from the given USD."""
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
        ),
        actuators={
            "joints": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)
        },
    )

def create_simple_scene_cfg(urdf_path: str) -> Type[InteractiveSceneCfg]:
    """
    Returns a new subclass of InteractiveSceneCfg which:
      - Converts the given URDF into USD (once)
      - Adds a ground plane, a dome light, and spawns the converted USD under /World
    """
    try:
        usd_path = importURDF(urdf_path, "assets/usd")
        print(f"Successfully converted URDF to: {usd_path}")
    except Exception as e:
        print(f"Error importing URDF: {e}")
        raise

    @configclass
    class SimpleSceneCfg(InteractiveSceneCfg):
        # (1) Ground plane
        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg()
        )
        # (2) Dome light
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )
        # (3) Our “agent” from the just-converted USD
        agent = configure_agent(usd_path, "{ENV_REGEX_NS}/Agent")

        # top_camera = CameraCfg(
        #     prim_path="/World/TopCamera",
        #     spawn=sim_utils.PinholeCameraCfg(focal_length=2.0),
        #     width=args_cli.width,
        #     height=args_cli.height,
        #     offset=CameraCfg.OffsetCfg(
        #         pos=(4.0, 0.0, 5.0),
        #         rot=(1.0, 0.0, 0.0, 0.0),
        #         convention="ros"
        #     ),
        #     data_types=["rgb"]
        # )
    
    return SimpleSceneCfg

@configclass
class ActionsCfg:
    joint_efforts = mdp.JointEffortActionCfg(
        asset_name="agent",
        joint_names=[".*"],
        scale=5.0
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # STATES OF AGENT
        # The agent sees how high it is in terms of the simulation world frame
        base_height = ObsTerm(func=mdp.base_pos_z, params={"asset_cfg": SceneEntityCfg("agent")})
        # The agent sees its linear velocity in terms of its URDF's root
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("agent")})
        # The agent sees its angular velocity in terms of its URDF's root
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("agent")})
        # The agent sees its own yaw and roll in terms of the environment frame as a quaternion
        base_yaw_roll = ObsTerm(func=observations.base_yaw_roll)
        # The agent sees its angle between its root forward vector and the target forward vector
        base_angle_to_target = ObsTerm(func=observations.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
        # The agent sees the projection of gravity on its root frame (1 means same)
        base_up_proj = ObsTerm(func=observations.base_up_proj)
        # The agent sees the projection of the direction it is facing onto the desired forward
        base_heading_proj = ObsTerm(func=observations.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
        # The agent sees the value of its joints normalized from -1 to 1
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized, params={"asset_cfg": SceneEntityCfg("agent")})
        # The agent sees its own joint velocities (rad/s)
        joint_vel = ObsTerm(func=mdp.joint_vel, scale=0.2, params={"asset_cfg": SceneEntityCfg("agent")})
        # And the agent can see the last action it took, as part of the MDP (can sense acceleration this way)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    reset_agent_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("agent"),
            "pose_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": tuple(args_cli.start_height),
                "roll": (-0.01, 0.01),
                "pitch": (-0.01, 0.01),
                "yaw": (-0.01, 0.01),
            },
            "velocity_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
                "roll": (-0.01, 0.01),
                "pitch": (-0.01, 0.01),
                "yaw": (-0.01, 0.01),
            },
        }
    )
    
    # reset_scene_to_default = EventTerm(
    #     func=mdp.reset_scene_to_default,
    #     mode="reset",
    #     params={},
    # )

    
def root_xyz_out_of_bounds(env, asset_cfg, bounds: list[tuple[float,float]]):
    root_world = mdp.root_pos_w(env, asset_cfg=asset_cfg)
    lows = torch.tensor([b[0] for b in bounds], device=root_world.device)
    highs = torch.tensor([b[1] for b in bounds], device=root_world.device)

    below_min = root_world < lows.unsqueeze(0)   # (num_envs,3) bool
    above_max = root_world > highs.unsqueeze(0)  # (num_envs,3) bool

    out_any_axis = (below_min | above_max).any(dim=1)
    return out_any_axis

def alive_term(env):
    return mdp.is_alive(env)

def terminating_term(env):
    return mdp.is_terminated(env)

def forward_vel_term(env, asset_cfg):
    return mdp.base_lin_vel(env, asset_cfg=asset_cfg)[:, 0]

def joint_vel_penalty_term(env, asset_cfg):
    return mdp.joint_vel_l1(env, asset_cfg=asset_cfg)

def exit_bonus_term(env, asset_cfg, bounds):
    mask = root_xyz_out_of_bounds(env, asset_cfg, bounds)
    return mask.to(dtype=torch.float32)

def spin_rate(env, axis: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("agent")) -> torch.Tensor:
    """Returns a penalty = |ang_rate| for each env, where ang_rate is  the X/Y/Z-component of the agent's angular velocity in world frame."""
    agent = env.scene[asset_cfg.name]
    # World‐frame angular velocity is a (num_envs, 3) tensor
    ang_vel: torch.Tensor = agent.data.root_ang_vel_w
    if axis == "x":
        ang_rate: torch.Tensor = ang_vel[:, 0]
    elif axis == "y":
        ang_rate: torch.Tensor = ang_vel[:, 1]
    else:
        ang_rate: torch.Tensor = ang_vel[:, 2]
    # Clip yaw rate to prevent extreme values
    ang_rate = torch.clamp(ang_rate, -10.0, 10.0)
    print(f"AGENT RATE ENV0: {ang_rate[0]}")
    return torch.abs(ang_rate)


def joint_motion_reward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("agent")) -> torch.Tensor:
    """Encourage the agent to move its joisnts by returning the sum of absolute joint velocities for each environment. Larger joint‐space movement → larger reward.
    """
    agent = env.scene[asset_cfg.name]
    # joint_vel has shape (num_envs, num_joints)
    joint_vel: torch.Tensor = agent.data.joint_vel  # (M, J)
    
    # Clip joint velocities to prevent extreme values
    joint_vel = torch.clamp(joint_vel, -50.0, 50.0)
    
    # Use L1 norm instead of squared to avoid explosion
    # Also apply a soft saturation using tanh
    joint_motion = torch.sum(torch.abs(joint_vel), dim=-1)
    #print(f"AGENT JOINT ENV0: {torch.tanh(joint_motion * 0.1)}")
    return torch.tanh(joint_motion * 0.1)  # Scale and saturate

@configclass
class RewardsCfg:
    progress = RewTerm(func=rewards.progress_reward, weight=15.0, params={"target_pos": (1000.0, 0.0, 0.0)})
    alive = RewTerm(func=mdp.is_alive, weight=-3.0)
    upright = RewTerm(func=rewards.upright_posture_bonus, weight=0.1, params={"threshold": 0.93})
    move_to_target = RewTerm(
        func=rewards.move_to_target_bonus, 
        weight=15.0, 
        params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)}
    )
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.5) #consider bumping
    #energy = RewTerm(func=rewards.power_consumption, weight=-0.05, params={"gear_ratio": {".*": 15.0}})
    # joint_pos_limits = RewTerm(
    #     func=rewards.joint_pos_limits_penalty_ratio, weight=-0.01, params={"threshold": 0.99, "gear_ratio": {".*": 15.0}}
    # )
    # spin_penalty_x = RewTerm(func=spin_rate, weight=-0.1, params={"axis": "x"})
    # spin_penalty_y = RewTerm(func=spin_rate, weight=-0.1, params={"axis": "y"})
    # spin_penalty_z = RewTerm(func=spin_rate, weight=-0.1, params={"axis": "z"})

    joint_motion = RewTerm(
        func=joint_motion_reward,
        weight=10.0,    # required field—scale this to adjust how strongly you reward motion
    )

    # terminating = RewTerm(
    #     func = terminating_term,
    #     weight = -2.0
    # )

    # exit_bonus = RewTerm(
    #     func = exit_bonus_term,
    #     params = {
    #         "asset_cfg": SceneEntityCfg("agent"),
    #         "bounds": [(-2.0, 2.0), (-2.0, 2.0), (-1.0, 10.0)]
    #     },
    #     weight = +10.0
    # )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(
        func=mdp.time_out,     # no params here
        time_out=True
    )
    # torso_height =  DoneTerm(func=mdp.root_height_below_minimum, params={"asset_cfg": SceneEntityCfg("agent"),
    #                                                                      "minimum_height": 0.1})

    # out_of_bounds = DoneTerm(
    #     func=root_xyz_out_of_bounds,
    #     params={
    #         "asset_cfg": SceneEntityCfg("agent"),
    #         "bounds": [(-2.0, 2.0), (-2.0, 2.0), (0.0, 10.0)],
    #     },
    #     time_out=False   # does not count as primary timeout
    # )

@configclass
class AgentCfg(ManagerBasedRLEnvCfg):
    _scene_cls = create_simple_scene_cfg(args_cli.urdf)
    scene: InteractiveSceneCfg = _scene_cls(num_envs=args_cli.num_envs, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    episode_length_s: float = 5.0   # must match max_time from term.time_out
    seed: int = 42

    def __post_init__(self):
        self.viewer.eye = [2.0, -2.3, 0.5]
        self.viewer.lookat = [0.1, -0.3, 0.0]
        self.decimation = 1
        self.sim.dt = 1 / 240
        if hasattr(args_cli, "device"):
            self.sim.device = args_cli.device
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0


# ─── Routine: Run the ManagerBasedRLEnv ────────────────────────────────────────
def run_modular_morphology_sim(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Instantiate the env, step with random actions, and print some outputs."""

    # Override the number of envs & device at runtime
    env_cfg.scene.num_envs = args_cli.num_envs
    if hasattr(args_cli, "device"):
        env_cfg.sim.device = args_cli.device

    # Create the environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Do one reset so torch tensors are built
    obs_dict, _ = env.reset()
    print("Running", env.num_envs, "envs")
    print("Policy obs_dict dim:", obs_dict["policy"].shape)
    print("Action dim:", env.action_manager.total_action_dim)

    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            if count % 200 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environments...")

            # Sample random joint controls
            actions = torch.randn(
                env.num_envs,
                env.action_manager.total_action_dim,
                device=env.device
            )

            obs, rewards, terminated, truncated, extras = env.step(actions)

            if count % 50 == 0:
                # Print out the “morphology encoding” portion of your observations
                # (Assuming your MDP’s “policy” obs includes that as the first dims)
                for e in range(env.num_envs):
                    print(f"env{e} morph:", obs["policy"][e])

            count += 1

    env.close()

class Policy(GaussianMixin, Model):
    """Actor network for PPO"""
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, self.num_actions)
        )
        
        self.log_std_parameter = torch.nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # Policy network forward pass
        return self.net(inputs["states"]), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    """Critic network for PPO"""
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 1)  # This is different than Ant SKRL PPO config? No, output: ONE
        )

    def compute(self, inputs, role):
        # Value network forward pass
        return self.net(inputs["states"]), {}


timestamp = datetime.now().strftime("%d_%H-%M-%S")

def train_custom_agent():
    """Main training function"""
    
    # Set random seed
    set_seed(args_cli.seed)

    # Create environment config
    env_cfg = AgentCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if hasattr(args_cli, "device"):
        env_cfg.sim.device = args_cli.device
    
    log_dir = os.path.join("logs", f"{args_cli.experiment_name}_{timestamp}_PPO")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

        
    wandb.init(
        project="custom-agent-training",
        group="low-vals",
        name=f"{args_cli.experiment_name}",
        dir=log_dir,
        sync_tensorboard=True,
        config={
            "num_envs": args_cli.num_envs,
            "learning_rate": 3e-4,
            "discount_factor": 0.99,
            "rollouts": 16,
            "episode_length": 5.0,
        }
    )
    writer = SummaryWriter(log_dir)

    # Save configs
    # os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    # dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    #dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    
    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Wrap environment for video recording if requested
    # if args_cli.video:
    #     from gymnasium.wrappers import RecordVideo
    #     video_kwargs = {
    #         "video_folder": os.path.join(log_dir, "videos"),
    #         "step_trigger": lambda step: step % (args_cli.video_interval * env.max_episode_length) == 0,
    #         "video_length": args_cli.video_length,
    #         "disable_logger": True,
    #     }
    #     env = RecordVideo(env, **video_kwargs)
    #     print("[INFO] Video recording enabled")
    
    # Wrap environment for skrl
    env = wrap_env(env)
    device = env.device
    
    # Create PPO agent configuration
    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg["rollouts"] = 16  # Number of rollouts before each update
    ppo_cfg["learning_epochs"] = 8
    ppo_cfg["mini_batches"] = 4
    ppo_cfg["discount_factor"] = 0.99
    ppo_cfg["lambda"] = 0.95
    ppo_cfg["learning_rate"] = 3e-4
    ppo_cfg["learning_rate_scheduler"] = KLAdaptiveLR
    ppo_cfg["learning_rate_scheduler_kwargs"] = {
        "kl_threshold": 0.01,
        "min_lr": 1e-5,
    }
    ppo_cfg["random_timesteps"] = 0
    ppo_cfg["learning_starts"] = 0
    ppo_cfg["grad_norm_clip"] = 1.0
    ppo_cfg["ratio_clip"] = 0.2
    ppo_cfg["value_clip"] = 0.2
    ppo_cfg["clip_predicted_values"] = True
    ppo_cfg["entropy_loss_scale"] = 0.01
    ppo_cfg["value_loss_scale"] = 0.5
    ppo_cfg["kl_threshold"] = 0.01
    ppo_cfg["state_preprocessor"] = None
    ppo_cfg["state_preprocessor_kwargs"] = {}
    ppo_cfg["value_preprocessor"] = None
    ppo_cfg["value_preprocessor_kwargs"] = {}
    ppo_cfg["experiment"] = {
        "directory": log_dir,
        "experiment_name": "sub_folder",
        "write_interval": 1 , #"auto",
        "checkpoint_interval": 10, #"auto",
        # "wandb": True,
        # "wandb_kwargs": {
        #     "project": "custom-agent-training",
        #     "name": run_name,
        #     "dir": log_sub,
        #     "sync_tensorboard": True,
        #     "config": {
        #         "num_envs": env.num_envs,
        #         "learning_rate": ppo_cfg["learning_rate"],
        #         "discount_factor": ppo_cfg["discount_factor"],
        #         "rollouts": ppo_cfg["rollouts"],
        #         "episode_length": env_cfg.episode_length_s,
        #     }
        # }
    }
    
    # Create memory
    memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)
    
    # Create models
    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, device)
    models["value"] = Value(env.observation_space, env.action_space, device)

    wandb.watch(models["policy"], log="all", log_freq=100)

    
    # Initialize model weights properly
    # for model in models.values():
    #     for layer in model.net:
    #         if isinstance(layer, torch.nn.Linear):
    #             torch.nn.init.orthogonal_(layer.weight, gain=1.0)
    #             if layer.bias is not None:
    #                 torch.nn.init.zeros_(layer.bias)

    # Create PPO agent
    agent = PPO(
        models=models,
        memory=memory,
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )

    
    # Load checkpoint if provided
    if args_cli.checkpoint:
        print(f"[INFO] Loading checkpoint from: {args_cli.checkpoint}")
        agent.load(args_cli.checkpoint)
    
    # Configure trainer
    trainer_cfg = {
        "timesteps": args_cli.max_iterations * ppo_cfg["rollouts"],
        "headless": args_cli.headless,
        "disable_progressbar": False,
        "close_environment_at_exit": False,
        "video_logging": args_cli.video,  
    }
    
    # Create trainer
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
    
    # Start training
    print(f"[INFO] Starting training for {trainer_cfg['timesteps']} timesteps")
    print(f"[INFO] Rollouts: {ppo_cfg['rollouts']}, Num envs: {env.num_envs}")
    print(f"[INFO] Max iterations: {args_cli.max_iterations}")
    
    trainer.train()

    # Final rollout recording
    # trainer.log_wandb_rollout(update_id=args_cli.max_iterations)

    
    # Save final model
    final_checkpoint_path = os.path.join(log_dir, "final_model.pt")
    agent.save(final_checkpoint_path)
    print(f"[INFO] Saved final model to: {final_checkpoint_path}")
    
    # Close environment
    env.close()


# ─── Play/evaluation function ──────────────────────────────────────────────────
def play_custom_agent(checkpoint_path: str, num_episodes: int = 10):
    """Play trained agent"""
    
    # Create environment config
    env_cfg = AgentCfg()
    env_cfg.scene.num_envs = 5  # Single environment for visualization
    if hasattr(args_cli, "device"):
        env_cfg.sim.device = args_cli.device
    
    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = wrap_env(env)
    device = env.device
    
    # Create models
    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, device)
    models["value"] = Value(env.observation_space, env.action_space, device)
    
    # Create agent (minimal config for inference)
    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg["experiment"]["write_interval"] = 0  # No logging during play
    
    agent = PPO(
        models=models,
        memory=None,  # No memory needed for inference
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )
    
    # Load checkpoint
    print(f"[INFO] Loading model from: {checkpoint_path}")
    agent.load(checkpoint_path)
    agent.set_running_mode("eval")
    
    # Run episodes
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\n[INFO] Episode {episode + 1}/{num_episodes}")
        
        while not done:
            # Get action from policy
            with torch.no_grad():
                action = agent.act(obs, timestep=0, timesteps=0)[0]
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated.any() or truncated.any()
            
            total_reward += reward.mean().item()
            steps += 1
            
        print(f"  Total reward: {total_reward:.2f}, Steps: {steps}")
    
    env.close()


# ─── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Check if we're in play mode
    if args_cli.checkpoint:
        # Play mode
        play_custom_agent(args_cli.checkpoint, num_episodes=5)
    else:
        # Training mode
        train_custom_agent()
    
    # Close simulation
    simulation_app.close()
