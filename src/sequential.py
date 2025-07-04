# NOT OUR CODE, FROM SKRL SOURCE CODE, MODIFICAITONS BELOW

from typing import List, Optional, Union

import copy
import sys
import tqdm

import wandb
import imageio
import os
import numpy as np

import torch

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.trainers.torch import Trainer


# fmt: off
# [start-config-dict-torch]
SEQUENTIAL_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,            # number of timesteps to train for
    "headless": False,              # whether to use headless mode (no rendering)
    "disable_progressbar": False,   # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,   # whether to close the environment on normal program termination
    "environment_info": "episode",       # key used to get and log environment info
    "stochastic_evaluation": False,      # whether to use actions rather than (deterministic) mean actions during evaluation
}
# [end-config-dict-torch]
# fmt: on

# Modification by SAFA, WORK IN PROGRESS. Figure out if this works.
def log_wandb_rollout(self, update_id: int):
    """
    Collect a video_length‐long rollout from env 0, save as GIF, and log to WandB.
    """
    pass

    # # 1) Make sure the camera exists
    # camera = self.env.unwrapped.scene.get("top_camera", None)
    # if camera is None or not hasattr(camera, "data") or "rgb" not in camera.data.output:
    #     print("[WARN] No camera or RGB output available; cannot record rollout.")
    #     return

    # # 2) Prepare a small buffer to store frames for env 0
    # images = []
    # video_length = self.cfg.get("video_length", 200)  # or pull from Trainer cfg

    # # 3) Run a short rollout from scratch (env.reset + video_length steps)
    # with torch.no_grad():
    #     obs, infos = self.env.reset()
    #     done_mask = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)

    #     for step_i in range(video_length):
    #         # Ask agent for actions
    #         action = None
    #         if isinstance(self.agents, list):
    #             # If multiple agents, pick the first for simplicity
    #             act_out = self.agents[0].act(obs, timestep=step_i, timesteps=self.timesteps)
    #             action = act_out[0]
    #         else:
    #             action = self.agents.act(obs, timestep=step_i, timesteps=self.timesteps)[0]

    #         # Step environment
    #         obs, reward, terminated, truncated, infos = self.env.step(action)

    #         # Grab the camera image for env 0 and append to list
    #         rgba = camera.data.output["rgb"]  # shape: (num_envs, H, W, 3), float32 in [0..1]
    #         frame0 = (rgba[0] * 255.0).clamp(0, 255).byte().cpu().numpy()  # → (H, W, 3) uint8
    #         images.append(frame0)

    #         # If all envs are done, break early
    #         if (terminated | truncated).all():
    #             break

    # # 4) If we didn't capture any frames, warn and exit
    # if len(images) == 0:
    #     print("[WARN] Captured 0 frames; skipping GIF creation.")
    #     return

    # # 5) Write the frames list to a GIF file
    # out_dir = self.cfg["experiment"]["directory"]
    # os.makedirs(out_dir, exist_ok=True)
    # gif_path = os.path.join(out_dir, f"rollout_{update_id}.gif")
    # try:
    #     imageio.mimsave(gif_path, images, fps=10)
    # except Exception as e:
    #     print(f"[WARN] Failed to write GIF to disk: {e}")
    #     return

    # # 6) Log to WandB
    # try:
    #     wandb.log({"rollout/gif": wandb.Image(gif_path)}, step=update_id)
    # except Exception as e:
    #     print(f"[WARN] WandB logging failed: {e}")




class SequentialTrainer(Trainer):
    def __init__(
        self,
        env: Wrapper,
        agents: Union[Agent, List[Agent]],
        agents_scope: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Sequential trainer

        Train agents sequentially (i.e., one after the other in each interaction with the environment)

        :param env: Environment to train on
        :type env: skrl.envs.wrappers.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: ``None``)
        :type agents_scope: tuple or list of int, optional
        :param cfg: Configuration dictionary (default: ``None``).
                    See SEQUENTIAL_TRAINER_DEFAULT_CONFIG for default values
        :type cfg: dict, optional
        """
        _cfg = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        agents_scope = agents_scope if agents_scope is not None else []
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)

        # init agents
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)

    def train(self) -> None:
        """Train the agents sequentially

        This method executes the following steps in loop:

        - Pre-interaction (sequentially)
        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Record transitions (sequentially)
        - Post-interaction (sequentially)
        - Reset environments
        """
        # set running mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("train")
        else:
            self.agents.set_running_mode("train")

        # non-simultaneous agents
        if self.num_simultaneous_agents == 1:
            # single-agent
            if self.env.num_agents == 1:
                self.single_agent_train()
            # multi-agent
            else:
                self.multi_agent_train()
            return

        # reset env
        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                actions = torch.vstack(
                    [
                        agent.act(states[scope[0] : scope[1]], timestep=timestep, timesteps=self.timesteps)[0]
                        for agent, scope in zip(self.agents, self.agents_scope)
                    ]
                )

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                for agent, scope in zip(self.agents, self.agents_scope):
                    agent.record_transition(
                        states=states[scope[0] : scope[1]],
                        actions=actions[scope[0] : scope[1]],
                        rewards=rewards[scope[0] : scope[1]],
                        next_states=next_states[scope[0] : scope[1]],
                        terminated=terminated[scope[0] : scope[1]],
                        truncated=truncated[scope[0] : scope[1]],
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.timesteps,
                    )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            for agent in self.agents:
                                agent.track_data(f"Info / {k}", v.item())

                # EDIT Log individual reward terms (if reward manager supports compute_individual)
                try:
                    reward_terms = self.env.unwrapped.reward_manager.compute_individual()
                    for key, val in reward_terms.items():
                        wandb.log({f"reward/{key}": val.mean().item()}, step=timestep)
                except Exception as e:
                    print(f"[WARN] Could not log individual rewards: {e}")

            # post-interaction
            for agent in self.agents:
                agent.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # EDIT Log rollout GIF every N iterations (based on rollouts, not timesteps)
            if timestep % (self.rollouts * 100) == 0:
                self.log_wandb_rollout(update_id=timestep // self.rollouts)

            # reset environments
            if terminated.any() or truncated.any():
                with torch.no_grad():
                    states, infos = self.env.reset()
            else:
                states = next_states

    def eval(self) -> None:
        """Evaluate the agents sequentially

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        # set running mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("eval")
        else:
            self.agents.set_running_mode("eval")

        # non-simultaneous agents
        if self.num_simultaneous_agents == 1:
            # single-agent
            if self.env.num_agents == 1:
                self.single_agent_eval()
            # multi-agent
            else:
                self.multi_agent_eval()
            return

        # reset env
        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                outputs = [
                    agent.act(states[scope[0] : scope[1]], timestep=timestep, timesteps=self.timesteps)
                    for agent, scope in zip(self.agents, self.agents_scope)
                ]
                actions = torch.vstack(
                    [
                        output[0] if self.stochastic_evaluation else output[-1].get("mean_actions", output[0])
                        for output in outputs
                    ]
                )

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                for agent, scope in zip(self.agents, self.agents_scope):
                    agent.record_transition(
                        states=states[scope[0] : scope[1]],
                        actions=actions[scope[0] : scope[1]],
                        rewards=rewards[scope[0] : scope[1]],
                        next_states=next_states[scope[0] : scope[1]],
                        terminated=terminated[scope[0] : scope[1]],
                        truncated=truncated[scope[0] : scope[1]],
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.timesteps,
                    )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            for agent in self.agents:
                                agent.track_data(f"Info / {k}", v.item())

            # post-interaction
            for agent in self.agents:
                super(type(agent), agent).post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if terminated.any() or truncated.any():
                with torch.no_grad():
                    states, infos = self.env.reset()
            else:
                states = next_states
